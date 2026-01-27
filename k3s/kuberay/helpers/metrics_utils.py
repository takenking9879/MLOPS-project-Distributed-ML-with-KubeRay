from __future__ import annotations

import os
import tempfile
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import xgboost
from ray.train.xgboost import RayTrainReportCallback
from sklearn.metrics import classification_report

logger_std = logging.getLogger(__name__)


def metrics_from_confusion_np(conf, *, prefix: str = "val") -> Dict[str, float]:
    """Compute classification-report-like metrics from a confusion matrix.

    Expected conf shape: [C, C] where rows=true labels, cols=pred labels.
    """
    conf = np.asarray(conf, dtype=np.int64)
    support = conf.sum(axis=1)
    tp = np.diag(conf)
    pred_sum = conf.sum(axis=0)

    precision = np.divide(tp, np.maximum(pred_sum, 1), dtype=np.float64)
    recall = np.divide(tp, np.maximum(support, 1), dtype=np.float64)
    f1 = np.divide(2 * precision * recall, np.maximum(precision + recall, 1e-12), dtype=np.float64)

    accuracy = float(tp.sum() / max(conf.sum(), 1))
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    weights = support / max(support.sum(), 1)
    weighted_precision = float(np.sum(precision * weights))
    weighted_recall = float(np.sum(recall * weights))
    weighted_f1 = float(np.sum(f1 * weights))

    metrics: Dict[str, float] = {
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_precision_macro": macro_precision,
        f"{prefix}_recall_macro": macro_recall,
        f"{prefix}_f1_macro": macro_f1,
        f"{prefix}_precision_weighted": weighted_precision,
        f"{prefix}_recall_weighted": weighted_recall,
        f"{prefix}_f1_weighted": weighted_f1,
    }

    for i in range(conf.shape[0]):
        metrics[f"{prefix}_precision_class_{i}"] = float(precision[i])
        metrics[f"{prefix}_recall_class_{i}"] = float(recall[i])
        metrics[f"{prefix}_f1_class_{i}"] = float(f1[i])
        metrics[f"{prefix}_support_class_{i}"] = float(support[i])

    return metrics


def xgb_multiclass_metrics_on_ds(
    *,
    ds,
    split: str,
    target: str,
    num_classes: int,
    booster_checkpoint,
) -> Dict[str, Any]:
    """Compute multiclass metrics for XGBoost on a Ray Dataset split.

    This avoids collecting the full dataset to the driver by aggregating a confusion matrix.
    """

    try:
        # Ray Train stores XGBoost models inside a generic `ray.train.Checkpoint`.
        # Per Ray docs, use RayTrainReportCallback.get_model(checkpoint) to load it.
        booster = RayTrainReportCallback.get_model(booster_checkpoint)
        model_bytes = booster.save_raw()

        # NOTE: The previous implementation used `groupby(...).count()` which forces
        # a shuffle + hash aggregate (slow for small/medium datasets on Kubernetes).
        # Instead, compute a confusion matrix per batch and reduce on the driver.

        def predict_and_cm_batch(df: "pd.DataFrame") -> "pd.DataFrame":
            y_true = df[target].astype("int64").to_numpy()
            X = df.drop(columns=[target])

            # Load model from bytes inside the worker
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ubj")
            try:
                tmp.write(model_bytes)
                tmp.close()
                b = xgboost.Booster()
                b.load_model(tmp.name)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception as e:
                    logger_std.debug(
                        "No se pudo borrar archivo temporal %s: %s",
                        tmp.name,
                        str(e),
                        exc_info=True,
                    )

            dmat = xgboost.DMatrix(X)
            probs = b.predict(dmat)
            if probs.ndim == 1:
                y_pred = (probs > 0.5).astype("int64")
            else:
                y_pred = probs.argmax(axis=1).astype("int64")

            # Compute confusion matrix counts for this batch.
            # Vectorized bincount avoids Python loops.
            mask = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
            yt = y_true[mask].astype(np.int64, copy=False)
            yp = y_pred[mask].astype(np.int64, copy=False)
            idx = yt * num_classes + yp
            cm = np.bincount(idx, minlength=num_classes * num_classes).reshape((num_classes, num_classes))

            # One row per batch: store flattened counts.
            return pd.DataFrame({"cm": [cm.ravel().tolist()]})

        cm_rows = ds.map_batches(predict_and_cm_batch, batch_format="pandas").take_all()
        conf = np.zeros((num_classes, num_classes), dtype=np.int64)
        for r in cm_rows:
            flat = np.asarray(r["cm"], dtype=np.int64)
            if flat.size != num_classes * num_classes:
                continue
            conf += flat.reshape((num_classes, num_classes))

        out: Dict[str, Any] = metrics_from_confusion_np(conf, prefix=split)
        out["confusion_matrix"] = conf.tolist()

        # Build y_true/y_pred for sklearn classification_report.
        # If very large, sample pairs from confusion matrix distribution.
        try:
            total = int(conf.sum())
            max_rows = int(os.getenv("MLFLOW_CLASSIFICATION_REPORT_MAX_ROWS", "200000"))
            seed = int(os.getenv("SEED", "42"))
            flat = conf.ravel()
            if total > 0:
                if max_rows > 0 and total > max_rows:
                    rng = np.random.default_rng(seed)
                    p = flat / max(float(total), 1.0)
                    sampled = rng.multinomial(max_rows, p)
                    idx = np.repeat(np.arange(flat.size, dtype=np.int64), sampled)
                else:
                    idx = np.repeat(np.arange(flat.size, dtype=np.int64), flat)

                y_true = (idx // num_classes).astype(np.int64)
                y_pred = (idx % num_classes).astype(np.int64)
                out["classification_report"] = classification_report(
                    y_true,
                    y_pred,
                    labels=list(range(num_classes)),
                    digits=4,
                    zero_division=0,
                )
        except Exception as e:
            logger_std.warning(
                "No se pudo generar classification_report para XGBoost: %s",
                str(e),
                exc_info=True,
            )

        return out

    except Exception as e:
        logger_std.error(
            f"Error calculando métricas multiclass de XGBoost: {str(e)}",
            exc_info=True,
        )
        return {}


def xgb_multiclass_metrics_on_val(
    *,
    val_ds,
    target: str,
    num_classes: int,
    booster_checkpoint,
) -> Dict[str, Any]:
    """Backward-compatible wrapper (validation split)."""

    return xgb_multiclass_metrics_on_ds(
        ds=val_ds,
        split="val",
        target=target,
        num_classes=num_classes,
        booster_checkpoint=booster_checkpoint,
    )
