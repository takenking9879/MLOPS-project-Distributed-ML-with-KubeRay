from __future__ import annotations

import os
import tempfile
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import xgboost

logger_std = logging.getLogger(__name__)


def metrics_from_confusion_np(conf) -> Dict[str, float]:
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
        "val_accuracy": accuracy,
        "val_precision_macro": macro_precision,
        "val_recall_macro": macro_recall,
        "val_f1_macro": macro_f1,
        "val_precision_weighted": weighted_precision,
        "val_recall_weighted": weighted_recall,
        "val_f1_weighted": weighted_f1,
    }

    for i in range(conf.shape[0]):
        metrics[f"val_precision_class_{i}"] = float(precision[i])
        metrics[f"val_recall_class_{i}"] = float(recall[i])
        metrics[f"val_f1_class_{i}"] = float(f1[i])
        metrics[f"val_support_class_{i}"] = float(support[i])

    return metrics


def xgb_multiclass_metrics_on_val(
    *,
    val_ds,
    target: str,
    num_classes: int,
    booster_checkpoint,
) -> Dict[str, float]:
    """Compute multiclass metrics for XGBoost on a Ray Dataset validation split.

    This avoids collecting the full dataset to the driver by aggregating a confusion matrix.
    """

    try:
        booster = booster_checkpoint.get_model()
        model_bytes = booster.save_raw()

        def predict_batch(df: "pd.DataFrame") -> "pd.DataFrame":
            y_true = df[target].astype("int64").to_numpy()
            X = df.drop(columns=[target])

            # Load model from bytes inside the worker
            tmp = tempfile.NamedTemporaryFile(delete=False)
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

            return pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

        pred_ds = val_ds.map_batches(predict_batch, batch_format="pandas")
        counts = pred_ds.groupby(["y_true", "y_pred"]).count()
        rows = counts.take_all()

        conf = np.zeros((num_classes, num_classes), dtype=np.int64)
        for r in rows:
            ti = int(r["y_true"])
            pi = int(r["y_pred"])
            c = int(r["count()"])
            if 0 <= ti < num_classes and 0 <= pi < num_classes:
                conf[ti, pi] = c

        return metrics_from_confusion_np(conf)

    except Exception as e:
        logger_std.error(
            f"Error calculando métricas multiclass de XGBoost: {str(e)}",
            exc_info=True,
        )
        return {}
