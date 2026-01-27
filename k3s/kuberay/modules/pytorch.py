import os
import time
import logging
import torch
import numpy as np
import ray.train
from ray.train.torch import TorchTrainer
from torch import nn
from typing import Any, Dict
from schemas.pytorch_params import PYTORCH_PARAMS 
from helpers.pytorch_utils import train_func
from pytorch_models.models import NeuralNetwork
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)


def _metrics_from_confusion(conf: torch.Tensor, *, prefix: str) -> Dict[str, float]:
    conf = conf.to(torch.float32)
    support = conf.sum(dim=1)
    tp = torch.diag(conf)
    pred_sum = conf.sum(dim=0)

    precision = tp / torch.clamp(pred_sum, min=1.0)
    recall = tp / torch.clamp(support, min=1.0)
    f1 = (2.0 * precision * recall) / torch.clamp(precision + recall, min=1e-12)

    accuracy = tp.sum() / torch.clamp(conf.sum(), min=1.0)
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    weights = support / torch.clamp(support.sum(), min=1.0)
    weighted_precision = (precision * weights).sum()
    weighted_recall = (recall * weights).sum()
    weighted_f1 = (f1 * weights).sum()

    return {
        f"{prefix}_accuracy": float(accuracy.item()),
        f"{prefix}_precision_macro": float(macro_precision.item()),
        f"{prefix}_recall_macro": float(macro_recall.item()),
        f"{prefix}_f1_macro": float(macro_f1.item()),
        f"{prefix}_precision_weighted": float(weighted_precision.item()),
        f"{prefix}_recall_weighted": float(weighted_recall.item()),
        f"{prefix}_f1_weighted": float(weighted_f1.item()),
    }


def _classification_report_from_confusion(conf_np: np.ndarray, *, num_classes: int) -> str | None:
    total = int(conf_np.sum())
    if total <= 0:
        return None

    max_rows = int(os.getenv("MLFLOW_CLASSIFICATION_REPORT_MAX_ROWS", "200000"))
    seed = int(os.getenv("SEED", "42"))
    flat = conf_np.ravel()

    if max_rows > 0 and total > max_rows:
        rng = np.random.default_rng(seed)
        p = flat / max(float(total), 1.0)
        sampled = rng.multinomial(max_rows, p)
        idx = np.repeat(np.arange(flat.size, dtype=np.int64), sampled)
    else:
        idx = np.repeat(np.arange(flat.size, dtype=np.int64), flat)

    y_true = (idx // num_classes).astype(np.int64)
    y_pred = (idx % num_classes).astype(np.int64)
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        digits=4,
        zero_division=0,
    )


def _evaluate_on_dataset(
    *,
    checkpoint: ray.train.Checkpoint,
    ds,
    target: str,
    num_classes: int,
    input_dim: int,
    batch_size: int,
) -> Dict[str, Any]:
    # Load model weights from the final Ray Train checkpoint.
    model = NeuralNetwork(input_dim=input_dim, num_classes=num_classes)
    model.eval()

    with checkpoint.as_directory() as ckpt_dir:
        model_path = os.path.join(ckpt_dir, "model.pt")
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])

    loss_fn = nn.CrossEntropyLoss()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_loss = 0.0
    total_batches = 0

    feature_cols = None
    for batch in ds.iter_torch_batches(batch_size=batch_size, dtypes=torch.float32):
        y = batch.pop(target).long()
        if feature_cols is None:
            feature_cols = sorted(batch.keys())
        X = torch.stack([batch[c] for c in feature_cols], dim=1)
        with torch.no_grad():
            logits = model(X)
            total_loss += float(loss_fn(logits, y).item())
            total_batches += 1
            y_pred = logits.argmax(dim=1)
            idx = y * num_classes + y_pred
            counts = torch.bincount(idx, minlength=num_classes * num_classes)
            conf += counts.reshape(num_classes, num_classes)

    metrics: Dict[str, Any] = {}
    metrics["test_loss"] = total_loss / max(total_batches, 1)
    metrics.update(_metrics_from_confusion(conf, prefix="test"))

    conf_np = conf.detach().cpu().numpy().astype(np.int64)
    metrics["test_confusion_matrix"] = conf_np.tolist()
    cr = _classification_report_from_confusion(conf_np, num_classes=num_classes)
    if cr is not None:
        metrics["test_classification_report"] = cr

    return metrics

# =========================
# Model Definition
# =========================
# Train loop is shared from `pytorch_models.train_loop.train_func`


# =========================
# Trainer
# =========================
def train(train_dataset, val_dataset, test_dataset, target, storage_path, name, num_classes: int = 6, pytorch_params=None):
    scaling_config = ray.train.ScalingConfig(
        num_workers=int(os.getenv("NUM_WORKERS", 2)),
        resources_per_worker={"CPU": int(os.getenv("CPUS_PER_WORKER", 2))},
        use_gpu=torch.cuda.is_available(),
    )

    cpus_per_worker = int(os.getenv("CPUS_PER_WORKER", 2))

    params = pytorch_params if pytorch_params is not None else PYTORCH_PARAMS
    config = {
        "target": target,
        "pytorch_params": params,
        "input_dim": 14,  # Ajustado a las columnas de preprocessing_001.py (3 cat + 11 num)
        "num_classes": int(num_classes),
        "cpus_per_worker": cpus_per_worker,
        }

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
        scaling_config=scaling_config,
        datasets={"train": train_dataset, "val": val_dataset},
        run_config=ray.train.RunConfig(storage_path=storage_path, name=name),
        )

    start_time = time.perf_counter()
    result = trainer.fit()
    train_time_sec = time.perf_counter() - start_time
    print(f"[pytorch] distributed train_time_sec={train_time_sec:.2f}")

    final_metrics: Dict[str, Any] = {}
    try:
        if getattr(result, "metrics", None):
            for k, v in result.metrics.items():
                if isinstance(v, (int, float)):
                    final_metrics[k] = float(v)
    except Exception as e:
        logger.warning(
            "[pytorch] No se pudieron extraer métricas numéricas de result.metrics: %s",
            str(e),
            exc_info=True,
        )

    # Evaluate on full test dataset (driver-side), like XGBoost.
    final_metrics.update(
        _evaluate_on_dataset(
            checkpoint=result.checkpoint,
            ds=test_dataset,
            target=target,
            num_classes=int(num_classes),
            input_dim=int(config.get("input_dim", 14)),
            batch_size=int(params.get("batch_size", 256)),
        )
    )


    final_metrics["train_time_sec"] = train_time_sec

    return result, final_metrics
