from typing import Dict

import os
import logging

import ray
import ray.train
import torch
from torch import nn

from pytorch_models.models import NeuralNetwork

logger = logging.getLogger(__name__)


def _metrics_from_confusion(conf: torch.Tensor) -> Dict[str, float]:
    # conf shape: [C, C] where rows=true, cols=pred
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

    metrics: Dict[str, float] = {
        "val_accuracy": float(accuracy.item()),
        "val_precision_macro": float(macro_precision.item()),
        "val_recall_macro": float(macro_recall.item()),
        "val_f1_macro": float(macro_f1.item()),
        "val_precision_weighted": float(weighted_precision.item()),
        "val_recall_weighted": float(weighted_recall.item()),
        "val_f1_weighted": float(weighted_f1.item()),
    }

    # Per-class metrics (similar to classification_report())
    for i in range(conf.shape[0]):
        metrics[f"val_precision_class_{i}"] = float(precision[i].item())
        metrics[f"val_recall_class_{i}"] = float(recall[i].item())
        metrics[f"val_f1_class_{i}"] = float(f1[i].item())
        metrics[f"val_support_class_{i}"] = float(support[i].item())

    return metrics


def train_func(config: Dict):
    # Ray defaults to OMP_NUM_THREADS=1 unless the task/actor sets num_cpus.
    # For Ray Train workers, we pass the intended CPU budget via train_loop_config.
    try:
        cpus_per_worker = int(config.get("cpus_per_worker", os.getenv("CPUS_PER_WORKER", "1")))
    except Exception:
        cpus_per_worker = 1
    cpus_per_worker = max(cpus_per_worker, 1)

    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(cpus_per_worker)

    torch.set_num_threads(cpus_per_worker)
    # Inter-op threads >2 often hurts on small CPU pods.
    try:
        torch.set_num_interop_threads(min(2, cpus_per_worker))
    except Exception as e:
        logger.warning(
            "[pytorch_utils] No se pudo setear torch.set_num_interop_threads(%s): %s",
            str(min(2, cpus_per_worker)),
            str(e),
            exc_info=True,
        )

    params = config["pytorch_params"]
    batch_size = params.get("batch_size", 64)
    lr = params.get("lr", 1e-3)
    weight_decay = params.get("weight_decay", 0)
    max_epochs = params.get("max_epochs", 10)
    target = config.get("target", "attack")

    train_shard = ray.train.get_dataset_shard("train")
    val_shard = ray.train.get_dataset_shard("val")

    model = NeuralNetwork(
        input_dim=config.get("input_dim", 28 * 28),
        num_classes=config.get("num_classes", 10),
    )
    model = ray.train.torch.prepare_model(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Log actual CPU configuration for debugging
    if ray.train.get_context().get_world_rank() == 0:
        print(f"[pytorch_utils] Worker using {cpus_per_worker} CPU thread(s) | "
              f"torch.get_num_threads()={torch.get_num_threads()}")

    for epoch in range(max_epochs):
        model.train()
        # prefetch_batches > 1 enables async data loading using background threads
        train_loader = train_shard.iter_torch_batches(
            batch_size=batch_size, 
            dtypes=torch.float32,
            prefetch_batches=max(2, cpus_per_worker // 2),  # Async prefetch
        )
        train_loss, train_batches = 0.0, 0
        for batch in train_loader:
            # Separar target de features dinámicamente
            y = batch.pop(target).long()
            # X son todas las columnas restantes concatenadas
            feature_cols = sorted(batch.keys())
            X = torch.stack([batch[c] for c in feature_cols], dim=1)
            
            optimizer.zero_grad()
            preds = model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        avg_train_loss = train_loss / max(train_batches, 1)

        model.eval()
        val_loader = val_shard.iter_torch_batches(
            batch_size=batch_size, 
            dtypes=torch.float32,
            prefetch_batches=max(2, cpus_per_worker // 2),  # Async prefetch
        )
        val_loss, val_batches = 0.0, 0
        num_classes = int(config.get("num_classes", 2))
        conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        with torch.no_grad():
            for batch in val_loader:
                y = batch.pop(target).long()
                feature_cols = sorted(batch.keys())
                X = torch.stack([batch[c] for c in feature_cols], dim=1)
                
                preds = model(X)
                loss = loss_fn(preds, y)
                val_loss += loss.item()
                val_batches += 1
                y_pred = preds.argmax(dim=1)
                for t, p in zip(y, y_pred):
                    ti = int(t.item())
                    pi = int(p.item())
                    if 0 <= ti < num_classes and 0 <= pi < num_classes:
                        conf[ti, pi] += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        metrics = _metrics_from_confusion(conf)

        report = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            **metrics,
        }
        ray.train.report(report)
