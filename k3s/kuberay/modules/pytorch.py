import os
import torch
import ray.train
from ray.train.torch import TorchTrainer
from torch import nn
from typing import Dict
from schemas.pytorch_params import PYTORCH_PARAMS 
from helpers.pytorch_utils import train_func

# =========================
# Model Definition
# =========================
# Train loop is shared from `pytorch_models.train_loop.train_func`


# =========================
# Trainer
# =========================
def train(train_dataset, val_dataset, target, storage_path, name, num_classes: int = 6, pytorch_params=None):
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

    result = trainer.fit()

    final_metrics: Dict[str, float] = {}
    try:
        if getattr(result, "metrics", None):
            for k, v in result.metrics.items():
                if isinstance(v, (int, float)):
                    final_metrics[k] = float(v)
    except Exception:
        pass

    return result, final_metrics
