"""
PyTorch training module using Ray Train + Ray Tune.
Supports cheap HPT with ASHA + ResourceChangingScheduler.
"""

import os
import torch
import ray
from ray import tune
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from ray.tune import RunConfig
from ray.tune.schedulers import ASHAScheduler, ResourceChangingScheduler
from torch import nn
from typing import Dict

from schemas.pytorch_params import SEARCH_SPACE_PYTORCH_PARAMS, PYTORCH_TUNE_SETTINGS
from helpers.pytorch_utils import train_func
from ray.air.integrations.mlflow import MLflowLoggerCallback


# --------------------------
# Tuning entrypoint
# --------------------------
def tune_model(
    train_dataset,
    val_dataset,
    target,
    storage_path,
    name,
    num_classes: int = 6,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
):
    """
    config: dict con 'max_epochs' para ASHA
    """

    num_workers = int(os.getenv("NUM_WORKERS_TUNE", os.getenv("NUM_WORKERS", 2)))
    cpus_per_worker = int(os.getenv("CPUS_PER_WORKER_TUNE", os.getenv("CPUS_PER_WORKER", 1)))

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        resources_per_worker={"CPU": cpus_per_worker},
        use_gpu=torch.cuda.is_available(),
    )

    # --- Hyperparameter search space (cheap tuning) ---
    train_loop_config = {
        "target": target,
        "pytorch_params": SEARCH_SPACE_PYTORCH_PARAMS,
        "input_dim": 14,  # Ajustado a las columnas de preprocessing_001.py (3 cat + 11 num)
        "num_classes": int(num_classes),
    }

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        datasets={"train": train_dataset, "val": val_dataset},
    )

    # --- Early stopping scheduler ---
    asha = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=PYTORCH_TUNE_SETTINGS["max_epochs"],  # <-- sincronizado con el config
        grace_period=PYTORCH_TUNE_SETTINGS["grace_period"],
        reduction_factor=PYTORCH_TUNE_SETTINGS["reduction_factor"],
    )

    # --- Dynamic CPU allocation (optional) ---
    # ResourceChangingScheduler can request larger resource bundles over time.
    # In small clusters this may produce "infeasible resource requests" warnings.
    enable_rcs = os.getenv("ENABLE_RESOURCE_CHANGING_SCHEDULER", "false").lower() in ("1", "true", "yes")
    scheduler = ResourceChangingScheduler(base_scheduler=asha) if enable_rcs else asha

    # NOTE:
    # Ray Tune requires the trainable to be serializable (picklable). Use
    # `trainer.as_trainable()` to avoid pickling the Trainer instance itself.
    if not hasattr(trainer, "as_trainable"):
        raise TypeError(
            "Ray Train Trainer is not serializable for Tune in this Ray version. "
            "Expected `trainer.as_trainable()` to exist."
        )
    trainable = trainer.as_trainable()

    callbacks = []
    if mlflow_tracking_uri and mlflow_experiment_name:
        callbacks.append(
            MLflowLoggerCallback(
                tracking_uri=mlflow_tracking_uri,
                experiment_name=mlflow_experiment_name,
                save_artifact=False,
                log_params_on_trial_end=True,
            )
        )

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            num_samples=5,
            scheduler=scheduler,
            metric="val_loss",
            mode="min",
            max_concurrent_trials=int(os.getenv("MAX_CONCURRENT_TRIALS", "1")),
        ),
        run_config=RunConfig(storage_path=storage_path, name=name, callbacks=callbacks),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="val_loss", mode="min")
    return best.config