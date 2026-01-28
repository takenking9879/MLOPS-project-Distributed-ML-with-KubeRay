"""
PyTorch training module using Ray Train + Ray Tune.
Supports cheap HPT with ASHA + ResourceChangingScheduler.
"""

import os
import numbers
import torch
import ray
from ray import tune
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from ray.tune import RunConfig
from ray.tune.schedulers import ASHAScheduler, ResourceChangingScheduler
from typing import Dict

from schemas.pytorch_params import SEARCH_SPACE_PYTORCH_PARAMS, PYTORCH_TUNE_SETTINGS
from helpers.pytorch_utils import train_func
from ray.air.integrations.mlflow import MLflowLoggerCallback


# --------------------------
# Tuning entrypoint
# --------------------------
def tune_model(
    train_path: str,
    val_path: str,
    target,
    storage_path,
    name,
    num_classes: int = 6,
    sample_fraction: float | None = None,
    seed: int = 42,
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
    # Keep key aligned with main.py expectations (best_config.get("pytorch_params")).
    param_space = {
        "pytorch_params": SEARCH_SPACE_PYTORCH_PARAMS,
    }

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
    # Do NOT pass Ray Datasets via `tune.with_parameters(...)`.
    # Tune stores those params via `ray.put`, and Ray Datasets are not picklable
    # under Ray 2.52 in that code path.
    # Workaround: pass dataset *paths* and load datasets inside each trial.

    def _maybe_sample_train_ds(ds: ray.data.Dataset) -> ray.data.Dataset:
        if sample_fraction is None:
            return ds
        frac = float(sample_fraction)
        if frac >= 1.0:
            return ds
        if frac <= 0.0:
            return ds

        # Avoid groupby/map_groups here because it triggers shuffles that request the
        # implicit `memory` resource, which may not be present in Tune trial placement
        # group bundles.
        if hasattr(ds, "random_sample"):
            return ds.random_sample(frac, seed=seed)

        return ds

    # Same workaround as xgboost: build the Trainer inside a function trainable.
    def _trainable(trial_config: Dict):
        
        train_dataset = _maybe_sample_train_ds(ray.data.read_parquet(train_path))
        val_dataset = ray.data.read_parquet(val_path)

        max_train_rows = int(os.getenv("TUNE_MAX_TRAIN_ROWS", "0"))
        max_val_rows = int(os.getenv("TUNE_MAX_VAL_ROWS", "0"))
        if max_train_rows > 0:
            train_dataset = train_dataset.limit(max_train_rows)
        if max_val_rows > 0:
            val_dataset = val_dataset.limit(max_val_rows)

        # Optional: materialize per-trial datasets to avoid repeated `ReadParquet`
        # executions across epochs inside the Train loop.
        # Keep default OFF because many concurrent trials can pressure the object store.
        if os.getenv("RAY_MATERIALIZE_DATASETS_TUNE", "0").lower() in ("1", "true", "yes"):
            train_dataset = train_dataset.materialize()
            val_dataset = val_dataset.materialize()

        train_loop_config = {
            "target": target,
            "pytorch_params": trial_config["pytorch_params"],
            "input_dim": 14,
            "num_classes": int(num_classes),
            "cpus_per_worker": cpus_per_worker,
        }

        # Ensure Ray Train persists checkpoints/metrics to shared storage.
        # Without this, Ray Train falls back to local /home/ray/ray_results which
        # is NOT shared across pods in Kubernetes, and checkpoint upload fails.
        try:
            trial_id = tune.get_context().get_trial_id()
        except Exception:
            trial_id = str(os.getpid())

        trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            train_loop_config=train_loop_config,
            scaling_config=scaling_config,
            datasets={"train": train_dataset, "val": val_dataset},
            run_config=RunConfig(
                storage_path=storage_path,
                name=f"{name}_train_{trial_id}",
            ),
        )
        result = trainer.fit()
        metrics = getattr(result, "metrics", None) or {}
        report_dict: Dict[str, numbers.Real] = {}
        for k, v in metrics.items():
            if not isinstance(v, numbers.Real) or isinstance(v, bool):
                continue
            if k in ("training_iteration", "epoch", "step"):
                report_dict[k] = int(v)
            else:
                report_dict[k] = float(v)
        tune.report(report_dict)

    # IMPORTANT:
    # Do NOT reserve `num_workers * cpus_per_worker` on the Tune trial actor.
    # The TorchTrainer will create a placement group with those resources.
    # Reserving them here can cause fragmentation and unschedulable placement groups.
    trainable = _trainable

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
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=5,
            scheduler=scheduler,
            max_concurrent_trials=int(os.getenv("MAX_CONCURRENT_TRIALS", "1")),
        ),
        run_config=RunConfig(storage_path=storage_path, name=name, callbacks=callbacks),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="val_loss", mode="min")
    return best.config