"""
XGBoost training module using Ray Train + Ray Tune.
Supports cheap HPT with ASHA + ResourceChangingScheduler.
"""

import os
import numbers
from typing import Dict

import xgboost
import ray
import ray.train
from ray import tune
from ray.train import ScalingConfig
from ray.tune import RunConfig
from ray.train.xgboost import RayTrainReportCallback, XGBoostTrainer
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import ASHAScheduler, ResourceChangingScheduler

from schemas.xgboost_params import SEARCH_SPACE_XGBOOST_PARAMS, XGBOOST_TUNE_SETTINGS
from helpers.xgboost_utils import get_train_val_dmatrix, run_xgboost_train


CHECKPOINT_FILENAME = "xgb_checkpoint.json"

# --------------------------------------------------
# Train loop (runs on each Ray Train worker)
# --------------------------------------------------
def train_func(config: Dict):
    # Copy to avoid mutating the Tune search-space dict across calls.
    params = dict(config["xgboost_params"])
    target = config["target"]
    num_classes = int(config.get("num_classes", 2))

    num_boost_round = int(config.get("num_boost_round", 50))

    dtrain, dval = get_train_val_dmatrix(target)

    # IMPORTANT:
    # In Ray Train integration, the train loop runs on Train workers.
    # We keep `nthread` consistent with the CPU allocated per worker bundle.
    cpus_per_worker = int(config.get("cpus_per_worker", os.getenv("CPUS_PER_WORKER", "1")))
    cpus_per_worker = max(cpus_per_worker, 1)

    # IMPORTANT (Ray docs): if num_cpus isn't set on the worker, Ray defaults
    # to OMP_NUM_THREADS=1. We set it explicitly to actually use the allocated CPUs.
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(cpus_per_worker)

    params["nthread"] = cpus_per_worker
    params["num_class"] = num_classes

    run_xgboost_train(
        params=params,
        dtrain=dtrain,
        dval=dval,
        num_boost_round=num_boost_round,
        callbacks=[
            RayTrainReportCallback(
                metrics=["validation-mlogloss", "validation-merror"],
                frequency=1,
            ),
        ],
    )

# --------------------------------------------------
# Tuning entrypoint
# --------------------------------------------------
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

    # Two-stage idea:
    # - Use a cheaper resource config for tuning (fewer workers / CPUs)
    # - Use a larger resource config for the final distributed training
    num_workers = int(os.getenv("NUM_WORKERS_TUNE", os.getenv("NUM_WORKERS", 2)))
    cpus_per_worker = int(os.getenv("CPUS_PER_WORKER_TUNE", os.getenv("CPUS_PER_WORKER", 1)))

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        resources_per_worker={"CPU": cpus_per_worker},
    )

    # --- Search space (cheap tuning) ---
    # NOTE: We keep the keys aligned with main.py expectations.
    # main.py does: best_config.get("xgboost_params")
    param_space = {
        "xgboost_params": SEARCH_SPACE_XGBOOST_PARAMS,
    }

    # --- Early stopping ---
    asha = ASHAScheduler(
        metric="validation-mlogloss",
        mode="min",
        max_t=XGBOOST_TUNE_SETTINGS["num_boost_round"],
        grace_period=XGBOOST_TUNE_SETTINGS["grace_period"],
        reduction_factor=XGBOOST_TUNE_SETTINGS["reduction_factor"],
    )

    # --- Dynamic CPU allocation (optional) ---
    # ResourceChangingScheduler can request larger resource bundles over time.
    # In small clusters this may produce "infeasible resource requests" warnings.
    enable_rcs = os.getenv("ENABLE_RESOURCE_CHANGING_SCHEDULER", "false").lower() in ("1", "true", "yes")
    scheduler = ResourceChangingScheduler(base_scheduler=asha) if enable_rcs else asha

    # IMPORTANT:
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

        # NOTE:
        # Avoid groupby/map_groups here because it triggers shuffles. In Ray Tune, trial
        # placement groups can capture child tasks, and Ray Data shuffle tasks request
        # the implicit `memory` resource, which isn't in the placement group bundles by
        # default (causing scheduling errors).
        if hasattr(ds, "random_sample"):
            return ds.random_sample(frac, seed=seed)

        # Fallback (best-effort): no sampling.
        return ds

    def _trainable(trial_config: Dict):
        train_dataset = _maybe_sample_train_ds(ray.data.read_parquet(train_path))
        val_dataset = ray.data.read_parquet(val_path)

        # Extra safety for laptop-sized RAM: cap rows for tuning to avoid large
        # per-worker pandas materializations.
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
            "num_classes": int(num_classes),
            "cpus_per_worker": cpus_per_worker,
            "num_boost_round": int(XGBOOST_TUNE_SETTINGS["num_boost_round"]),
            "xgboost_params": trial_config["xgboost_params"],
        }

        # Ray Train may emit checkpoints via RayTrainReportCallback. In a multi-pod
        # Ray cluster, the default local filesystem path is not shared across pods.
        # Ensure Train uses a shared storage URI (S3/MinIO) so checkpoint/reporting
        # doesn't fail with "cluster storage" errors.
        try:
            trial_id = tune.get_context().get_trial_id()
        except Exception:
            trial_id = str(os.getpid())

        trainer = XGBoostTrainer(
            train_loop_per_worker=train_func,
            train_loop_config=train_loop_config,
            scaling_config=scaling_config,
            datasets={"train": train_dataset, "val": val_dataset},
            run_config=ray.train.RunConfig(
                storage_path=storage_path,
                name=f"{name}_train_{trial_id}",
            ),
        )
        result = trainer.fit()
        metrics = getattr(result, "metrics", None) or {}
        tune.report(
            {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, numbers.Real) and not isinstance(v, bool)
            }
        )

    # Make Tune account for the full CPU budget per trial.
    # This controls the "Logical resource usage" line in Tune output.
    # Default to num_workers * cpus_per_worker unless overridden.
    cpus_per_trial = int(os.getenv("CPUS_PER_TRIAL", str(num_workers * cpus_per_worker)))
    trainable = tune.with_resources(_trainable, resources={"cpu": cpus_per_trial})

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
            num_samples=3,
            scheduler=scheduler,
            max_concurrent_trials=int(os.getenv("MAX_CONCURRENT_TRIALS", "1")),
        ),
        run_config=RunConfig(
            storage_path=storage_path,
            name=name,
            callbacks=callbacks,
        ),
    )

    results = tuner.fit()
    best = results.get_best_result(
        metric="validation-mlogloss",
        mode="min",
    )
    # Return the best trial config (contains `xgboost_params` and other keys)
    return best.config
