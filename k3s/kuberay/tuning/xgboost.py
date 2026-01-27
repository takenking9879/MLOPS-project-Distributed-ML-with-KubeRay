"""
XGBoost training module using Ray Train + Ray Tune.
Supports cheap HPT with ASHA + ResourceChangingScheduler.
"""

import os
import numbers
import time
from typing import Dict
import logging
import ray
import ray.train
from ray import tune
from ray.train import ScalingConfig
from ray.tune import RunConfig
from ray.train.xgboost import XGBoostTrainer
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import ASHAScheduler, ResourceChangingScheduler

from schemas.xgboost_params import SEARCH_SPACE_XGBOOST_PARAMS, XGBOOST_TUNE_SETTINGS
from helpers.xgboost_utils import get_train_val_dmatrix, run_xgboost_train, RayTrainPeriodicReportCheckpointCallback

logger = logging.getLogger(__name__)
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

    params["nthread"] = cpus_per_worker
    params["num_class"] = num_classes

    start_time = time.perf_counter()
    run_xgboost_train(
        params=params,
        dtrain=dtrain,
        dval=dval,
        num_boost_round=num_boost_round,
        callbacks=[
            RayTrainPeriodicReportCheckpointCallback(
                metrics=["validation-mlogloss", "validation-merror"],
                report_every=5,
                checkpoint_every=50,
                filename="model.ubj",
            ),
        ],
    )
    train_time_sec = time.perf_counter() - start_time
    logger.info(f"[xgboost-tune] Worker train_time_sec={train_time_sec:.2f}")

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
    num_workers = int(os.getenv("NUM_WORKERS_TUNE", os.getenv("NUM_WORKERS", 2)))
    cpus_per_worker = int(os.getenv("CPUS_PER_WORKER_TUNE", os.getenv("CPUS_PER_WORKER", 1)))

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        resources_per_worker={"CPU": cpus_per_worker},
    )

    # --- Search space (cheap tuning) ---
    param_space = {"xgboost_params": SEARCH_SPACE_XGBOOST_PARAMS}

    # --- Early stopping ---
    asha = ASHAScheduler(
        metric="validation-mlogloss",
        mode="min",
        max_t=XGBOOST_TUNE_SETTINGS["num_boost_round"],
        grace_period=XGBOOST_TUNE_SETTINGS["grace_period"],
        reduction_factor=XGBOOST_TUNE_SETTINGS["reduction_factor"],
    )

    enable_rcs = os.getenv("ENABLE_RESOURCE_CHANGING_SCHEDULER", "false").lower() in ("1", "true", "yes")
    scheduler = ResourceChangingScheduler(base_scheduler=asha) if enable_rcs else asha

    def _maybe_sample_train_ds(ds: ray.data.Dataset) -> ray.data.Dataset:
        if sample_fraction is None:
            return ds
        frac = float(sample_fraction)
        if frac >= 1.0 or frac <= 0.0:
            return ds
        if hasattr(ds, "random_sample"):
            # do sampling; repartition afterwards to control parallelism
            ds = ds.random_sample(frac, seed=seed)
            return ds
        return ds

    def _trainable(trial_config: Dict):

        # Compute default CPUs for Ray Data during tuning:
        # env NUM_CPUS_DATA_TUNE overrides; otherwise use cluster_total - (num_workers * cpus_per_worker)
        total_cluster_cpus = int(os.getenv("NUM_CPUS_CLUSTER"))
        default_data_cpus = max(1, (total_cluster_cpus - (num_workers * cpus_per_worker * int(os.getenv("MAX_CONCURRENT_TRIALS", "1"))))/num_workers)
        cpus_for_data  = int(os.getenv("NUM_CPUS_DATA_TUNE", str(default_data_cpus)))

        cpus_for_data = max(1, cpus_for_data)
        
        print(
            f"[tune_model] total_cluster_cpus={total_cluster_cpus}\n"
            f"num_workers={num_workers}, cpus_per_worker={cpus_per_worker}, num_concurrent_trials={int(os.getenv('MAX_CONCURRENT_TRIALS', '1'))}\n"
            f"cpus_for_data_per_worker={cpus_for_data}"
        )

        # read with controlled parallelism and repartition to cpus_for_data blocks
        train_ds = ray.data.read_parquet(train_path, num_cpus=cpus_for_data)
        val_ds = ray.data.read_parquet(val_path, num_cpus=cpus_for_data)

        train_ds = _maybe_sample_train_ds(train_ds)
        # apply limits if configured
        max_train_rows = int(os.getenv("TUNE_MAX_TRAIN_ROWS", "0"))
        max_val_rows = int(os.getenv("TUNE_MAX_VAL_ROWS", "0"))
        if max_train_rows > 0:
            train_ds = train_ds.limit(max_train_rows)
        if max_val_rows > 0:
            val_ds = val_ds.limit(max_val_rows)

        # optionally materialize (careful: materialize uses cluster CPUs but we've limited blocks)
        if os.getenv("RAY_MATERIALIZE_DATASETS_TUNE", "0").lower() in ("1", "true", "yes"):
            train_ds = train_ds.materialize()
            val_ds = val_ds.materialize()

        train_loop_config = {
            "target": target,
            "num_classes": int(num_classes),
            "cpus_per_worker": cpus_per_worker,
            "num_boost_round": int(XGBOOST_TUNE_SETTINGS["num_boost_round"]),
            "xgboost_params": trial_config["xgboost_params"],
        }

        try:
            trial_id = tune.get_context().get_trial_id()
        except Exception:
            trial_id = str(os.getpid())

        trainer = XGBoostTrainer(
            train_loop_per_worker=train_func,
            train_loop_config=train_loop_config,
            scaling_config=scaling_config,
            datasets={"train": train_ds, "val": val_ds},
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
    cpus_per_trial = num_workers * cpus_per_worker
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
    best = results.get_best_result(metric="validation-mlogloss", mode="min")
    return best.config

