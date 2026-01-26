"""
XGBoost training module using Ray Train + Ray Tune.
Supports cheap HPT with ASHA + ResourceChangingScheduler.
"""

import os
from typing import Dict

import xgboost
import ray
import ray.train
from ray import tune
from ray.train import ScalingConfig
from ray.air import RunConfig
from ray.train.xgboost import RayTrainReportCallback, XGBoostTrainer
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import ASHAScheduler, ResourceChangingScheduler
from ray.tune.integration.xgboost import TuneReportCheckpointCallback

from schemas.xgboost_params import SEARCH_SPACE_XGBOOST_PARAMS, XGBOOST_TUNE_SETTINGS
from helpers.xgboost_utils import get_train_val_dmatrix, run_xgboost_train


CHECKPOINT_FILENAME = "xgb_checkpoint.json"

# --------------------------------------------------
# Train loop (runs on each Ray Train worker)
# --------------------------------------------------
def train_func(config: Dict):
    params = config["xgboost_params"]
    target = config["target"]
    num_classes = int(config.get("num_classes", 2))

    num_boost_round = int(config.get("num_boost_round", 50))

    dtrain, dval = get_train_val_dmatrix(target)

    # IMPORTANT:
    # In Ray Train integration, the train loop runs on Train workers.
    # We keep `nthread` consistent with the CPU allocated per worker bundle.
    params["nthread"] = int(config.get("cpus_per_worker", os.getenv("CPUS_PER_WORKER", "1")))
    params["num_class"] = num_classes

    xgb_model = None
    checkpoint = tune.get_checkpoint()
    if checkpoint:
        xgb_model = TuneReportCheckpointCallback.get_model(
            checkpoint, filename=CHECKPOINT_FILENAME
        )

    run_xgboost_train(
        params=params,
        dtrain=dtrain,
        dval=dval,
        num_boost_round=num_boost_round,
        xgb_model=xgb_model,
        callbacks=[
            RayTrainReportCallback(
                metrics=["validation-mlogloss", "validation-merror"],
                frequency=1,
                checkpoint_frequency=1,
                checkpoint_filename=CHECKPOINT_FILENAME,
            ),
            TuneReportCheckpointCallback(
                metrics={
                    "validation-mlogloss": "validation-mlogloss",
                    "validation-merror": "validation-merror",
                },
                filename=CHECKPOINT_FILENAME,
                frequency=1,
            ),
        ],
    )

# --------------------------------------------------
# Tuning entrypoint
# --------------------------------------------------
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

    # Two-stage idea:
    # - Use a cheaper resource config for tuning (fewer workers / CPUs)
    # - Use a larger resource config for the final distributed training
    num_workers = int(os.getenv("NUM_WORKERS_TUNE", os.getenv("NUM_WORKERS", 2)))
    cpus_per_worker = int(os.getenv("CPUS_PER_WORKER_TUNE", os.getenv("CPUS_PER_WORKER", 1)))

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        resources_per_worker={"CPU": cpus_per_worker},
    )

    trainer = XGBoostTrainer(
        train_loop_per_worker=train_func,
        scaling_config=scaling_config,
        datasets={"train": train_dataset, "val": val_dataset},
    )

    # --- Search space (cheap tuning) ---
    param_space = {
        "target": target,
        "num_classes": int(num_classes),
        # Used by train_func to set xgboost nthread consistently with allocated CPUs.
        "cpus_per_worker": cpus_per_worker,
        "xgboost_params": SEARCH_SPACE_XGBOOST_PARAMS
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
    # `tune.with_resources()` only supports function trainables or classes inheriting from
    # `tune.Trainable`. Ray Train's `XGBoostTrainer` is not one of those, so wrapping it
    # raises the ValueError you saw.
    #
    # Resource control for Ray Train + Tune should be done via `ScalingConfig`.
    trainable = trainer

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
            num_samples=8,
            scheduler=scheduler,
            metric="validation-mlogloss",
            mode="min",
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
