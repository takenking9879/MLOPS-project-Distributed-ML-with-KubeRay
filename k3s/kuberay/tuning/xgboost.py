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

    run_xgboost_train(
        params=params,
        dtrain=dtrain,
        dval=dval,
        num_boost_round=num_boost_round,
        callbacks=[
            RayTrainReportCallback(
                metrics=["validation-mlogloss", "validation-merror"],
                frequency=1,
                checkpoint_frequency=1,
                checkpoint_filename=CHECKPOINT_FILENAME,
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

        # Stratified sample to avoid dropping minority classes during tuning.
        def _sample_group(df):
            n = int(len(df) * frac)
            n_final = max(min(len(df), 5), n)
            return df.sample(n=n_final, random_state=seed)

        # Materialize to avoid huge lineage and keep per-trial execution deterministic.
        return ds.groupby(target).map_groups(_sample_group).materialize()

    def _trainable(trial_config: Dict):
        train_dataset = _maybe_sample_train_ds(ray.data.read_parquet(train_path))
        val_dataset = ray.data.read_parquet(val_path)

        train_loop_config = {
            "target": target,
            "num_classes": int(num_classes),
            "cpus_per_worker": cpus_per_worker,
            "num_boost_round": int(XGBOOST_TUNE_SETTINGS["num_boost_round"]),
            "xgboost_params": trial_config["xgboost_params"],
        }

        trainer = XGBoostTrainer(
            train_loop_per_worker=train_func,
            train_loop_config=train_loop_config,
            scaling_config=scaling_config,
            datasets={"train": train_dataset, "val": val_dataset},
        )
        result = trainer.fit()
        metrics = getattr(result, "metrics", None) or {}
        tune.report(
            training_iteration=1,
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        )

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
