""""
XGBoost training module using Ray Train. It only supports RAM-based training."""

import os
import time
import logging
from typing import Any, Dict

import ray.train
from ray.train.xgboost import XGBoostTrainer

from schemas.xgboost_params import XGBOOST_PARAMS
from helpers.metrics_utils import xgb_multiclass_metrics_on_ds
from helpers.xgboost_utils import get_train_val_dmatrix, run_xgboost_train, RayTrainPeriodicReportCheckpointCallback

logger = logging.getLogger(__name__)

# Training function for each worker
def train_func(config: Dict):
    """Runs on each Ray Train worker."""
    # Copy params to avoid mutating shared dicts across workers/trials.
    params = dict(config.get("xgboost_params", XGBOOST_PARAMS))
    target = config["target"]
    params["num_class"] = int(config.get("num_classes", 2))

    # Align XGBoost threading with the CPU allocated per Ray Train worker.
    cpus_per_worker = int(config.get("cpus_per_worker", os.getenv("CPUS_PER_WORKER", "1")))
    cpus_per_worker = max(cpus_per_worker, 1)
    params["nthread"] = cpus_per_worker

    # `num_boost_round` is a top-level argument to xgboost.train, not a param.
    # Keep it out of `params` to avoid XGBoost warnings (and keep logs clean).
    num_boost_round = int(params.pop("num_boost_round", 100))
    dtrain, dval = get_train_val_dmatrix(target)

    #Aqui el tiempo de entrenamiento
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
            )
        ],
    )
    train_time_sec = time.perf_counter() - start_time
    logger.info(f"[xgboost] Worker train_time_sec={train_time_sec:.2f}")
    #Aqui termina el tiempo de entrenamiento

# Main training function
def train(train_dataset, val_dataset, test_dataset, target, storage_path, name, num_classes: int = 6, xgboost_params=None):
    scaling_config = ray.train.ScalingConfig(
        num_workers=int(os.getenv("NUM_WORKERS", 2)),
        resources_per_worker={"CPU": int(os.getenv("CPUS_PER_WORKER", 2))})
    
    params = xgboost_params if xgboost_params is not None else XGBOOST_PARAMS
    config = {
        "target": target,
        "num_classes": int(num_classes),
        "xgboost_params": params,
        "cpus_per_worker": int(os.getenv("CPUS_PER_WORKER", 2)),
    }

    trainer = XGBoostTrainer(
        train_loop_per_worker=train_func, #Función de entrenamiento
        train_loop_config=config, #Configuración del entrenamiento
        scaling_config=scaling_config, #Configuración de recursos
        datasets={"train": train_dataset, "val": val_dataset}, #Pasar datasets leidos
        run_config=ray.train.RunConfig(storage_path=storage_path, name=name), #Donde guardar los resultados
    )

    result = trainer.fit()
    
    # Métricas finales (mezcla de métricas reportadas por Ray + multiclass en val)
    final_metrics: Dict[str, Any] = {}

    if getattr(result, "metrics", None):
        for k, v in result.metrics.items():
            if isinstance(v, (int, float)):
                final_metrics[k] = float(v)

    # Final Evaluation (Driver-Side) for full datasets
    mc_start = time.perf_counter()

    # 1. Validation metrics (full dataset)
    final_metrics.update(
        xgb_multiclass_metrics_on_ds(
            ds=val_dataset,
            split="val",
            target=target,
            num_classes=int(num_classes),
            booster_checkpoint=result.checkpoint,
        )
    )

    # 2. Test metrics (if exists)
    if test_dataset is not None:
        final_metrics.update(
            xgb_multiclass_metrics_on_ds(
                ds=test_dataset,
                split="test",
                target=target,
                num_classes=int(num_classes),
                booster_checkpoint=result.checkpoint,
            )
        )
    final_metrics["multiclass_metrics_time_sec"] = time.perf_counter() - mc_start

    return result, final_metrics