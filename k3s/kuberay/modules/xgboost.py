""""
XGBoost training module using Ray Train. It only supports RAM-based training."""

import os
import time
import logging
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import xgboost
import ray.train
from ray.train.xgboost import RayTrainReportCallback, XGBoostTrainer

from schemas.xgboost_params import XGBOOST_PARAMS
from helpers.metrics_utils import xgb_multiclass_metrics_on_val
from helpers.xgboost_utils import get_train_val_dmatrix, run_xgboost_train

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
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(cpus_per_worker)
    
    # Log actual CPU configuration for debugging
    if ray.train.get_context().get_world_rank() == 0:
        print(f"[xgboost] Worker using nthread={cpus_per_worker} | "
              f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'not set')}")
    
    num_boost_round = params.get("num_boost_round", 100)
    dtrain, dval = get_train_val_dmatrix(target)
    run_xgboost_train(
        params=params,
        dtrain=dtrain,
        dval=dval,
        num_boost_round=num_boost_round,
        callbacks=[
            RayTrainReportCallback(
                metrics=["validation-mlogloss", "validation-merror"],
                frequency=1,
            )
        ],
    )

# Main training function
def train(train_dataset, val_dataset, target, storage_path, name, num_classes: int = 6, xgboost_params=None):
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

    start_time = time.perf_counter()
    result = trainer.fit()
    train_time_sec = time.perf_counter() - start_time
    print(f"[xgboost] distributed train_time_sec={train_time_sec:.2f}")

    # Métricas finales (mezcla de métricas reportadas por Ray + multiclass en val)
    final_metrics: Dict[str, float] = {}
    try:
        if getattr(result, "metrics", None):
            for k, v in result.metrics.items():
                if isinstance(v, (int, float)):
                    final_metrics[k] = float(v)
    except Exception as e:
        logger.warning(
            "[xgboost] No se pudieron extraer métricas numéricas de result.metrics: %s",
            str(e),
            exc_info=True,
        )

    final_metrics["train_time_sec"] = train_time_sec

    try:
        if getattr(result, "checkpoint", None):
            final_metrics.update(
                xgb_multiclass_metrics_on_val(
                    val_ds=val_dataset,
                    target=target,
                    num_classes=int(num_classes),
                    booster_checkpoint=result.checkpoint,
                )
            )
    except Exception as e:
        logger.warning(
            "[xgboost] Falló el cálculo de métricas multiclass en val: %s",
            str(e),
            exc_info=True,
        )

    return result, final_metrics