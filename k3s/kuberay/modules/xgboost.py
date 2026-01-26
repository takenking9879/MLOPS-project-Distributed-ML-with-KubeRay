""""
XGBoost training module using Ray Train. It only supports RAM-based training."""

import os
import time
import logging
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import xgboost
import ray.train
from ray.train import Checkpoint
from ray.train.xgboost import XGBoostTrainer

from schemas.xgboost_params import XGBOOST_PARAMS
from helpers.metrics_utils import xgb_multiclass_metrics_on_val
from helpers.xgboost_utils import get_train_val_dmatrix, run_xgboost_train

logger = logging.getLogger(__name__)


class RayTrainPeriodicReportCheckpointCallback(xgboost.callback.TrainingCallback):
    """Periodic metric reporting + checkpointing for Ray Train.

    Ray's built-in `RayTrainReportCallback` reports metrics every iteration.
    For small datasets this overhead can dominate runtime in Kubernetes.
    This callback reports every `report_every` iterations and checkpoints
    every `checkpoint_every` iterations (rank 0 only), plus a final checkpoint
    at the end.
    """

    def __init__(
        self,
        *,
        report_every: int = 5,
        checkpoint_every: int = 50,
        filename: str = "model.ubj",
    ):
        self.report_every = max(int(report_every), 1)
        self.checkpoint_every = max(int(checkpoint_every), 1)
        self.filename = filename
        self._last_checkpoint_iter: int | None = None

    def _latest_metric(self, evals_log, dataset: str, metric: str):
        try:
            v = evals_log[dataset][metric]
            return v[-1] if isinstance(v, list) else v
        except Exception:
            return None

    def _report(self, report_dict: Dict, model: xgboost.Booster, *, checkpoint: bool) -> None:
        world_rank = ray.train.get_context().get_world_rank()
        if checkpoint and world_rank in (0, None):
            import tempfile
            import os

            with tempfile.TemporaryDirectory() as tmpdir:
                model.save_model(os.path.join(tmpdir, self.filename))
                ray_checkpoint = Checkpoint.from_directory(tmpdir)
                ray.train.report(report_dict, checkpoint=ray_checkpoint)
        else:
            ray.train.report(report_dict)

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        # XGBoost counts epochs from 0.
        it = epoch + 1
        if it % self.report_every != 0:
            return False

        report_dict: Dict[str, float] = {}
        mlogloss = self._latest_metric(evals_log, "validation", "mlogloss")
        merror = self._latest_metric(evals_log, "validation", "merror")
        if mlogloss is not None:
            report_dict["validation-mlogloss"] = float(mlogloss)
        if merror is not None:
            report_dict["validation-merror"] = float(merror)

        do_ckpt = (it % self.checkpoint_every == 0)
        if do_ckpt:
            self._last_checkpoint_iter = epoch
        self._report(report_dict, model, checkpoint=do_ckpt)
        return False

    def after_training(self, model):
        # Avoid duplicate checkpoint if we checkpointed on the last iteration.
        try:
            last_iter = model.num_boosted_rounds() - 1
        except Exception:
            last_iter = None

        if last_iter is not None and self._last_checkpoint_iter == last_iter:
            return model

        # Final report+checkpoint.
        self._report({}, model, checkpoint=True)
        return model

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

    result = trainer.fit()
    
    # Métricas finales (mezcla de métricas reportadas por Ray + multiclass en val)
    final_metrics: Dict[str, float] = {}

    if getattr(result, "metrics", None):
        for k, v in result.metrics.items():
            if isinstance(v, (int, float)):
                final_metrics[k] = float(v)

    mc_start = time.perf_counter()
    final_metrics.update(
        xgb_multiclass_metrics_on_val(
            val_ds=val_dataset,
            target=target,
            num_classes=int(num_classes),
            booster_checkpoint=result.checkpoint,
        )
    )
    final_metrics["multiclass_metrics_time_sec"] = time.perf_counter() - mc_start

    return result, final_metrics