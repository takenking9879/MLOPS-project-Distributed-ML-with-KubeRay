from typing import Tuple
import os
import tempfile
import ray
import xgboost
from ray.train import Checkpoint
from typing import Dict, Optional, List

def get_train_val_dmatrix(target: str) -> Tuple[xgboost.DMatrix, xgboost.DMatrix]:
    train_shard = ray.train.get_dataset_shard("train")
    val_shard = ray.train.get_dataset_shard("val")

    # Use Ray Data's nthread for parallelization during materialization
    # This respects the CPU allocation for the worker
    cpus = int(os.getenv("OMP_NUM_THREADS", "1"))
    
    train_df = train_shard.materialize().to_pandas()
    val_df = val_shard.materialize().to_pandas()

    train_X = train_df.drop(columns=target)
    train_y = train_df[target]
    val_X = val_df.drop(columns=target)
    val_y = val_df[target]

    # XGBoost DMatrix construction can use multiple threads via nthread parameter
    return (
        xgboost.DMatrix(train_X, label=train_y, nthread=cpus),
        xgboost.DMatrix(val_X, label=val_y, nthread=cpus),
    )


def run_xgboost_train(
    *,
    params: dict,
    dtrain: xgboost.DMatrix,
    dval: xgboost.DMatrix,
    callbacks: list,
    num_boost_round: int,
    xgb_model=None,
):
    return xgboost.train(
        params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "validation")],
        verbose_eval=False,
        xgb_model=xgb_model,
        callbacks=callbacks,
    )



class RayTrainPeriodicReportCheckpointCallback(xgboost.callback.TrainingCallback):
    """Periodic metric reporting + checkpointing for Ray Train.

    Reporta métricas cada `report_every` iteraciones y checkpoint cada
    `checkpoint_every` iteraciones (solo rank 0), más un checkpoint final.
    Acepta `metric` (o `metrics`) como lista de strings en forma:
      - "validation-mlogloss"  (dataset-metric tal como XGBoost lo produce)
      - "mlogloss"             (se asume dataset "validation")
    """

    def __init__(
        self,
        *,
        report_every: int = 5,
        checkpoint_every: int = 50,
        filename: str = "model.ubj",
        # Aquí aceptamos tanto `metric` como `metrics` por compatibilidad
        metrics: Optional[List[str]] = None,
    ):
        self.report_every = max(int(report_every), 1)
        self.checkpoint_every = max(int(checkpoint_every), 1)
        self.filename = filename

        # Normalizamos alias: metrics override metric if ambos provistos
        if metrics is not None:
            self.metrics = list(metrics)
        else:
            # None significa "reporta todo" (comportamiento similar al oficial)
            self.metrics = None

        self._last_checkpoint_iter: Optional[int] = None

    def _latest_metric(self, evals_log, dataset: str, metric: str):
        try:
            v = evals_log[dataset][metric]
            return v[-1] if isinstance(v, list) else v
        except Exception:
            return None

    def _build_report_dict(self, evals_log) -> Dict[str, float]:
        report: Dict[str, float] = {}

        # Si no se especificaron métricas, reportamos todo (like Ray official)
        if self.metrics is None:
            for dataset, metrics in (evals_log or {}).items():
                for name, values in metrics.items():
                    try:
                        report[f"{dataset}-{name}"] = float(values[-1])
                    except Exception:
                        # ignoramos valores que no podamos convertir
                        pass
            return report

        # Si el usuario pasó una lista, la tratamos
        for spec in self.metrics:
            if not isinstance(spec, str):
                continue
            if "-" in spec:
                # Formato dataset-metric (ej: "validation-mlogloss")
                dataset, metric = spec.split("-", 1)
                key = spec
            else:
                # Solo metric (ej: "mlogloss") -> asumimos "validation"
                dataset = "validation"
                metric = spec
                key = f"{dataset}-{metric}"

            val = self._latest_metric(evals_log, dataset, metric)
            if val is not None:
                try:
                    report[key] = float(val)
                except Exception:
                    # si no convertible a float, lo saltamos
                    pass

        return report

    def _report(self, report_dict: Dict, model: xgboost.Booster, *, checkpoint: bool) -> None:
        world_rank = ray.train.get_context().get_world_rank()
        if checkpoint and world_rank in (0, None):
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

        report_dict: Dict[str, float] = self._build_report_dict(evals_log)
        # Añadimos training_iteration por compatibilidad con Ray dashboards
        report_dict["training_iteration"] = it

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