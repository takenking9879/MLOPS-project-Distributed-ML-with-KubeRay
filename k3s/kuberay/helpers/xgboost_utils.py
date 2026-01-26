from typing import Tuple
import os

import ray
import xgboost


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
