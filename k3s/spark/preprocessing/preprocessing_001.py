import json
from typing import Dict, Tuple
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# ======================================================
# 1. FEATURE ENGINEERING (Spark SQL only)
# ======================================================

def feature_engineering(df: DataFrame) -> DataFrame:
    df = df.withColumn('timestamp_ts', F.col('timestamp').cast('timestamp'))
    df = df.withColumn('hour', F.hour('timestamp_ts'))
    df = df.withColumn('dayofweek', F.dayofweek('timestamp_ts') - 1)

    df = df.withColumn(
        'is_weekend',
        F.when(F.col('dayofweek').isin(5, 6), 1).otherwise(0)
    )

    df = df.withColumn(
        'hour_sin',
        F.sin(2 * 3.1415926535 * F.col('hour') / 24.0)
    )
    df = df.withColumn(
        'hour_cos',
        F.cos(2 * 3.1415926535 * F.col('hour') / 24.0)
    )

    df = df.withColumn('bytes_log', F.log1p(F.col('bytes_transferred')))
    df = df.withColumn('packet_log', F.log1p(F.col('packet_count')))

    df = df.withColumn(
        'protocol_conn',
        F.concat_ws('_', F.col('protocol'), F.col('conn_state'))
    )

    return df


# ======================================================
# 2. CATEGORICAL ENCODERS (FIT / APPLY)
# ======================================================

def fit_encoders(df: DataFrame, cols: list) -> Dict[str, Dict[str, int]]:
    encoders = {}

    for c in cols:
        values = (
            df.select(c)
              .distinct()
              .rdd
              .map(lambda r: r[0])
              .collect()
        )

        encoders[c] = {v: i for i, v in enumerate(sorted(values))}

    return encoders


def apply_encoders(df: DataFrame, encoders: Dict[str, Dict[str, int]]) -> DataFrame:
    for c, mapping in encoders.items():
        mapping_expr = F.create_map(
            *[F.lit(x) for kv in mapping.items() for x in kv]
        )

        df = df.withColumn(
            f"{c}_idx",
            F.coalesce(mapping_expr[F.col(c)], F.lit(-1))
        )

    return df


# ======================================================
# 3. SCALER (FIT / APPLY)
# ======================================================

def fit_scaler(df: DataFrame, cols: list) -> Dict[str, Dict[str, float]]:
    stats = {}

    for c in cols:
        row = df.select(
            F.mean(c).alias("mean"),
            F.stddev(c).alias("std")
        ).collect()[0]

        stats[c] = {
            "mean": float(row["mean"]),
            "std": float(row["std"] or 1.0)
        }

    return stats


def apply_scaler(df: DataFrame, stats: Dict[str, Dict[str, float]]) -> DataFrame:
    for c, s in stats.items():
        df = df.withColumn(
            f"{c}_norm",
            (F.col(c) - F.lit(s["mean"])) / F.lit(s["std"])
        )

    return df


# ======================================================
# 4. MAIN ENTRY (MISMA FIRMA QUE ANTES)
# ======================================================

def preprocess_spark(
    spark_df: DataFrame,
    model: Dict = None,
    train: bool = True
) -> Tuple[DataFrame, Dict]:

    df = feature_engineering(spark_df)

    cat_cols = ['protocol', 'conn_state', 'protocol_conn']

    num_cols = [
        'src_port',
        'dst_port',
        'packet_count',
        'bytes_transferred',
        'bytes_log',
        'packet_log',
        'hour',
        'dayofweek',
        'is_weekend',
        'hour_sin',
        'hour_cos'
    ]

    if train:
        encoders = fit_encoders(df, cat_cols)
        scaler = fit_scaler(df, num_cols)

        model = {
            "encoders": encoders,
            "scaler": scaler
        }
    else:
        encoders = model["encoders"]
        scaler = model["scaler"]

    df = apply_encoders(df, encoders)
    df = apply_scaler(df, scaler)

    return df, model


