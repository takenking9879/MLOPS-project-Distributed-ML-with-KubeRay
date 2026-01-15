from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def preprocess_spark(spark_df, model=None, train=True):
    df = spark_df
    
    # --- 1. Feature Engineering (Igual que antes) ---
    df = df.withColumn('timestamp_ts', F.col('timestamp').cast('timestamp'))
    df = df.withColumn('hour', F.hour('timestamp_ts'))
    df = df.withColumn('dayofweek', F.dayofweek('timestamp_ts') - 1)
    df = df.withColumn('is_weekend', F.when(F.col('dayofweek').isin(5, 6), 1).otherwise(0))
    df = df.withColumn('hour_sin', F.sin(2 * 3.1415926535 * F.col('hour') / 24.0))
    df = df.withColumn('hour_cos', F.cos(2 * 3.1415926535 * F.col('hour') / 24.0))
    
    df = df.withColumn('bytes_log', F.log1p(F.col('bytes_transferred')))
    df = df.withColumn('packet_log', F.log1p(F.col('packet_count')))
    df = df.withColumn('protocol_conn', F.concat_ws('_', F.col('protocol'), F.col('conn_state')))
    
    # --- 2. Pipeline de ML (Indexers + Assembler + Scaler) ---
    feat_cols = ['src_port', 'dst_port', 'packet_count', 'bytes_transferred', 'bytes_log', 
                 'packet_log', 'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos']
    
    if train:
        # Creamos los pasos del pipeline
        indexers = [
            StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") 
            for c in ['protocol', 'conn_state', 'protocol_conn']
        ]
        
        # Agregamos las columnas indexadas a la lista de features
        indexed_feats = feat_cols + [f"{c}_idx" for c in ['protocol', 'conn_state', 'protocol_conn']]
        
        assembler = VectorAssembler(inputCols=indexed_feats, outputCol='features_vec')
        scaler = StandardScaler(inputCol='features_vec', outputCol='features_scaled', withMean=True, withStd=True)
        
        # Encapsulamos todo en un Pipeline
        pipeline = Pipeline(stages=indexers + [assembler, scaler])
        model = pipeline.fit(df)
        df_out = model.transform(df)
        return df_out, model
    else:
        # En Test/Val simplemente aplicamos el modelo cargado
        df_out = model.transform(df)
        return df_out, model