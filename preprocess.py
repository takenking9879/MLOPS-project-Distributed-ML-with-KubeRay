

# -------------------------
# PREPROCESSING - PANDAS (large, realistic)
# -------------------------
def preprocess_pandas(df, fit_encoders=True, fit_scaler=True, global_stats=None):
    """
    Preprocesamiento complejo estilo PySpark-ready:
    - Categóricas -> LabelEncoder
    - Features log-transform / ratios
    - Rolling/window stats por session
    - Estadísticas globales (mean/std) para normalización
    """

    df = df.copy()

    # 1) timestamp -> segundos desde epoch
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp_epoch'] = df['timestamp'].astype('int64') // 10**9

    # 2) crear feature log + small offset para evitar log(0)
    df['bytes_log'] = np.log1p(df['bytes_transferred'])
    df['packet_log'] = np.log1p(df['packet_count'])

    # 3) ratios
    df['bytes_per_packet'] = df['bytes_transferred'] / (df['packet_count'] + 1)
    df['bytes_per_packet_log'] = np.log1p(df['bytes_per_packet'])

    # 4) session_id: combinación de src_port + dst_port + protocol (puede traducirse a PySpark)
    df['session_id'] = df['src_port'].astype(str) + '-' + df['dst_port'].astype(str) + '-' + df['protocol']

    # 5) rolling stats por sesión (usando transform para mantener índice)
    df['prev_bytes_mean_3'] = df.groupby('session_id')['bytes_log'] \
                                .transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
    df['prev_packet_mean_3'] = df.groupby('session_id')['packet_log'] \
                                 .transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
    df['prev_event_count_3'] = df.groupby('session_id')['packet_count'] \
                                  .transform(lambda s: s.shift().rolling(3, min_periods=1).count())

    # 6) features agregadas por session (global session stats)
    df['session_bytes_max'] = df.groupby('session_id')['bytes_log'].transform('max')
    df['session_bytes_min'] = df.groupby('session_id')['bytes_log'].transform('min')
    df['session_packet_mean'] = df.groupby('session_id')['packet_log'].transform('mean')

    # 7) rolling stats pueden producir NaN en los primeros eventos de cada sesión
    roll_cols = ['prev_bytes_mean_3', 'prev_packet_mean_3', 'prev_event_count_3']
    df[roll_cols] = df[roll_cols].fillna(0)

    # 8) codificación de categóricas (fit nuevo o reusar encoders existentes)
    if isinstance(fit_encoders, dict):
        encoders = fit_encoders
        fit_new_encoders = False
    else:
        encoders = {}
        fit_new_encoders = bool(fit_encoders)

    for col in ['protocol', 'conn_state']:
        if fit_new_encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            mapping = {cls: int(i) for i, cls in enumerate(le.classes_)}
            df[col] = df[col].map(mapping).fillna(-1).astype(int)

    # 9) columnas finales para modelado (antes de normalizar)
    base_feature_cols = ['src_port', 'dst_port', 'packet_count', 'bytes_transferred', 'bytes_log', 'packet_log',
                         'bytes_per_packet', 'bytes_per_packet_log',
                         'prev_bytes_mean_3', 'prev_packet_mean_3', 'prev_event_count_3',
                         'session_bytes_max', 'session_bytes_min', 'session_packet_mean',
                         'protocol', 'conn_state', 'timestamp_epoch']

    # 10) escalado: aceptar `fit_scaler=True` (fit) o un StandardScaler ya entrenado
    X = df[base_feature_cols].fillna(0).values
    if isinstance(fit_scaler, StandardScaler):
        scaler = fit_scaler
    elif bool(fit_scaler):
        scaler = StandardScaler()
        scaler.fit(X)
    else:
        scaler = None

    if scaler is not None:
        Xs = scaler.transform(X)
        global_stats = {
            f: {'mean': float(scaler.mean_[i]), 'std': float(scaler.scale_[i])}
            for i, f in enumerate(base_feature_cols)
        }
        for i, f in enumerate(base_feature_cols):
            df[f + '_norm'] = Xs[:, i]
    else:
        # fallback: usar stats provistas o calcularlas desde el df
        if global_stats is None:
            global_stats = {f: {'mean': df[f].mean(), 'std': df[f].std()} for f in base_feature_cols}
        for f in base_feature_cols:
            df[f + '_norm'] = (df[f].fillna(0) - global_stats[f]['mean']) / (global_stats[f]['std'] + 1e-6)

    feature_cols = [f + '_norm' for f in base_feature_cols]
    return df, encoders, scaler, global_stats, feature_cols


# -------------------------
# Pyspark equivalent preprocessing (if you want distributed)
# -------------------------
def preprocess_spark(spark, spark_df):
    """
    Example PySpark pipeline equivalent. This returns a Spark DataFrame with similar derived features.
    This function is illustrative; to run it you need pyspark configured.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler

    df = spark_df
    # time features
    df = df.withColumn('timestamp_ts', F.col('timestamp').cast('timestamp'))
    df = df.withColumn('hour', F.hour('timestamp_ts'))
    df = df.withColumn('dayofweek', F.dayofweek('timestamp_ts') - 1)  # pyspark dayofweek 1..7
    df = df.withColumn('is_weekend', F.when(F.col('dayofweek').isin(5, 6), 1).otherwise(0))
    # sin/cos for hour
    df = df.withColumn('hour_sin', F.sin(2 * 3.1415926535 * F.col('hour') / F.lit(24.0)))
    df = df.withColumn('hour_cos', F.cos(2 * 3.1415926535 * F.col('hour') / F.lit(24.0)))

    # flags
    well_known = [21, 22, 23, 25, 53, 80, 110, 143, 443]
    df = df.withColumn('dst_well_known', F.when(F.col('dst_port').isin(well_known), 1).otherwise(0))
    df = df.withColumn('src_port_bucket', (F.col('src_port') / 10000).cast('int'))

    # logs & ratios
    df = df.withColumn('bytes_log', F.log1p(F.col('bytes_transferred')))
    df = df.withColumn('packet_log', F.log1p(F.col('packet_count')))
    df = df.withColumn('pkt_per_byte', F.col('packet_count') / (F.col('bytes_transferred') + F.lit(1.0)))

    # crosses
    df = df.withColumn('protocol_conn', F.concat_ws('_', F.col('protocol'), F.col('conn_state')))

    # session id
    df = df.withColumn('session_id', F.concat_ws('_', F.col('src_port').cast('string'), F.col('dst_port').cast('string'), F.col('protocol')))

    # window functions per session (previous 3 events)
    w = Window.partitionBy('session_id').orderBy('timestamp_ts').rowsBetween(-3, -1)
    df = df.withColumn('prev_bytes_mean_3', F.avg('bytes_log').over(w))
    df = df.withColumn('prev_packet_mean_3', F.avg('packet_log').over(w))
    df = df.withColumn('prev_event_count_3', F.count('packet_count').over(w))

    # encode categorical columns using StringIndexer (example)
    protocol_indexer = StringIndexer(inputCol='protocol', outputCol='protocol_idx').fit(df)
    df = protocol_indexer.transform(df)
    conn_indexer = StringIndexer(inputCol='conn_state', outputCol='conn_state_idx').fit(df)
    df = conn_indexer.transform(df)
    pc_indexer = StringIndexer(inputCol='protocol_conn', outputCol='protocol_conn_idx').fit(df)
    df = pc_indexer.transform(df)

    # choose features and assemble
    feat_cols = ['src_port', 'dst_port', 'packet_count', 'bytes_transferred', 'bytes_log', 'packet_log', 'pkt_per_byte',
                 'hour', 'dayofweek', 'is_weekend', 'hour_sin', 'hour_cos',
                 'dst_well_known', 'src_port_bucket', 'protocol_idx', 'conn_state_idx', 'protocol_conn_idx',
                 'prev_bytes_mean_3', 'prev_packet_mean_3', 'prev_event_count_3']
    assembler = VectorAssembler(inputCols=feat_cols, outputCol='features_vec')
    df = assembler.transform(df)

    # standardize
    scaler = StandardScaler(inputCol='features_vec', outputCol='features_scaled', withMean=True, withStd=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)

    return df  # features in 'features_scaled' vector


# -------------------------
# Helper: try to save Parquet (simulate distributed storage)
# -------------------------
def save_parquet(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved parquet to {path}")


# -------------------------
# Full demo pipeline (generate -> preprocess -> train) using pandas preprocess
# -------------------------
def run_demo():
    gen = SyntheticTrafficGenerator(start_ts="2026-01-12 18:00:00", epsilon_seconds=60, rng=RNG)
    TRAIN_N, VAL_N = 2000, 3000

    print("Generating datasets...")
    df_train = gen.generate_dataset(TRAIN_N, trend='normal')
    df_val_normal = gen.generate_dataset(VAL_N, trend='normal')
    df_val_data = gen.generate_dataset(VAL_N, trend='data_drift')
    df_val_concept = gen.generate_dataset(VAL_N, trend='concept_drift')

    # build concept_train (val_normal features relabeled by concept)
    df_concept_train = df_val_normal.copy()
    # Keep prior behavior (concept oracle over normal-distribution features), but vectorized for scale.
    proto_idx = df_concept_train['protocol'].map(gen._proto_to_idx).astype(np.int32).values
    conn_idx = df_concept_train['conn_state'].map(gen._conn_to_idx).astype(np.int32).values
    df_concept_train['attack'] = gen._fX_concept_vec(
        src_port=df_concept_train['src_port'].astype(np.int32).values,
        dst_port=df_concept_train['dst_port'].astype(np.int32).values,
        proto_idx=proto_idx,
        packet_count=df_concept_train['packet_count'].astype(np.int32).values,
        conn_idx=conn_idx,
        bytes_transferred=df_concept_train['bytes_transferred'].astype(np.float64).values,
    ).astype(int)

    # preprocess train (fits encoders/scaler)
    print("Preprocessing train (Pandas)...")
    df_train_proc, encoders, scaler, global_stats, feature_cols = preprocess_pandas(df_train)

    # preprocess val using same encoders/scaler / global_stats
    print("Preprocessing val_normal (Pandas)...")
    df_valn_proc, _, _, _, _ = preprocess_pandas(df_val_normal, fit_encoders=encoders, fit_scaler=scaler, global_stats=global_stats)
    df_vald_proc, _, _, _, _ = preprocess_pandas(df_val_data, fit_encoders=encoders, fit_scaler=scaler, global_stats=global_stats)
    df_valc_proc, _, _, _, _ = preprocess_pandas(df_val_concept, fit_encoders=encoders, fit_scaler=scaler, global_stats=global_stats)
    df_concept_train_proc, _, _, _, _ = preprocess_pandas(df_concept_train, fit_encoders=encoders, fit_scaler=scaler, global_stats=global_stats)

    # optionally save to parquet for distributed load
    save_parquet(df_train_proc, "out/train_proc.parquet")
    save_parquet(df_valn_proc, "out/val_normal_proc.parquet")
    save_parquet(df_vald_proc, "out/val_data_proc.parquet")
    save_parquet(df_valc_proc, "out/val_concept_proc.parquet")

    # prepare arrays for xgboost (use feature_cols returned by preprocess)
    X_train = df_train_proc[feature_cols].values
    y_train = df_train_proc['attack'].values
    Xn = df_valn_proc[feature_cols].values
    yn = df_valn_proc['attack'].values
    Xd = df_vald_proc[feature_cols].values
    yd = df_vald_proc['attack'].values
    Xc = df_valc_proc[feature_cols].values
    yc = df_valc_proc['attack'].values

    # train helper (GPU fallback)
    def train_xgb(X_tr, y_tr, eval_set):
        params = {
            'n_estimators': 100,
            'max_depth': 7,
            'learning_rate': 0.1,
            'objective': 'multi:softprob',
            'num_class': NUM_CLASSES,
            'verbosity': 1,
            'eval_metric': ['mlogloss', 'merror'],
        }
        try:
            model = xgb.XGBClassifier(**params, tree_method='hist', device='gpu')
            model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
            print("Trained with GPU.")
        except Exception as e:
            print("GPU failed:", e, "Falling back to CPU.")
            model = xgb.XGBClassifier(**params, tree_method='hist', device='cpu')
            model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
        return model

    print("\n=== EXP: train on normal, eval on normal/data/concept ===")
    model_norm = train_xgb(X_train, y_train, eval_set=[(Xn, yn)])

    def run_report(m, Xv, yv, name):
        ypred = m.predict(Xv)
        print(f"\n--- {name} --- Accuracy: {accuracy_score(yv, ypred):.4f}")
        print(classification_report(yv, ypred, target_names=[ATTACK_LABELS[i] for i in range(NUM_CLASSES)], digits=4))

    run_report(model_norm, Xn, yn, "val_normal")
    run_report(model_norm, Xd, yd, "val_data_drift")
    run_report(model_norm, Xc, yc, "val_concept_drift")

    # retrain on concept (as before)
    print("\n=== EXP: retrain on concept (using df_concept_train_proc) ===")
    X_concept_train = df_concept_train_proc[feature_cols].values
    y_concept_train = df_concept_train_proc['attack'].values
    X_concept_val = df_valc_proc[feature_cols].values
    y_concept_val = df_valc_proc['attack'].values
    model_concept = train_xgb(X_concept_train, y_concept_train, eval_set=[(X_concept_val, y_concept_val)])
    run_report(model_concept, X_concept_val, y_concept_val, "concept_val (after retrain)")

    print("\nSample processed train head:")
    print(df_train_proc.head(6))
    print("\nFeature columns used:", feature_cols)

