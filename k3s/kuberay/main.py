from utils import BaseUtils, create_logger
import boto3
import ray
import re
import os
import importlib
import pickle
import subprocess
import tempfile
from urllib.parse import urlparse
from typing import Dict, Any
from ray.train.xgboost import RayTrainReportCallback

from schemas.pytorch_params import PYTORCH_PARAMS
from schemas.xgboost_params import XGBOOST_PARAMS
from helpers.mlflow_utils import log_training_run

class KubeRayTraining(BaseUtils):
    def __init__(self, params_path: str, data_dir: str, output_dir: str):
        logger = create_logger('KubeRayTraining', 'kuberay_training.log')
        super().__init__(logger, params_path)
        self.params = self.load_params()['kuberay']['model']
        self.data_dir = data_dir
        self.output_dir = output_dir

    def _check_minio_connection(self):
        try:
            s3 = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION', 'us-east-2'),
            )

            buckets = s3.list_buckets()
            bucket_names = [b['Name'] for b in buckets.get('Buckets', [])]
            self.s3 = s3
            self.logger.info('Minio connection verified. Buckets: %s', bucket_names)
        except Exception as e:
            self.logger.error('Minio connection failed: %s', str(e), exc_info=True)
            raise
    
    def _load_data(self, path: str):
        try:
            ds = ray.data.read_parquet(path, override_num_blocks=self.params.get('num_data_blocks', None))
            
            # Senior validation before heavy processing
            self._validate_schema(ds)

            # Senior Optimization: Materialize data in the Object Store once.
            # This avoids re-scanning S3/Parquet files during each epoch of training.
            # It keeps only 1 batch in the Python worker memory while sharing blocks via shared-memory.
            if os.getenv("RAY_MATERIALIZE_DATASETS", "0") in ("1", "true", "True"):
                self.logger.info("Materializing dataset in Ray Object Store for performance.")
                ds = ds.materialize()
            self.logger.info(f"Data loaded from {path}.")
            return ds
        except Exception as e:
            self.logger.error(f"Failed to load data from {path}: {str(e)}", exc_info=True)
            raise

    def _validate_schema(self, ds: ray.data.Dataset):
        """Validates dataset schema against Spark preprocessing contract."""
        try:
            cols = set(ds.schema().names)
            expected = {
                self.params.get('target', 'attack'),
                'protocol_idx', 'conn_state_idx', 'protocol_conn_idx',
                'src_port_norm', 'dst_port_norm', 'packet_count_norm',
                'bytes_transferred_norm', 'bytes_log_norm', 'packet_log_norm',
                'hour_norm', 'dayofweek_norm', 'is_weekend_norm',
                'hour_sin_norm', 'hour_cos_norm'
            }
            
            missing = expected - cols
            if missing:
                raise ValueError(f"Data Validation failed: missing {list(missing)}")

            self.logger.info(f"Schema validation passed. Total columns: {len(cols)}")
        except Exception as e:
            self.logger.error(str(e))
            raise

    def _save_model(self, result, framework):
        """
        Extrae el mejor modelo del resultado y lo guarda en S3 como un archivo .pkl.
        Usa boto3 directo para evitar archivos temporales y el error de checksum de pyarrow.
        """
        self.logger.info(f"Exportando modelo final de {framework} a S3...")
        try:
            checkpoint = result.checkpoint
            if not checkpoint:
                self.logger.warning("No se encontró un checkpoint válido en el resultado.")
                return

            # 1. Obtener el path de S3 del checkpoint
            parsed_ckpt = urlparse(checkpoint.path)
            bucket_in = parsed_ckpt.netloc
            prefix_in = parsed_ckpt.path.lstrip('/')

            # 2. Definir qué archivo buscar según el framework
            if framework == "xgboost":
                target_file = "model.ubj"
            elif framework == "pytorch":
                target_file = "model.pt"

            key_in = os.path.join(prefix_in, target_file)

            # 3. Leer directamente de S3 a memoria
            self.logger.info(f"Leyendo {target_file} desde {checkpoint.path}")
            response = self.s3.get_object(Bucket=bucket_in, Key=key_in)
            model_bytes = response['Body'].read()

            if framework == "xgboost":
                payload = model_bytes # O pickle.dumps(model_bytes) si prefieres mantener el formato
            else:
                # Para PyTorch mantenemos tu estructura de dict
                payload = pickle.dumps({"model_pt": model_bytes})

            # 4. Subir al destino final
            parsed_out = urlparse(self.output_dir)
            bucket_out = parsed_out.netloc
            s3_key_out = os.path.join(parsed_out.path.lstrip('/'), f"model_{framework}.pkl")

            self.s3.put_object(Bucket=bucket_out, Key=s3_key_out, Body=payload if isinstance(payload, bytes) else pickle.dumps(payload))
            self.logger.info(f"Modelo guardado exitosamente en: s3://{bucket_out}/{s3_key_out}")
                
        except Exception as e:
            self.logger.error(f"Error en el export del modelo (S3 direct): {str(e)}", exc_info=True)

    def _log_final_to_mlflow(self, *, framework: str, params: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        # Senior Optimization: Encapsulated MLflow logic into a helper utility.
        # This reduces noise in the main orchestrator and separates plotting concerns.
        artifact_location = (
            params.get("mlflow_artifact_location")
            or os.getenv("MLFLOW_ARTIFACT_LOCATION")
            or "s3://k8s-mlops-platform-bucket/v1/mlflow_artifacts/"
        )

        try:
            log_training_run(
                framework=framework,
                params=params,
                metrics=metrics,
                artifact_location=artifact_location
            )
        except Exception as e:
            self.logger.error(f"Error al loggear en MLflow: {str(e)}", exc_info=True)


    def _stratified_sample(self, ds, target_col, fraction):
        """
        Realiza un muestreo estratificado distribuido usando Ray Data.
        Asegura que cada clase tenga al menos un número mínimo de muestras para no perder ninguna.
        """
        self.logger.info(f"Realizando muestreo estratificado ({fraction*100}%) sobre la columna '{target_col}'...")

        def sample_group(df):
            # Calculamos cuántas muestras representa la fracción
            n = int(len(df) * fraction)
            # Aseguramos al menos 5 muestras (o el total si el grupo es más pequeño) 
            # para no perder clases minoritarias en el tuning.
            n_final = max(min(len(df), 5), n)
            return df.sample(n=n_final, random_state=self.params.get('seed', 42))

        # map_groups permite aplicar operaciones sobre cada grupo de forma distribuida
        return ds.groupby(target_col).map_groups(sample_group)
    
    def train(self):
        try:
            status = subprocess.run(
                ["ray", "status"], capture_output=True, text=True, check=False
            )
            stdout = status.stdout
            cpu = re.search(r"([\d.]+)/([\d.]+) CPU", stdout)
            mem = re.search(r"([\dA-Za-z.]+)/([\dA-Za-z.]+) memory", stdout)
            obj = re.search(r"([\dA-Za-z.]+)/([\dA-Za-z.]+) object_store_memory", stdout)
            cpu_used, cpu_total = cpu.groups() if cpu else ("?", "?")
            mem_used, mem_total = mem.groups() if mem else ("?", "?")
            obj_used, obj_total = obj.groups() if obj else ("?", "?")

            pretty_log = f"""
            [RAY CLUSTER RESOURCES]
            ────────────────────────────────
            CPU           : {cpu_used} / {cpu_total}
            Memory        : {mem_used} / {mem_total}
            Object Store  : {obj_used} / {obj_total}
            ────────────────────────────────
            """.strip()

            self.logger.info(pretty_log)
            self._check_minio_connection()
            module = importlib.import_module(f"{'modules.' + self.params.get('framework', 'xgboost')}")

            framework = self.params.get("framework", "xgboost")

            self.logger.info(f"Starting training using framework: {framework}")
            best_params = None
            num_classes = int(self.params.get("num_classes", 2))
            mlflow_tracking_uri = self.params.get("mlflow_tracking_uri")
            mlflow_experiment_name = self.params.get("mlflow_experiment_name")
            if self.params.get('tune', False):
                self.logger.info("Starting hyperparameter tuning...")

                # NOTE:
                # We do NOT pass Ray Datasets into Tune trainables.
                # Ray Tune's `tune.with_parameters(...)` stores parameters via `ray.put`, and
                # Ray Datasets are not picklable under Ray 2.52 in that code path.
                # Instead, we pass *paths* and load/sample datasets inside each trial.
                sample_frac = self.params.get('sample_fraction_for_tuning')
                train_path = os.path.join(self.data_dir, 'train')
                val_path = os.path.join(self.data_dir, 'val')
                
                tuner = importlib.import_module('tuning.'+ framework)
                best_config = tuner.tune_model(
                    train_path=train_path,
                    val_path=val_path,
                    target=self.params['target'],
                    sample_fraction=sample_frac,
                    seed=int(self.params.get('seed', 42)),
                    storage_path=self.output_dir,
                    name=self.params.get('name', framework) + "_tune",
                    num_classes=num_classes,
                    mlflow_tracking_uri=mlflow_tracking_uri,
                    mlflow_experiment_name=mlflow_experiment_name,
                )

                best_params = best_config.get(framework + "_params")
                self.logger.info(f"Best hyperparameters found: {best_params}")

            self.logger.info("Loading datasets...")
            train_ds = self._load_data(os.path.join(self.data_dir, 'train'))
            val_ds = self._load_data(os.path.join(self.data_dir, 'val'))
            test_ds = self._load_data(os.path.join(self.data_dir, 'test'))

            train_kwargs = {
                "train_dataset": train_ds,
                "val_dataset": val_ds,
                "test_dataset": test_ds,
                "storage_path": self.output_dir,
                "name": self.params.get('name', framework),
                "target": self.params.get('target', 'attack'),
                "num_classes": num_classes,
            }

            # XGBoost specific tuned params
            if framework == "xgboost":
                if best_params is None:
                    best_params = dict(XGBOOST_PARAMS)
                best_params['num_boost_round'] = XGBOOST_PARAMS['num_boost_round']
                train_kwargs["xgboost_params"] = best_params

            # PyTorch specific tuned params
            if framework == "pytorch":
                if best_params is None:
                    best_params = dict(PYTORCH_PARAMS)
                best_params['max_epochs'] = PYTORCH_PARAMS['max_epochs']
                train_kwargs["pytorch_params"] = best_params

            train_out = module.train(**train_kwargs)
            if isinstance(train_out, tuple) and len(train_out) == 2:
                result, final_metrics = train_out
            else:
                result, final_metrics = train_out, {}
            self.logger.info("Training completed successfully.")

            # Guardar el modelo en S3 al finalizar
            self._save_model(result, framework)

            # Log final en MLflow (sin artifacts)
            # Prefer métricas ya calculadas por el módulo; si no hay, usamos las del Result.

            mlflow_payload = {
                **self.params,
                "name": train_kwargs.get("name", framework),
                "xgboost_params": train_kwargs.get("xgboost_params"),
                "pytorch_params": train_kwargs.get("pytorch_params"),
            }

            mc_time_sec = final_metrics.get("multiclass_metrics_time_sec")
            if mc_time_sec is not None:
                self.logger.info(
                    "%s multiclass metrics time = %.2f s",
                    framework,
                    float(mc_time_sec),
                )

            # If tuning is disabled, log default params as the effective params.
            
            if framework == "xgboost" and not mlflow_payload.get("xgboost_params"):
                mlflow_payload["xgboost_params"] = dict(XGBOOST_PARAMS)
            if framework == "pytorch" and not mlflow_payload.get("pytorch_params"):
                mlflow_payload["pytorch_params"] = dict(PYTORCH_PARAMS)
                
            self._log_final_to_mlflow(framework=framework, params=mlflow_payload, metrics=final_metrics)

            return result
        except Exception as e:
            self.logger.error(f'Training job failed: {str(e)}', exc_info=True)
            raise

def main():
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    data_dir = "s3://k8s-mlops-platform-bucket/v1/processed/" #Para kuberay
    output_dir = "s3://k8s-mlops-platform-bucket/v1/models" #Para el modelo

    model = KubeRayTraining(
        params_path="/home/ray/app/repo/k3s/params.yaml",
        data_dir=data_dir,
        output_dir=output_dir
    )
    model.train()

if __name__ == "__main__":
    main()