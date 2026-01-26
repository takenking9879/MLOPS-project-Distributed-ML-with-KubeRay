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
import mlflow
from ray.train.xgboost import RayTrainReportCallback

from schemas.pytorch_params import PYTORCH_PARAMS
from schemas.xgboost_params import XGBOOST_PARAMS

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
            # Optional: materialize now so downstream consumers (Train + metrics)
            # can reuse blocks from the object store instead of triggering multiple
            # `ReadParquet` executions.
            # Enable with: RAY_MATERIALIZE_DATASETS=1
            #if os.getenv("RAY_MATERIALIZE_DATASETS", "0") in ("1", "true", "True"):
                #ds = ds.materialize()
            self.logger.info(f"Data loaded from {path}.")
            return ds
        except Exception as e:
            self.logger.error(f"Failed to load data from {path}: {str(e)}", exc_info=True)
            raise

    def _save_model(self, result, framework):
        """
        Extrae el mejor modelo del resultado y lo guarda en S3 como un archivo .pkl.
        """
        self.logger.info(f"Exportando modelo final de {framework} a S3...")
        try:
            filename = f"model_{framework}.pkl"
            local_path = os.path.join(tempfile.gettempdir(), filename)
            checkpoint = result.checkpoint
            if not checkpoint:
                self.logger.warning("No se encontró un checkpoint válido en el resultado.")
                return
            if framework == "xgboost":
                # Ray Train stores the XGBoost model inside a generic `Checkpoint`.
                # Use RayTrainReportCallback.get_model(checkpoint) to load the Booster.
                model = RayTrainReportCallback.get_model(checkpoint)
                with open(local_path, "wb") as f:
                    pickle.dump(model, f)
            elif framework == "pytorch":
                # Para PyTorch u otros, guardamos el diccionario del checkpoint
                # que contiene los pesos/state_dict.
                with open(local_path, "wb") as f:
                    pickle.dump(checkpoint.to_dict(), f)

            # Subir a S3 usando el cliente boto3 ya configurado
            parsed_url = urlparse(self.output_dir)
            bucket = parsed_url.netloc
            prefix = parsed_url.path.lstrip('/')
            s3_key = os.path.join(prefix, filename)

            self.s3.upload_file(local_path, bucket, s3_key)
            self.logger.info(f"Modelo guardado exitosamente en: s3://{bucket}/{s3_key}")
            
            if os.path.exists(local_path):
                os.remove(local_path)
                
        except Exception as e:
            self.logger.error(f"Error en el export del modelo: {str(e)}", exc_info=True)

    def _log_final_to_mlflow(self, *, framework: str, params: Dict[str, Any], metrics: Dict[str, float]) -> None:
        tracking_uri = params.get("mlflow_tracking_uri")
        experiment_name = params.get("mlflow_experiment_name")
        if not tracking_uri or not experiment_name:
            return

        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

            run_name = f"{params.get('name', framework)}_final_{framework}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("framework", framework)
                mlflow.log_param("target", params.get("target"))
                mlflow.log_param("num_classes", params.get("num_classes"))
                mlflow.log_param("num_workers", os.getenv("NUM_WORKERS", ""))
                mlflow.log_param("cpus_per_worker", os.getenv("CPUS_PER_WORKER", ""))

                # Log hyperparameters (flatten safely)
                if framework == "xgboost" and params.get("xgboost_params"):
                    for k, v in params["xgboost_params"].items():
                        mlflow.log_param(f"xgb_{k}", v)
                if framework == "pytorch" and params.get("pytorch_params"):
                    for k, v in params["pytorch_params"].items():
                        mlflow.log_param(f"pt_{k}", v)

                # Log metrics (NO artifacts)
                for k, v in metrics.items():
                    mlflow.log_metric(k, float(v))

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

            train_ds = self._load_data(os.path.join(self.data_dir, 'train'))
            val_ds = self._load_data(os.path.join(self.data_dir, 'val'))

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

            train_kwargs = {
                "train_dataset": train_ds,
                "val_dataset": val_ds,
                "storage_path": self.output_dir,
                "name": self.params.get('name', framework),
                "target": self.params.get('target', 'attack'),
                "num_classes": num_classes,
            }

            # XGBoost specific tuned params
            if framework == "xgboost":
                train_kwargs["xgboost_params"] = best_params

            # PyTorch specific tuned params
            if framework == "pytorch":
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
            self.logger.info("%s multiclass metrics time = %.2f s",framework,float(mc_time_sec))

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