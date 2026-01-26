import traceback

import ray
from ray import serve
import os
import boto3
import tempfile
import importlib
from typing import Dict
from starlette.requests import Request
from kuberay.utils import create_logger, BaseUtils

from inference.modules.preprocessor import InferencePreprocessor
from inference.modules.pytorch import PyTorchHandler
from inference.modules.xgboost import XGBoostHandler
# Export variables
FRAMEWORK = os.getenv("FRAMEWORK", "xgboost")
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "mlops-platform")
# Defaults assume standard paths
MODEL_KEY = os.getenv("MODEL_KEY", f"models/model_{FRAMEWORK}.pkl")
ARTIFACTS_KEY = os.getenv("ARTIFACTS_KEY", "processed/pipeline_model.json")

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1})
class InferenceService(BaseUtils):
    def __init__(self, params_path: str, artifacts_path: str):
        logger = create_logger("InferenceService")
        super().__init__(logger, params_path)
        self.params = self.load_params(params_path)['kuberay']['model']
        self.framework = self.params.get("framework", "xgboost")
        self.s3_client = self._init_s3()
        self.bucket = os.getenv("S3_BUCKET_NAME", "k8s-mlops-platform-bucket")
        # Download resources
        self.model_path = self._download_file(f"models/model_{self.framework}.pkl", f"model_{self.framework}.pkl")
        self.artifacts_path = self._download_file(artifacts_path)
        
        # Initialize components
        self.preprocessor = InferencePreprocessor(self.artifacts_path)
        self.handler = self._load_handler()

    def _init_s3(self):
        try:
            return boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION', 'us-east-2'),
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize S3 client: {e}")
            raise

    def _download_file(self, s3_key, local_name):
        bucket = self.bucket
        if not bucket or not s3_key:
             # Fallback for local testing if needed
             return f"/tmp/{local_name}"
             
        local_path = os.path.join(tempfile.gettempdir(), local_name)
        print(f"Downloading s3://{bucket}/{s3_key} to {local_path}...")
        try:
            self.s3_client.download_file(bucket, s3_key, local_path)
        except Exception as e:
            print(f"Failed to download {s3_key}: {e}")
            raise
        return local_path

    def _load_handler(self):
        try:
            if self.framework == "xgboost":
                return XGBoostHandler(self.model_path)
            elif self.framework == "pytorch":
                return PyTorchHandler(self.model_path)
            else:
                raise ValueError(f"Unsupported framework: {self.framework}")
        except Exception as e:
            self.logger.error(f"Failed to load handler for framework {self.framework}: {e}")
            raise

    async def __call__(self, http_request: Request) -> Dict:
        json_input = await http_request.json()
        
        # Expecting generic raw data: {"data": [{"src_port": 80, ...}]}
        raw_data = json_input.get("data")
        if raw_data is None:
            return {"error": "Missing 'data' field in JSON body"}
            
        try:
            # 1. Preprocess Raw Data -> Numeric Vectors
            processed_df = self.preprocessor.transform(raw_data)
            
            # 2. Predict using Vectors
            # Handlers expect list of lists for compatibility
            result = self.handler.predict(processed_df.values.tolist())
            
            return result
        except Exception as e:
            traceback.print_exc()
            return {"error": str(e)}

# Entrypoint for Serve
app = InferenceService.bind()
