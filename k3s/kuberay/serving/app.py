import os
import random
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import boto3

from ray import serve
from starlette.requests import Request

from k3s.kuberay.utils import create_logger
from k3s.kuberay.serving.modules.preprocessor import InferencePreprocessor
from k3s.kuberay.serving.modules.xgboost import XGBoostHandler


@dataclass(frozen=True)
class ModelSpec:
    framework: str
    model_key: str
    artifacts_key: str


class S3Store:
    def __init__(self, *, bucket: str, endpoint_url: Optional[str] = None):
        self._logger = create_logger("S3Store")
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-2"),
            endpoint_url=endpoint_url or os.getenv("S3_ENDPOINT_URL") or None,
        )

    def download_to_tmp(self, *, key: str, filename: str) -> str:
        local_path = os.path.join(tempfile.gettempdir(), filename)
        self._logger.info("Downloading s3://%s/%s -> %s", self._bucket, key, local_path)
        self._client.download_file(self._bucket, key, local_path)
        return local_path


def _normalize_payload(payload: Any) -> List[Dict[str, Any]]:
    # Expect: {"data": [ {...}, {...} ]} or {"data": {...}}
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    data = payload.get("data")
    if data is None:
        raise ValueError("Missing 'data' field")
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("'data' must be an object or a list of objects")


class _ModelRuntime:
    """Shared (non-deployment) runtime.

    IMPORTANT: Do not subclass a class decorated by @serve.deployment.
    Ray wraps deployment classes, and Python inheritance breaks with that wrapper.
    """

    def __init__(self, *, name: str, variant: str):
        self._logger = create_logger(name)
        self._variant = variant
        self._store: Optional[S3Store] = None
        self._pre: Optional[InferencePreprocessor] = None
        self._handler: Optional[XGBoostHandler] = None
        self._spec: Optional[ModelSpec] = None

    def _load_from_config(self, config: Dict[str, Any]) -> None:
        bucket = os.getenv("S3_BUCKET_NAME", "k8s-mlops-platform-bucket")
        framework = str(config.get("framework", os.getenv("FRAMEWORK", "xgboost")))
        model_key = str(config.get("model_key", os.getenv("MODEL_KEY", f"models/model_{framework}.pkl")))
        artifacts_key = str(
            config.get(
                "artifacts_key",
                os.getenv("ARTIFACTS_KEY", "v1/artifacts/pipeline_model.json"),
            )
        )
        self._spec = ModelSpec(framework=framework, model_key=model_key, artifacts_key=artifacts_key)

        self._store = S3Store(bucket=bucket)
        model_path = self._store.download_to_tmp(
            key=self._spec.model_key,
            filename=f"{self._variant}_{framework}.pkl",
        )
        artifacts_path = self._store.download_to_tmp(
            key=self._spec.artifacts_key,
            filename="pipeline_model.json",
        )

        self._pre = InferencePreprocessor(artifacts_path)
        if framework != "xgboost":
            raise ValueError(f"Unsupported framework for this deployment: {framework}")
        self._handler = XGBoostHandler(model_path)

        self._logger.info("Model loaded (%s): %s", self._variant, self._spec)

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._pre is None or self._handler is None or self._spec is None:
            return {"error": "Model not initialized"}

        started = time.perf_counter()
        rows = _normalize_payload(payload)
        processed = self._pre.transform(rows)
        result = self._handler.predict(processed.values.tolist())
        result["latency_ms"] = (time.perf_counter() - started) * 1000.0
        result["model"] = {
            "variant": self._variant,
            "framework": self._spec.framework,
            "model_key": self._spec.model_key,
        }
        return result


@serve.deployment(name="StableModel")
class StableModel:
    def __init__(self):
        self._rt = _ModelRuntime(name="StableModel", variant="stable")

    def reconfigure(self, config: Dict[str, Any]) -> None:
        self._rt._load_from_config(config)

    async def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self._rt.predict(payload)
        except Exception as e:
            return {"error": str(e)}


@serve.deployment(name="CanaryModel")
class CanaryModel:
    def __init__(self):
        self._rt = _ModelRuntime(name="CanaryModel", variant="canary")

    def reconfigure(self, config: Dict[str, Any]) -> None:
        self._rt._load_from_config(config)

    async def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self._rt.predict(payload)
        except Exception as e:
            return {"error": str(e)}


@serve.deployment(name="ModelRouter")
class ModelRouter:
    def __init__(self, stable, canary):
        self._logger = create_logger("ModelRouter")
        self._stable = stable
        self._canary = canary
        self._canary_probability = 0.0

    def reconfigure(self, config: Dict[str, Any]) -> None:
        p = float(config.get("canary_probability", 0.0))
        self._canary_probability = max(0.0, min(1.0, p))
        self._logger.info("Router configured: canary_probability=%s", self._canary_probability)

    async def __call__(self, request: Request):
        if request.url.path.endswith("/healthz"):
            return {"status": "ok"}

        payload = await request.json()

        use_canary = random.random() < self._canary_probability
        handle = self._canary if use_canary else self._stable
        # Delegate prediction to the chosen model deployment.
        return await handle.predict.remote(payload)


# Serve application graph.
# Note: deployment names are pinned above to match serveConfigV2 updates.
deployment_graph = ModelRouter.bind(StableModel.bind(), CanaryModel.bind())
