import json
import numpy as np
import pandas as pd


class InferencePreprocessor:
    def __init__(self, artifacts_path):
        self.artifacts = self._load_artifacts(artifacts_path)

        self.cat_cols = ["protocol", "conn_state", "protocol_conn"]
        self.num_cols = [
            "src_port",
            "dst_port",
            "packet_count",
            "bytes_transferred",
            "bytes_log",
            "packet_log",
            "hour",
            "dayofweek",
            "is_weekend",
            "hour_sin",
            "hour_cos",
        ]

    def _load_artifacts(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df["bytes_log"] = np.log1p(df["bytes_transferred"])
            df["packet_log"] = np.log1p(df["packet_count"])
            df["protocol_conn"] = (
                df["protocol"].astype(str) + "_" + df["conn_state"].astype(str)
            )
            return df
        except Exception as e:
            raise RuntimeError(f"Feature engineering failed: {e}")

    def _apply_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            encoders = self.artifacts["encoders"]
            for col_name in self.cat_cols:
                mapping = encoders.get(col_name, {})
                df[f"{col_name}_idx"] = df[col_name].map(mapping).fillna(-1.0).astype(float)
            return df
        except Exception as e:
            raise RuntimeError(f"Encoding failed: {e}")

    def _apply_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            scaler = self.artifacts["scaler"]
            for col_name in self.num_cols:
                stats = scaler.get(col_name, {"mean": 0.0, "std": 1.0})
                mean = stats.get("mean", 0.0)
                std = stats.get("std", 1.0) or 1.0
                df[f"{col_name}_norm"] = (df[col_name] - mean) / std
            return df
        except Exception as e:
            raise RuntimeError(f"Scaling failed: {e}")

    def transform(self, data: list) -> pd.DataFrame:
        try:
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)

            df = self._feature_engineering(df)
            df = self._apply_encoders(df)
            df = self._apply_scaler(df)

            final_cols = [f"{c}_idx" for c in self.cat_cols] + [f"{c}_norm" for c in self.num_cols]
            return df[final_cols].fillna(0.0)
        except Exception as e:
            raise RuntimeError(f"Transformation failed: {e}")
