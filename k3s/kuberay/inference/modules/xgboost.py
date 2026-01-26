
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import pickle

class XGBoostHandler:
    def __init__(self, model_path):
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        try: 
            # XGBoost models in this pipeline are saved as .pkl
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load XGBoost model from {model_path}: {e}")

    def predict(self, input_data):
        try:
            # input_data is a dictionary or list, convert to DataFrame/DMatrix
            if isinstance(input_data, dict):
                # Assumes single row dict
                df = pd.DataFrame([input_data])
            elif isinstance(input_data, list):
                df = pd.DataFrame(input_data)
            else:
                df = pd.DataFrame(input_data)
            
            dmatrix = xgb.DMatrix(df)
            probs = self.model.predict(dmatrix)
            
            # Multiclass: argmax
            if len(probs.shape) > 1 and probs.shape[1] > 1:
                predictions = np.argmax(probs, axis=1)
            else:
                predictions = (probs > 0.5).astype(int)
                
            return {"predictions": predictions.tolist(), "probabilities": probs.tolist()}
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")