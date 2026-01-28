import pickle
import torch
from k3s.kuberay.pytorch_models.models import NeuralNetwork


class PyTorchHandler:
    def __init__(self, model_path, input_dim=14, num_classes=6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path, input_dim, num_classes)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path, input_dim, num_classes):
        try:
            with open(model_path, "rb") as f:
                state_dict = pickle.load(f)
            model = NeuralNetwork(input_dim=input_dim, num_classes=num_classes)
            model.load_state_dict(state_dict)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model from {model_path}: {e}")

    def predict(self, input_data):
        try:
            if isinstance(input_data, dict):
                data_list = list(input_data.values())
                tensor_data = torch.tensor([data_list], dtype=torch.float32)
            elif isinstance(input_data, list):
                tensor_data = torch.tensor(input_data, dtype=torch.float32)
            else:
                raise ValueError("Unsupported input format")

            tensor_data = tensor_data.to(self.device)
            with torch.no_grad():
                outputs = self.model(tensor_data)
                probs = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probs, dim=1)

            return {
                "predictions": predictions.cpu().numpy().tolist(),
                "probabilities": probs.cpu().numpy().tolist(),
            }
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
