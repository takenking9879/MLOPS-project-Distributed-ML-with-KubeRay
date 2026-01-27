import os
import json
import tempfile
import numbers
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from typing import Dict, Any

def log_evaluation_artifacts(metrics: Dict[str, Any], framework: str):
    """
    Generates and logs confusion matrix plots and classification reports to MLflow.
    """
    cm_val = metrics.get("confusion_matrix")
    report_val = metrics.get("classification_report")
    cm_test = metrics.get("test_confusion_matrix")
    report_test = metrics.get("test_classification_report")

    if not any([cm_val, report_val, cm_test, report_test]):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        def _log_single_eval(prefix: str, cm_obj, report_text) -> None:
            if report_text:
                report_path = os.path.join(tmpdir, f"{prefix}_classification_report.txt")
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(str(report_text))
                mlflow.log_artifact(report_path, artifact_path="evaluation")

            if cm_obj is not None:
                # Save JSON
                cm_json_path = os.path.join(tmpdir, f"{prefix}_confusion_matrix.json")
                with open(cm_json_path, "w", encoding="utf-8") as f:
                    json.dump(cm_obj, f)
                mlflow.log_artifact(cm_json_path, artifact_path="evaluation")

                # Save PNG
                cm_np = np.asarray(cm_obj)
                fig_w = min(14, max(6, cm_np.shape[0] * 0.75))
                fig_h = min(12, max(5, cm_np.shape[0] * 0.60))
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                im = ax.imshow(cm_np, cmap="Blues")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"{prefix.upper()} Confusion Matrix ({framework})")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_xticks(range(cm_np.shape[1]))
                ax.set_yticks(range(cm_np.shape[0]))
                ax.set_xticklabels([str(i) for i in range(cm_np.shape[1])], rotation=45, ha="right")
                ax.set_yticklabels([str(i) for i in range(cm_np.shape[0])])
                plt.tight_layout()

                png_path = os.path.join(tmpdir, f"{prefix}_confusion_matrix.png")
                fig.savefig(png_path, dpi=160)
                plt.close(fig)
                mlflow.log_artifact(png_path, artifact_path="evaluation")

        _log_single_eval("val", cm_val, report_val)
        _log_single_eval("test", cm_test, report_test)

def log_training_run(
    framework: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    artifact_location: str
):
    """
    Handles the full MLflow lifecycle for a training run.
    """
    tracking_uri = params.get("mlflow_tracking_uri")
    experiment_name = params.get("mlflow_experiment_name")
    
    if not tracking_uri or not experiment_name:
        return

    mlflow.set_tracking_uri(tracking_uri)
    
    # Ensure experiment exists
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    
    mlflow.set_experiment(experiment_name)

    run_name = f"{params.get('name', framework)}_final_{framework}"
    with mlflow.start_run(run_name=run_name):
        # 1. Log Params
        mlflow.log_param("framework", framework)
        mlflow.log_param("target", params.get("target"))
        mlflow.log_param("num_classes", params.get("num_classes"))
        mlflow.log_param("num_workers", os.getenv("NUM_WORKERS", ""))
        
        if framework == "xgboost" and params.get("xgboost_params"):
            for k, v in params["xgboost_params"].items():
                mlflow.log_param(f"xgb_{k}", v)
        elif framework == "pytorch" and params.get("pytorch_params"):
            for k, v in params["pytorch_params"].items():
                mlflow.log_param(f"pt_{k}", v)

        # 2. Log Numeric Metrics
        for k, v in (metrics or {}).items():
            if isinstance(v, numbers.Real) and not isinstance(v, bool):
                mlflow.log_metric(k, float(v))

        # 3. Log Artifacts (Plots & Reports)
        log_evaluation_artifacts(metrics, framework)
