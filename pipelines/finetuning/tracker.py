import json
from statistics import mean
import mlflow


def start_training_job(job_id: str, base_model: str, pairs: list[dict]) -> str:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("neuroflow-finetuning")

    with mlflow.start_run(run_name=f"finetune-{job_id}") as run:
        quality_scores = [p["quality_score"] for p in pairs if "quality_score" in p]
        mlflow.log_params({
            "base_model": base_model,
            "training_pair_count": len(pairs),
            "avg_quality_score": round(mean(quality_scores), 4) if quality_scores else 0,
            "job_id": job_id,
        })
        jsonl_path = f"training_data/{job_id}.jsonl"
        try:
            mlflow.log_artifact(jsonl_path)
        except Exception:
            pass

        return run.info.run_id


def log_job_completion(mlflow_run_id: str, metrics: dict):
    mlflow.set_tracking_uri("http://localhost:5000")
    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metrics({
            "training_loss": metrics.get("training_loss", 0),
            "validation_loss": metrics.get("validation_loss", 0),
            "training_token_count": metrics.get("trained_tokens", 0),
        })


def register_model(mlflow_run_id: str, job_id: str):
    mlflow.set_tracking_uri("http://localhost:5000")
    try:
        mlflow.register_model(
            f"runs:/{mlflow_run_id}/model",
            f"neuroflow-finetune-{job_id}",
        )
    except Exception as e:
        print(f"Model registration failed (non-fatal): {e}")
