from backend.worker.celery_app import celery_app


@celery_app.task(name="backend.worker.tasks.mood_inference.run_mood_inference")
def run_mood_inference(user_id: str) -> dict:
    """Analyse listening history, detect patterns, write to Weaviate. Implemented Day 6."""
    return {"status": "stub", "user_id": user_id}


@celery_app.task(name="backend.worker.tasks.mood_inference.run_mood_inference_all")
def run_mood_inference_all() -> dict:
    """Beat task — runs inference for all active users. Implemented Day 6."""
    return {"status": "stub"}
