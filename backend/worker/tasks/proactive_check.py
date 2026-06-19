from backend.worker.celery_app import celery_app


@celery_app.task(name="backend.worker.tasks.proactive_check.check_pattern_shift")
def check_pattern_shift(user_id: str) -> dict:
    """Detect mood pattern deviations and draft proactive message. Implemented Day 6."""
    return {"status": "stub", "user_id": user_id}
