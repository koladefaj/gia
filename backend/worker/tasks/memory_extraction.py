from backend.worker.celery_app import celery_app


@celery_app.task(name="backend.worker.tasks.memory_extraction.extract_session_memories")
def extract_session_memories(user_id: str, session_id: str) -> dict:
    """Extract durable preferences from a completed session. Implemented Day 3."""
    return {"status": "stub", "user_id": user_id, "session_id": session_id}
