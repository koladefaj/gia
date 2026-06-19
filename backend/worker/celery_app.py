from celery import Celery

from backend.app.config import settings

celery_app = Celery(
    "gia",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "backend.worker.tasks.memory_extraction",
        "backend.worker.tasks.mood_inference",
        "backend.worker.tasks.proactive_check",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_ack_late=True,
    beat_schedule={
        # Mood inference runs every 30 minutes during active use
        "mood-inference-periodic": {
            "task": "backend.worker.tasks.mood_inference.run_mood_inference_all",
            "schedule": 1800.0,
        },
    },
)
