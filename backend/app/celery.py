from celery import Celery
from celery.schedules import crontab

from app.core.config import settings


app = Celery(
    settings.PROJECT_NAME,
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_routes=[
        {"app.tasks.evaluate.*": {"queue": "evaluation"}},
        {"*": {"queue": "default"}},
    ],
    broker_connection_retry_on_startup=True,
    beat_schedule={
        # Run DB monitor every 5 minutes; the task will no-op if disabled by settings.
        "run-db-monitor-every-5-mins": {
            "task": "app.tasks.db_monitor.run_db_monitor",
            "schedule": 300.0,
        },
    },
)

app.autodiscover_tasks(["app"])
