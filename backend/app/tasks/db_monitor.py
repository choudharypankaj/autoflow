import logging
from typing import Any, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import text
from sqlmodel import Session

from app.celery import app as celery_app
from app.core.db import engine
from app.site_settings import SiteSetting
from app.core.config import settings

try:
    import sentry_sdk  # type: ignore
except Exception:  # pragma: no cover
    sentry_sdk = None  # type: ignore


logger = logging.getLogger(__name__)


Operator = Literal["gt", "gte", "lt", "lte", "eq", "ne"]


class MonitorQuery(BaseModel):
    name: str
    sql: str
    operator: Operator = "gt"
    threshold: float | int
    enabled: bool = True
    description: Optional[str] = None
    # If True, will take the first column of the first row as value (default behavior).
    # If False, will try to use COUNT(*) if available; fallback to first column.
    use_scalar_value: bool = True
    timeout_seconds: int = Field(default=10, ge=1, le=120)


def _compare(value: float | int, operator: Operator, threshold: float | int) -> bool:
    if operator == "gt":
        return value > threshold
    if operator == "gte":
        return value >= threshold
    if operator == "lt":
        return value < threshold
    if operator == "lte":
        return value <= threshold
    if operator == "eq":
        return value == threshold
    if operator == "ne":
        return value != threshold
    return False


def _extract_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    # Try to convert numeric-like strings
    try:
        return float(str(value))
    except Exception:
        return None


def _run_single_query(session: Session, cfg: MonitorQuery) -> Tuple[bool, Optional[float], Optional[str]]:
    sql_lower = cfg.sql.strip().lower()
    if not sql_lower.startswith("select"):
        return False, None, f"Non-SELECT SQL is not allowed in monitoring: {cfg.name}"

    try:
        rs = session.exec(text(cfg.sql))  # type: ignore[arg-type]
        first_row = rs.first()
        if first_row is None:
            return False, None, "Query returned no rows"

        raw_value: Any
        # Prefer scalar first column
        if isinstance(first_row, (tuple, list)):
            raw_value = first_row[0]
        elif hasattr(first_row, "_mapping"):
            # SQLAlchemy Row
            # Take the first column
            first_key = next(iter(first_row._mapping.keys()))
            raw_value = first_row._mapping[first_key]
        else:
            raw_value = first_row

        value = _extract_numeric(raw_value)
        if value is None:
            return False, None, f"Non-numeric result for '{cfg.name}': {raw_value}"

        ok = _compare(value, cfg.operator, cfg.threshold)
        return ok, value, None
    except Exception as e:
        return False, None, f"Exception running '{cfg.name}': {e}"


def run_db_monitor_once() -> List[dict]:
    results: List[dict] = []

    enabled = bool(SiteSetting.db_monitor_enabled)
    if not enabled:
        logger.debug("DB monitor disabled. Skipping run.")
        return results

    raw_items = SiteSetting.db_monitor_queries or []

    query_cfgs: List[MonitorQuery] = []
    for item in raw_items:
        try:
            query_cfgs.append(MonitorQuery(**item))
        except ValidationError as e:
            logger.warning("Invalid monitor item ignored: %s", e)

    if not query_cfgs:
        logger.info("DB monitor enabled but no queries configured.")
        return results

    notify_sentry = bool(SiteSetting.db_monitor_notify_sentry) and bool(settings.SENTRY_DSN)

    with Session(engine, expire_on_commit=False) as session:
        for cfg in query_cfgs:
            if not cfg.enabled:
                continue

            ok, value, error = _run_single_query(session, cfg)

            result = {
                "name": cfg.name,
                "ok": ok,
                "value": value,
                "operator": cfg.operator,
                "threshold": cfg.threshold,
                "error": error,
            }
            results.append(result)

            if ok:
                logger.info("[DB Monitor] PASS %s value=%s %s %s", cfg.name, value, cfg.operator, cfg.threshold)
            else:
                msg = f"[DB Monitor] FAIL {cfg.name} value={value} {cfg.operator} {cfg.threshold} error={error}"
                logger.warning(msg)
                if notify_sentry and sentry_sdk:
                    try:
                        sentry_sdk.capture_message(msg)  # type: ignore[attr-defined]
                    except Exception:
                        # Do not break monitoring flow if sentry capture fails
                        logger.debug("Sentry capture failed for: %s", cfg.name)

    return results


@celery_app.task(name="app.tasks.db_monitor.run_db_monitor")
def run_db_monitor():
    return run_db_monitor_once()

