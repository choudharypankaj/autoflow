"""
User-facing Prometheus metrics + RCA API.
Allows clients (e.g. CLI, frontend) to fetch Prometheus metrics and RCA summary for a time range,
similar to a readonly Prometheus proxy that returns processed metrics and recommendations.
"""
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from app.api.deps import CurrentUserDep, SessionDep
from app.rag.chat.slow_query_prometheus import (
    build_prometheus_tidb_metrics_analysis,
    build_rca_summary_from_metrics,
)
from app.site_settings import SiteSetting

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/prometheus/metrics-and-rca")
def get_prometheus_metrics_and_rca(
    session: SessionDep,
    user: CurrentUserDep,
    last_minutes: Optional[int] = Query(30, ge=1, le=10080, description="Time window in minutes (default 30)"),
    start_ts: Optional[str] = Query(None, description='Start time UTC "YYYY-MM-DD HH:MM:SS"'),
    end_ts: Optional[str] = Query(None, description='End time UTC "YYYY-MM-DD HH:MM:SS"'),
    user_question: Optional[str] = Query(None, description="Optional question for metric selection (RCA discovery)"),
    prometheus_host_name: Optional[str] = Query(None, description="Prometheus host name from site settings"),
) -> dict[str, Any]:
    """
    Fetch Prometheus metrics for a time window and return metrics text + RCA summary.
    Intended for CLI or frontend to access Prometheus (readonly) and get recommendations/RCA.
    Requires authentication.
    """
    SiteSetting.update_db_cache()
    hosts = getattr(SiteSetting, "prometheus_hosts", None) or []
    if not hosts:
        raise HTTPException(
            status_code=400,
            detail="No Prometheus host configured. Add prometheus_hosts in site settings.",
        )
    if start_ts and end_ts:
        try:
            datetime.strptime(start_ts, "%Y-%m-%d %H:%M:%S")
            datetime.strptime(end_ts, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail='start_ts and end_ts must be "YYYY-MM-DD HH:MM:SS" UTC',
            )
    else:
        end_dt = datetime.now(UTC)
        start_dt = end_dt - timedelta(minutes=last_minutes or 30)
        start_ts = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_ts = end_dt.strftime("%Y-%m-%d %H:%M:%S")

    name = prometheus_host_name
    if name:
        entry = next(
            (it for it in hosts if str((it or {}).get("name", "")).strip().lower() == name.strip().lower()),
            None,
        )
    else:
        entry = hosts[0]
        name = str((entry or {}).get("name", "")).strip()

    metrics_text = build_prometheus_tidb_metrics_analysis(
        start_ts,
        end_ts,
        name,
        logger,
        cluster_hint=None,
        session=session,
        user_question=user_question,
    )
    rca_summary = build_rca_summary_from_metrics(metrics_text, user_question)

    return {
        "start": start_ts,
        "end": end_ts,
        "prometheus_host": name,
        "metrics_text": metrics_text,
        "rca_summary": rca_summary,
    }
