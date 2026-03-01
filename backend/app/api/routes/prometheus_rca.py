"""
User-facing Prometheus metrics API. Returns metrics text for a time window.
RCA and recommendations are provided by the chat DB health agent, not by this endpoint.
"""
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from app.api.deps import CurrentUserDep, SessionDep
from app.rag.chat.slow_query_prometheus import run_promql_range
from app.site_settings import SiteSetting

logger = logging.getLogger(__name__)

router = APIRouter()

# Minimal default queries for the API only; the chat agent chooses metrics via run_promql.
DEFAULT_API_QUERIES: list[tuple[str, str]] = [
    ("Duration P99 (TiDB)", "histogram_quantile(0.99, sum(rate(tidb_server_handle_query_duration_seconds_bucket[5m])) by (le, instance)) or vector(0)"),
    ("CPU (TiDB)", 'sum(rate(process_cpu_seconds_total{job=~"tidb.*"}[5m])) by (instance) * 100 or vector(0)'),
    ("Memory (TiDB)", 'process_resident_memory_bytes{job=~"tidb.*"} / 1024 / 1024'),
    ("QPS (TiDB)", "sum(rate(tidb_executor_statement_total[5m])) by (type) or vector(0)"),
    ("Connections (TiDB)", "sum(tidb_server_connections) by (instance) or vector(0)"),
]


@router.get("/prometheus/metrics-and-rca")
def get_prometheus_metrics_and_rca(
    session: SessionDep,
    user: CurrentUserDep,
    last_minutes: Optional[int] = Query(30, ge=1, le=10080, description="Time window in minutes (default 30)"),
    start_ts: Optional[str] = Query(None, description='Start time UTC "YYYY-MM-DD HH:MM:SS"'),
    end_ts: Optional[str] = Query(None, description='End time UTC "YYYY-MM-DD HH:MM:SS"'),
    user_question: Optional[str] = Query(None, description="Reserved for future use"),
    prometheus_host_name: Optional[str] = Query(None, description="Prometheus host name from site settings"),
) -> dict[str, Any]:
    """
    Fetch Prometheus metrics for a time window. Returns metrics_text only.
    For RCA and recommendations, use the chat DB health agent.
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
    if not name and hosts:
        name = str((hosts[0] or {}).get("name", "")).strip()

    parts: list[str] = []
    for label, query in DEFAULT_API_QUERIES:
        text = run_promql_range(name, query, start_ts, end_ts, logger=logger)
        parts.append(f"{label}:\n{text}")
    metrics_text = "Prometheus TiDB metrics:\n\n" + "\n\n".join(parts)

    return {
        "start": start_ts,
        "end": end_ts,
        "prometheus_host": name,
        "metrics_text": metrics_text,
    }
