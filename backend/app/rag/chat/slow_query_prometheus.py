"""
Prometheus primitives for the DB health agent. The agent uses logical reasoning to choose
which metrics to query—no predefined metric list. RCA is produced by the AI agent from gathered data.
Provides: list metric names, get metadata, run arbitrary PromQL range query.
"""
import logging
from datetime import UTC, datetime
from typing import Any

from app.prometheus.client import (
    prometheus_metadata,
    prometheus_metric_names,
    prometheus_query_range,
)
from app.site_settings import SiteSetting

from app.rag.chat.series_utils import (
    extract_series_values,
    summarize_panel_series,
)


def _get_prometheus_entry(prometheus_host: str | None):
    """Resolve prometheus_host to site-setting entry (dict with prometheus_url, bearer_token, name)."""
    SiteSetting.update_db_cache()
    hosts = getattr(SiteSetting, "prometheus_hosts", None) or []
    if prometheus_host:
        for it in hosts:
            if str((it or {}).get("name", "")).strip().lower() == prometheus_host.strip().lower():
                return it
    if hosts:
        return hosts[0]
    return None


def list_prometheus_metric_names(prometheus_host: str | None = None) -> list[str]:
    """
    List all metric names from Prometheus. Used by the agent to discover what to query.
    """
    entry = _get_prometheus_entry(prometheus_host)
    if not entry:
        return []
    base_url = str((entry or {}).get("prometheus_url", "")).strip().rstrip("/")
    bearer_token = str((entry or {}).get("bearer_token", "")).strip() or None
    if not base_url or not base_url.startswith("http"):
        return []
    try:
        return prometheus_metric_names(base_url, timeout=30, bearer_token=bearer_token)
    except Exception:
        return []


def get_prometheus_metadata(prometheus_host: str | None = None) -> dict[str, dict[str, Any]]:
    """
    Fetch metadata (type, help) for all metrics. Agent uses this to build appropriate PromQL.
    """
    entry = _get_prometheus_entry(prometheus_host)
    if not entry:
        return {}
    base_url = str((entry or {}).get("prometheus_url", "")).strip().rstrip("/")
    bearer_token = str((entry or {}).get("bearer_token", "")).strip() or None
    if not base_url or not base_url.startswith("http"):
        return {}
    try:
        return prometheus_metadata(base_url, timeout=30, bearer_token=bearer_token)
    except Exception:
        return {}


def run_promql_range(
    prometheus_host: str | None,
    query: str,
    start_ts: str,
    end_ts: str,
    *,
    step_sec: int = 60,
    vars_map: dict[str, str] | None = None,
    logger: logging.Logger | None = None,
) -> str:
    """
    Run a single PromQL range query and return a short text summary. The agent chooses the query.
    start_ts / end_ts: "YYYY-MM-DD HH:MM:SS" UTC.
    """
    entry = _get_prometheus_entry(prometheus_host)
    if not entry:
        return "Prometheus host not configured or not found. Use list_hosts to see prometheus_hosts."
    base_url = str((entry or {}).get("prometheus_url", "")).strip().rstrip("/")
    bearer_token = str((entry or {}).get("bearer_token", "")).strip() or None
    if not base_url or not base_url.startswith("http"):
        return "Prometheus URL missing or invalid for this host."
    try:
        start_dt = datetime.strptime(start_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        start_sec = start_dt.timestamp()
        end_sec = end_dt.timestamp()
    except Exception:
        return "Invalid time window; use 'YYYY-MM-DD HH:MM:SS' UTC for start_ts and end_ts."
    expr = query.strip()
    if vars_map:
        for k, v in vars_map.items():
            expr = expr.replace(f"${k}", str(v)).replace(f"${{{k}}}", str(v))
    if not expr:
        return "Empty query."
    try:
        data = prometheus_query_range(
            base_url,
            expr,
            start_sec=start_sec,
            end_sec=end_sec,
            step_sec=step_sec,
            bearer_token=bearer_token,
        )
    except Exception as e:
        if logger:
            logger.exception("Prometheus query_range failed: query=%s", query[:200])
        return f"Query failed: {e}"
    result = data.get("data", {}).get("result") if isinstance(data, dict) else None
    if not isinstance(result, list):
        return "No data returned."
    series = [{"data": {"result": [item]}} for item in result] if result else []
    if not series:
        return "No series data."
    values = extract_series_values(series)
    if values:
        avg_v = sum(values) / len(values)
        max_v = max(values)
        if "duration" in query.lower() or "latency" in query.lower() or "seconds" in query.lower():
            return f"- avg: {avg_v * 1000:.2f} ms\n- max: {max_v * 1000:.2f} ms\n- points: {len(values)}"
        return f"- avg: {avg_v:.6f}\n- max: {max_v:.6f}\n- points: {len(values)}"
    text = summarize_panel_series(series, per_instance=True)
    return text if text else "No data points."
