"""
Prometheus-based TiDB metrics analysis using the Prometheus HTTP API.
Queries are flexible and derived from the user question; no MCP server required.
"""
import logging
import re
from datetime import UTC, datetime
from typing import Any

from app.prometheus.client import prometheus_query_range
from app.site_settings import SiteSetting

# Reuse series extraction and summarization from Grafana module (same Prometheus response shape).
from app.rag.chat.slow_query_grafana import (
    _extract_series_values,
    _summarize_cpu_series,
    _summarize_panel_series,
)


def _build_prometheus_vars(cluster_hint: str | None) -> dict[str, str]:
    vars_map = dict(getattr(SiteSetting, "prometheus_vars", None) or {})
    if cluster_hint:
        lowered = cluster_hint.strip()
        for key in list(vars_map.keys()):
            if "cluster" in str(key).lower():
                vars_map[key] = lowered
        if not vars_map:
            vars_map = {
                "tidb_cluster": lowered,
                "cluster": lowered,
                "k8s_cluster": lowered,
            }
    return vars_map


def _apply_vars_to_expr(expr: str, vars_map: dict[str, str]) -> str:
    out = expr
    for k, v in (vars_map or {}).items():
        out = out.replace(f"${k}", str(v))
        out = out.replace(f"${{{k}}}", str(v))
    return out


# (label, summary_type, promql_template). summary_type: duration | cpu | gauge_per_instance
# Templates may use $tidb_cluster, $instance, etc. from prometheus_vars.
PROMETHEUS_METRIC_CONFIGS: list[tuple[str, str, str]] = [
    (
        "Duration (TiDB)",
        "duration",
        'histogram_quantile(0.99, sum(rate(tidb_server_handle_query_duration_seconds_bucket[5m])) by (le, instance)) or vector(0)',
    ),
    (
        "Duration P95 (TiDB)",
        "duration",
        'histogram_quantile(0.95, sum(rate(tidb_server_handle_query_duration_seconds_bucket[5m])) by (le, instance)) or vector(0)',
    ),
    (
        "CPU (TiDB)",
        "cpu",
        'sum(rate(process_cpu_seconds_total{job=~"tidb.*"}[5m])) by (instance) * 100 or vector(0)',
    ),
    (
        "CPU (TiKV)",
        "cpu",
        'sum(rate(process_cpu_seconds_total{job=~"tikv.*"}[5m])) by (instance) * 100 or vector(0)',
    ),
    (
        "Memory (TiDB)",
        "gauge_per_instance",
        'process_resident_memory_bytes{job=~"tidb.*"} / 1024 / 1024',
    ),
    (
        "Memory (TiKV)",
        "gauge_per_instance",
        'process_resident_memory_bytes{job=~"tikv.*"} / 1024 / 1024',
    ),
    (
        "QPS (TiDB)",
        "gauge",
        'sum(rate(tidb_executor_statement_total[5m])) by (type) or vector(0)',
    ),
    (
        "Connections (TiDB)",
        "gauge",
        'sum(tidb_server_connections) by (instance) or vector(0)',
    ),
]


def infer_prometheus_metrics_from_user_question(
    user_question: str | None,
) -> list[tuple[str, str, str]] | None:
    """
    Infer which Prometheus metrics to run from the user's question.
    Returns list of (label, summary_type, promql_template) or None for default set.
    """
    if not user_question or not isinstance(user_question, str):
        return None
    q = user_question.strip().lower()
    if not q:
        return None

    wants_duration = any(
        re.search(p, q)
        for p in [r"\bduration\b", r"\blatency\b", r"\bp99\b", r"\bp95\b", r"\bpercentile\b"]
    )
    wants_cpu_tidb = bool(re.search(r"\btidb\b", q) and re.search(r"\bcpu\b", q))
    wants_cpu_tikv = bool(re.search(r"\btikv\b", q) and re.search(r"\bcpu\b", q))
    wants_cpu_any = bool(re.search(r"\bcpu\b", q) and not wants_cpu_tidb and not wants_cpu_tikv)
    wants_memory = any(
        re.search(p, q) for p in [r"\bmemory\b", r"\bram\b", r"\bheap\b"]
    )
    wants_tidb = bool(re.search(r"\btidb\b", q))
    wants_tikv = bool(re.search(r"\btikv\b", q))
    wants_qps = any(
        re.search(p, q)
        for p in [r"\bqps\b", r"\bthroughput\b", r"\bqueries?\s*per\s*sec", r"\btps\b"]
    )
    wants_connection = any(
        re.search(p, q) for p in [r"\bconnection\b", r"\bconn\b", r"\bconnect\b"]
    )

    selected: list[tuple[str, str, str]] = []
    for label, summary_type, promql in PROMETHEUS_METRIC_CONFIGS:
        if "Duration (TiDB)" in label and "P95" not in label and wants_duration:
            selected.append((label, summary_type, promql))
        elif "Duration P95" in label and wants_duration:
            selected.append((label, summary_type, promql))
        elif "CPU (TiDB)" in label and (wants_cpu_tidb or (wants_cpu_any and wants_tidb)):
            selected.append((label, summary_type, promql))
        elif "CPU (TiKV)" in label and (wants_cpu_tikv or (wants_cpu_any and wants_tikv)):
            selected.append((label, summary_type, promql))
        elif "Memory (TiDB)" in label and wants_memory and wants_tidb:
            selected.append((label, summary_type, promql))
        elif "Memory (TiKV)" in label and wants_memory and wants_tikv:
            selected.append((label, summary_type, promql))
        elif "QPS" in label and wants_qps:
            selected.append((label, summary_type, promql))
        elif "Connections" in label and wants_connection:
            selected.append((label, summary_type, promql))

    if selected:
        return selected
    if re.search(r"\bprometheus\b|\bmetrics?\b|\bmonitoring\b", q):
        return None
    return None


def build_prometheus_tidb_metrics_analysis(
    start_time: str,
    end_time: str,
    prometheus_host: str | None,
    logger: logging.Logger,
    cluster_hint: str | None = None,
    session: Any | None = None,
    user_question: str | None = None,
) -> str:
    """
    Run flexible Prometheus queries via HTTP API and return a text summary.
    Which metrics are run is inferred from user_question when provided.
    """
    SiteSetting.update_db_cache()
    hosts = getattr(SiteSetting, "prometheus_hosts", None) or []
    name = prometheus_host
    entry = None
    if name:
        for it in hosts:
            if str((it or {}).get("name", "")).strip().lower() == name.strip().lower():
                entry = it
                break
    if not entry and hosts:
        entry = hosts[0]
        name = str((entry or {}).get("name", "")).strip() or None
    if not entry:
        return "Prometheus TiDB metrics:\n\n- Prometheus host not configured. Add prometheus_hosts in site settings."

    try:
        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        start_sec = start_dt.timestamp()
        end_sec = end_dt.timestamp()
    except Exception:
        return "Prometheus TiDB metrics:\n\n- Invalid time window; expected 'YYYY-MM-DD HH:MM:SS' UTC."

    base_url = str((entry or {}).get("prometheus_url", "")).strip().rstrip("/")
    bearer_token = str((entry or {}).get("bearer_token", "")).strip() or None
    if not base_url or not base_url.startswith("http"):
        return "Prometheus TiDB metrics:\n\n- Prometheus URL missing or invalid for this host."

    vars_map = _build_prometheus_vars(cluster_hint)
    metric_configs = infer_prometheus_metrics_from_user_question(user_question)
    if not metric_configs:
        metric_configs = [
            (label, stype, promql)
            for label, stype, promql in PROMETHEUS_METRIC_CONFIGS
            if label in ("Duration (TiDB)", "CPU (TiDB)", "CPU (TiKV)")
        ]
    logger.info(
        "Prometheus metrics to run (user_question=%s): %s",
        user_question[:80] if user_question else None,
        [c[0] for c in metric_configs],
    )

    metrics: list[str] = []
    step_sec = 60

    for label, summary_type, promql_template in metric_configs:
        expr = _apply_vars_to_expr(promql_template, vars_map)
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
            logger.exception("Prometheus query_range failed: host=%s label=%s", name, label)
            metrics.append(f"{label}:\n- Query failed: {e}")
            continue

        # Normalize to same shape as Grafana panel response: list of series (each = Prometheus result).
        result = data.get("data", {}).get("result") if isinstance(data, dict) else None
        if not isinstance(result, list):
            metrics.append(f"{label}:\n- No data returned.")
            continue
        series = [{"data": {"result": [item]}} for item in result] if result else []
        if not series:
            metrics.append(f"{label}:\n- No series data.")
            continue

        if summary_type == "duration":
            values = _extract_series_values(series)
            if not values:
                metrics.append(f"{label}:\n- No data points.")
            else:
                avg_ms = (sum(values) / len(values)) * 1000.0
                max_ms = max(values) * 1000.0
                metrics.append(f"{label}:\n- avg: {avg_ms:.2f} ms\n- max: {max_ms:.2f} ms")
        elif summary_type == "cpu":
            text = _summarize_cpu_series(series, None, logger)
            metrics.append(f"{label}:\n{text}")
        elif summary_type == "gauge_per_instance":
            text = _summarize_panel_series(series, per_instance=True)
            metrics.append(f"{label}:\n{text}")
        else:
            text = _summarize_panel_series(series, per_instance=False)
            metrics.append(f"{label}:\n{text}")

    return "Prometheus TiDB metrics:\n\n" + "\n\n".join(metrics)
