"""
Prometheus-based TiDB metrics analysis for the DB health agent.
Used by db_health_agent tools: build_prometheus_tidb_metrics_analysis and build_rca_summary_from_metrics.
Queries are flexible and derived from the user question; no MCP server required.
"""
import logging
import re
from datetime import UTC, datetime
from typing import Any

from app.prometheus.client import (
    prometheus_metadata,
    prometheus_metric_names,
    prometheus_query_range,
)
from app.site_settings import SiteSetting

# Use shared series extraction/summarization for Prometheus response shape.
from app.rag.chat.series_utils import (
    extract_series_values,
    summarize_cpu_series,
    summarize_panel_series,
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
# This list defines all available metrics; selection is flexible from user_question
# (keyword match) or full set when user asks for generic "metrics"/"analyze".
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


# Keywords that indicate metrics useful for RCA (root cause analysis). Used to score discovered metrics.
RCA_KEYWORDS = frozenset({
    "slow", "duration", "latency", "query", "cpu", "memory", "ram", "heap", "io", "disk",
    "lock", "wait", "block", "connection", "conn", "qps", "throughput", "tps", "error", "fail",
    "replication", "replica", "tidb", "tikv", "pd", "raft", "scheduler", "compaction",
    "cache", "hit", "miss", "goroutine", "thread", "gc", "garbage",
})


def _build_promql_for_metric(metric_name: str, meta_type: str) -> tuple[str, str]:
    """
    Build a range-query PromQL and summary_type for a metric from its name and metadata type.
    Returns (promql_template, summary_type).
    """
    name = metric_name.strip()
    if not name:
        return ("vector(0)", "gauge")
    # Histogram: use _bucket and histogram_quantile
    if meta_type == "histogram" or name.endswith("_bucket"):
        base = name if name.endswith("_bucket") else f"{name}_bucket"
        return (
            f'histogram_quantile(0.99, sum(rate({base}[5m])) by (le, instance)) or vector(0)',
            "duration",
        )
    # Counter: rate
    if meta_type == "counter" or name.endswith("_total"):
        return f'sum(rate({name}[5m])) by (instance) or vector(0)', "gauge"
    # Gauge / unknown: use as-is with optional by (instance)
    if meta_type == "gauge":
        return f'sum({name}) by (instance) or vector(0)', "gauge_per_instance"
    return f'sum({name}) by (instance) or vector(0)', "gauge"


def _score_metric_for_rca(metric_name: str, help_text: str, user_question: str) -> float:
    """Score a metric by relevance to user question and RCA keywords. Higher = more relevant."""
    q = (user_question or "").strip().lower()
    name_lower = metric_name.lower()
    help_lower = (help_text or "").lower()
    tokens = set(re.findall(r"[a-z0-9_]+", q))
    tokens.update(RCA_KEYWORDS)
    score = 0.0
    for t in tokens:
        if len(t) < 2:
            continue
        if t in name_lower:
            score += 2.0
        if t in help_lower:
            score += 1.0
    return score


def discover_metrics_for_rca(
    base_url: str,
    bearer_token: str | None,
    user_question: str | None,
    vars_map: dict[str, str],
    logger: logging.Logger,
    *,
    max_metrics: int = 25,
    timeout: int = 15,
) -> list[tuple[str, str, str]]:
    """
    Discover metrics from Prometheus and return those relevant to the user's question for RCA.
    Fetches metric names and metadata, scores by relevance, builds PromQL per metric.
    Returns list of (label, summary_type, promql_template); may be empty on failure.
    """
    try:
        names = prometheus_metric_names(
            base_url,
            timeout=timeout,
            bearer_token=bearer_token,
        )
    except Exception as e:
        logger.warning("RCA discovery: failed to fetch metric names: %s", e)
        return []
    if not names:
        return []
    try:
        metadata = prometheus_metadata(
            base_url,
            timeout=timeout,
            bearer_token=bearer_token,
        )
    except Exception as e:
        logger.warning("RCA discovery: failed to fetch metadata: %s", e)
        metadata = {}
    # Score each metric
    scored: list[tuple[float, str, str, str]] = []
    for name in names:
        meta = metadata.get(name) or {}
        meta_type = (meta.get("type") or "unknown").strip().lower()
        help_text = meta.get("help") or ""
        score = _score_metric_for_rca(name, help_text, user_question or "")
        if score <= 0:
            continue
        promql, summary_type = _build_promql_for_metric(name, meta_type)
        label = name
        if help_text:
            label = f"{name} ({help_text[:60].strip()}{'...' if len(help_text) > 60 else ''})"
        scored.append((score, label, summary_type, promql))
    scored.sort(key=lambda x: (-x[0], x[1]))
    # Dedupe by metric name (keep highest score)
    seen_names: set[str] = set()
    unique: list[tuple[str, str, str]] = []
    for _s, label, stype, promql in scored:
        base_name = label.split(" (")[0].strip()
        if base_name in seen_names:
            continue
        seen_names.add(base_name)
        unique.append((label, stype, promql))
        if len(unique) >= max_metrics:
            break
    logger.info(
        "RCA discovery: user_question=%s discovered=%d",
        (user_question or "")[:80],
        len(unique),
    )
    return unique


def _normalize_metric_config(item: Any) -> tuple[str, str, str] | None:
    """Convert a dict from site setting to (label, summary_type, promql). Returns None if invalid."""
    if not isinstance(item, dict):
        return None
    label = str((item.get("label") or item.get("name")) or "").strip()
    promql = str((item.get("promql") or item.get("expr")) or "").strip()
    if not label or not promql:
        return None
    summary_type = str((item.get("summary_type") or item.get("type") or "gauge")).strip().lower()
    if summary_type not in ("duration", "cpu", "gauge_per_instance", "gauge"):
        summary_type = "gauge"
    return (label, summary_type, promql)


def get_effective_metric_configs() -> list[tuple[str, str, str]]:
    """
    Return the metric catalog: custom from site setting if set, otherwise built-in.
    Enables open, flexible metrics without code changes.
    """
    raw = getattr(SiteSetting, "prometheus_metric_configs", None) or []
    if not isinstance(raw, list) or len(raw) == 0:
        return list(PROMETHEUS_METRIC_CONFIGS)
    out: list[tuple[str, str, str]] = []
    for item in raw:
        cfg = _normalize_metric_config(item)
        if cfg:
            out.append(cfg)
    return out if out else list(PROMETHEUS_METRIC_CONFIGS)


def infer_prometheus_metrics_from_user_question(
    user_question: str | None,
    configs: list[tuple[str, str, str]] | None = None,
) -> list[tuple[str, str, str]] | None:
    """
    Infer which Prometheus metrics to run from the user's question.
    Returns list of (label, summary_type, promql_template), or None to use full catalog.
    - If the user mentions specific topics (duration, cpu, memory, qps, connections),
      only matching metrics are returned (by label or summary_type).
    - If the user asks generically for "metrics"/"prometheus"/"monitoring"/"analyze"
      without specific keywords, returns None so the caller runs the full set.
    configs: metric catalog to filter; if None, uses get_effective_metric_configs().
    """
    catalog = configs if configs is not None else get_effective_metric_configs()
    if not user_question or not isinstance(user_question, str):
        return None
    q = user_question.strip().lower()
    if not q:
        return None

    wants_duration = any(
        re.search(p, q)
        for p in [r"\bduration\b", r"\blatency\b", r"\bp99\b", r"\bp95\b", r"\bpercentile\b"]
    )
    wants_cpu = bool(re.search(r"\bcpu\b", q))
    wants_memory = any(
        re.search(p, q) for p in [r"\bmemory\b", r"\bram\b", r"\bheap\b"]
    )
    wants_qps = any(
        re.search(p, q)
        for p in [r"\bqps\b", r"\bthroughput\b", r"\bqueries?\s*per\s*sec", r"\btps\b"]
    )
    wants_connection = any(
        re.search(p, q) for p in [r"\bconnection\b", r"\bconn\b", r"\bconnect\b"]
    )

    def matches(label: str, summary_type: str) -> bool:
        lbl = label.lower()
        if wants_duration and ("duration" in lbl or "latency" in lbl or summary_type == "duration"):
            return True
        if wants_cpu and ("cpu" in lbl or summary_type == "cpu"):
            return True
        if wants_memory and ("memory" in lbl or "ram" in lbl or "heap" in lbl):
            return True
        if wants_qps and ("qps" in lbl or "throughput" in lbl):
            return True
        if wants_connection and ("connection" in lbl or "conn" in lbl):
            return True
        return False

    selected: list[tuple[str, str, str]] = []
    for label, summary_type, promql in catalog:
        if matches(label, summary_type):
            selected.append((label, summary_type, promql))

    if selected:
        return selected
    # Generic request: return None so caller runs full catalog.
    if re.search(r"\bprometheus\b|\bmetrics?\b|\bmonitoring\b|\banaly(?:s|z)e\b", q):
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
    effective_configs = get_effective_metric_configs()
    metric_configs: list[tuple[str, str, str]] = []

    raw_custom = getattr(SiteSetting, "prometheus_metric_configs", None) or []
    use_rca_discovery = (
        bool(getattr(SiteSetting, "prometheus_rca_discovery", True))
        and (user_question or "").strip()
        and (not isinstance(raw_custom, list) or len(raw_custom) == 0)
    )
    if use_rca_discovery:
        discovered = discover_metrics_for_rca(
            base_url,
            bearer_token,
            user_question,
            vars_map,
            logger,
            max_metrics=25,
        )
        if discovered:
            metric_configs = discovered
    if not metric_configs:
        metric_configs = infer_prometheus_metrics_from_user_question(user_question, effective_configs)
        if not metric_configs:
            metric_configs = list(effective_configs)
    logger.info(
        "Prometheus metrics to run (user_question=%s) metric_labels=%s",
        user_question or "",
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
            values = extract_series_values(series)
            if not values:
                metrics.append(f"{label}:\n- No data points.")
            else:
                avg_ms = (sum(values) / len(values)) * 1000.0
                max_ms = max(values) * 1000.0
                metrics.append(f"{label}:\n- avg: {avg_ms:.2f} ms\n- max: {max_ms:.2f} ms")
        elif summary_type == "cpu":
            text = summarize_cpu_series(series, None, logger)
            metrics.append(f"{label}:\n{text}")
        elif summary_type == "gauge_per_instance":
            text = summarize_panel_series(series, per_instance=True)
            metrics.append(f"{label}:\n{text}")
        else:
            text = summarize_panel_series(series, per_instance=False)
            metrics.append(f"{label}:\n{text}")

    return "Prometheus TiDB metrics:\n\n" + "\n\n".join(metrics)


def build_rca_summary_from_metrics(
    metrics_text: str,
    user_question: str | None = None,
) -> str:
    """
    Produce a short RCA (Root Cause Analysis) summary from the Prometheus metrics text.
    Parses the metrics output for elevated latency, high CPU, failures, and missing data.
    Used so the assistant (or CLI) can provide recommendations/RCA from fetched metrics.
    """
    if not metrics_text or not isinstance(metrics_text, str):
        return "RCA (from Prometheus):\n- No metrics available."
    lines = metrics_text.split("\n")
    bullets: list[str] = []
    current_label: str | None = None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.endswith(":") and "\n" not in s:
            current_label = s.rstrip(":").strip()
            continue
        if current_label and s.startswith("- "):
            content = s[2:].strip()
            # Duration: max X ms
            max_ms = re.search(r"max:\s*([\d.]+)\s*ms", content, re.IGNORECASE)
            if max_ms:
                try:
                    val = float(max_ms.group(1))
                    if val >= 1000:
                        bullets.append(f"Elevated latency in {current_label}: max {val:.0f} ms (consider indexing or plan review).")
                    elif val >= 500:
                        bullets.append(f"Moderate latency in {current_label}: max {val:.0f} ms.")
                except ValueError:
                    pass
            # CPU table: avg_pct | max_pct
            pct = re.search(r"(\d+(?:\.\d+)?)\s*%", content, re.IGNORECASE)
            if pct and ("avg_pct" in content or "max_pct" in content or "|" in content):
                try:
                    val = float(pct.group(1))
                    if val >= 80 and current_label and "cpu" in current_label.lower():
                        bullets.append(f"High CPU in {current_label}: {val:.1f}% (check load and resource limits).")
                except ValueError:
                    pass
            # Query failed
            if "query failed" in content.lower() or "failed:" in content.lower():
                bullets.append(f"Metric '{current_label}' failed to query; check Prometheus target and scrape.")
            # No data
            if "no data" in content.lower() or "no series" in content.lower():
                bullets.append(f"No data for '{current_label}' in the time window; verify metric exists and labels.")
    if not bullets:
        bullets.append("No obvious anomalies detected in the metrics; review the metrics above for trends.")
    return "RCA (from Prometheus metrics):\n\n" + "\n".join(f"- {b}" for b in bullets)
