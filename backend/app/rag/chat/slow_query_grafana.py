import logging
import re
from datetime import UTC, datetime

from app.mcp.client import run_mcp_tool, run_mcp_tool_url
from app.site_settings import SiteSetting


def _run_grafana_tool(
    grafana_entry: dict,
    grafana_name: str | None,
    tool: str,
    params: dict,
) -> dict | list | str | None:
    mcp_ws_url = str((grafana_entry or {}).get("mcp_ws_url", "")).strip()
    if mcp_ws_url:
        return run_mcp_tool_url(mcp_ws_url, tool, params)
    try:
        from app.mcp.managed import run_managed_mcp_grafana_tool  # local import
        return run_managed_mcp_grafana_tool(grafana_name, tool, params)
    except Exception:
        return run_mcp_tool(tool, params, host_name=grafana_name)


def _extract_series_values(series: list) -> list[float]:
    values = []
    for s in series:
        if isinstance(s, dict):
            data = s.get("data") or s
            result_items = None
            if isinstance(data, dict):
                if isinstance(data.get("result"), list):
                    result_items = data.get("result")
                elif isinstance(data.get("data"), dict) and isinstance(data["data"].get("result"), list):
                    result_items = data["data"].get("result")
            if isinstance(result_items, list):
                for item in result_items:
                    if not isinstance(item, dict):
                        continue
                    vals = item.get("values")
                    if isinstance(vals, list):
                        for v in vals:
                            if isinstance(v, (list, tuple)) and len(v) >= 2:
                                try:
                                    values.append(float(v[1]))
                                except Exception:
                                    continue
                    else:
                        single = item.get("value")
                        if isinstance(single, (list, tuple)) and len(single) >= 2:
                            try:
                                values.append(float(single[1]))
                            except Exception:
                                continue
    return values


def _build_grafana_vars(cluster_hint: str | None) -> dict:
    vars_map = dict(getattr(SiteSetting, "mcp_grafana_vars", None) or {})
    if not cluster_hint:
        return vars_map
    lowered = cluster_hint.strip()
    if vars_map:
        for key in list(vars_map.keys()):
            if "cluster" in str(key).lower():
                vars_map[key] = lowered
    else:
        vars_map = {
            "cluster": lowered,
            "tidb_cluster": lowered,
            "k8s_cluster": lowered,
        }
    return vars_map


def _summarize_panel_series(series: list) -> str:
    values = _extract_series_values(series)
    if not values:
        return "- No data points found."
    avg = sum(values) / len(values)
    max_v = max(values)
    return f"- avg: {avg:.6f}\n- max: {max_v:.6f}\n- points: {len(values)}"


def _pick_panel_entry(
    panels: list[dict],
    *,
    dashboard_title_pattern: str,
    panel_title_pattern: str,
) -> dict | None:
    dashboard_re = re.compile(dashboard_title_pattern, re.IGNORECASE)
    panel_re = re.compile(panel_title_pattern, re.IGNORECASE)
    for p in panels:
        if not isinstance(p, dict):
            continue
        if not dashboard_re.search(str(p.get("dashboard_title") or "")):
            continue
        if panel_re.search(str(p.get("panel_title") or "")):
            return p
    return None


def build_grafana_duration_analysis(
    start_time: str,
    end_time: str,
    grafana_host: str | None,
    logger: logging.Logger,
    cluster_hint: str | None = None,
) -> str:
    SiteSetting.update_db_cache()
    grafana_hosts = getattr(SiteSetting, "mcp_grafana_hosts", None) or []
    grafana_name = grafana_host
    grafana_entry = None
    if grafana_name:
        for it in grafana_hosts:
            if str((it or {}).get("name", "")).strip().lower() == grafana_name.strip().lower():
                grafana_entry = it
                break
    if not grafana_entry and grafana_hosts:
        grafana_entry = grafana_hosts[0]
        grafana_name = str((grafana_entry or {}).get("name", "")).strip() or None
    if not grafana_entry:
        return "Grafana panels:\n\n- Grafana MCP host not configured."

    try:
        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
    except Exception:
        return "Grafana Duration analysis:\n\n- Invalid time window; expected 'YYYY-MM-DD HH:MM:SS' UTC."

    dashboard_uid = str(getattr(SiteSetting, "mcp_grafana_dashboard_uid", "") or "").strip()
    if not dashboard_uid:
        return "Grafana Duration analysis:\n\n- Grafana dashboard UID not configured."
    panel_id = int(getattr(SiteSetting, "mcp_grafana_duration_panel_id", 0) or 0)
    if not panel_id:
        return "Grafana Duration analysis:\n\n- Grafana Duration panel id not configured."

    vars_map = _build_grafana_vars(cluster_hint)
    tool = "grafana_panel_query_range"
    params = {
        "uid": dashboard_uid,
        "panel_id": panel_id,
        "panel_title": "Duration",
        "from": start_ms,
        "to": end_ms,
        "intervalMs": 60000,
        "vars": vars_map,
    }
    try:
        result = _run_grafana_tool(grafana_entry, grafana_name, tool, params)
    except Exception as e:
        logger.exception(
            "Grafana MCP query failed: host=%s tool=%s params=%s",
            grafana_name,
            tool,
            {"uid": dashboard_uid, "panel_id": panel_id},
        )
        return f"Grafana Duration analysis:\n\n- Grafana panel query failed: {e}"

    if isinstance(result, dict):
        logger.info(
            "Grafana Duration panel meta host=%s uid=%s panel_id=%s title=%s",
            grafana_name,
            dashboard_uid,
            panel_id,
            result.get("panel", {}).get("title", ""),
        )
        logger.info("Grafana Duration vars=%s", params.get("vars"))
        if isinstance(result.get("series"), list):
            logger.info("Grafana Duration series_count=%s", len(result.get("series", [])))
            for s in result.get("series", []):
                try:
                    if isinstance(s, dict):
                        data = s.get("data") or s
                        if isinstance(data, dict) and isinstance(data.get("data"), dict):
                            # Prometheus responses include the query under data.data.result[*].metric / metadata
                            logger.info(
                                "Grafana Duration query result keys=%s",
                                ",".join(sorted(data.get("data", {}).keys())),
                            )
                except Exception:
                    continue
    series = result.get("series") if isinstance(result, dict) else None
    if not isinstance(series, list) or not series:
        return "Grafana Duration analysis:\n\n- No series data returned."
    return "Grafana Duration analysis:\n\n" + _summarize_panel_series(series)


def build_grafana_tidb_metrics_analysis(
    start_time: str,
    end_time: str,
    grafana_host: str | None,
    logger: logging.Logger,
    cluster_hint: str | None = None,
) -> str:
    SiteSetting.update_db_cache()
    grafana_hosts = getattr(SiteSetting, "mcp_grafana_hosts", None) or []
    grafana_name = grafana_host
    grafana_entry = None
    if grafana_name:
        for it in grafana_hosts:
            if str((it or {}).get("name", "")).strip().lower() == grafana_name.strip().lower():
                grafana_entry = it
                break
    if not grafana_entry and grafana_hosts:
        grafana_entry = grafana_hosts[0]
        grafana_name = str((grafana_entry or {}).get("name", "")).strip() or None
    if not grafana_entry:
        return "Grafana TiDB metrics:\n\n- Grafana MCP host not configured."

    try:
        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
    except Exception:
        return "Grafana TiDB metrics:\n\n- Invalid time window; expected 'YYYY-MM-DD HH:MM:SS' UTC."

    panels = getattr(SiteSetting, "mcp_grafana_panels", None) or []
    if grafana_name:
        panels = [p for p in panels if str((p or {}).get("host", "")).strip().lower() == grafana_name.strip().lower()]
    if not panels:
        return "Grafana TiDB metrics:\n\n- Grafana panels not synced; sync dashboards and panels first."

    latency_panel = _pick_panel_entry(panels, dashboard_title_pattern=r"\btidb\b", panel_title_pattern=r"latenc")
    cpu_panel = _pick_panel_entry(panels, dashboard_title_pattern=r"\btidb\b", panel_title_pattern=r"\bcpu\b")

    vars_map = _build_grafana_vars(cluster_hint)
    metrics = []
    for label, panel in [("Latency", latency_panel), ("CPU", cpu_panel)]:
        if not panel:
            metrics.append(f"{label}:\n- No matching panel found.")
            continue
        uid = str(panel.get("dashboard_uid") or "")
        panel_id = panel.get("panel_id")
        panel_title = str(panel.get("panel_title") or "")
        if not uid or not panel_id:
            metrics.append(f"{label}:\n- Panel metadata incomplete.")
            continue
        tool = "grafana_panel_query_range"
        params = {
            "uid": uid,
            "panel_id": panel_id,
            "panel_title": panel_title,
            "from": start_ms,
            "to": end_ms,
            "intervalMs": 60000,
            "vars": vars_map,
        }
        try:
            result = _run_grafana_tool(grafana_entry, grafana_name, tool, params)
        except Exception as e:
            logger.exception("Grafana MCP query failed: host=%s tool=%s panel=%s", grafana_name, tool, panel_title)
            metrics.append(f"{label}:\n- Grafana panel query failed: {e}")
            continue
        series = result.get("series") if isinstance(result, dict) else None
        if not isinstance(series, list) or not series:
            metrics.append(f"{label}:\n- No series data returned.")
            continue
        metrics.append(f"{label} (panel: {panel_title}):\n{_summarize_panel_series(series)}")

    return "Grafana TiDB metrics:\n\n" + "\n\n".join(metrics)
