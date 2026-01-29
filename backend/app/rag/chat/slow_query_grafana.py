import logging
from datetime import datetime, UTC

from app.mcp.client import run_mcp_tool, run_mcp_tool_url
from app.site_settings import SiteSetting


def build_grafana_duration_analysis(
    start_time: str,
    end_time: str,
    grafana_host: str | None,
    logger: logging.Logger,
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

    tool = "grafana_panel_query_range"
    params = {
        "uid": dashboard_uid,
        "panel_id": panel_id,
        "panel_title": "Duration",
        "from": start_ms,
        "to": end_ms,
        "intervalMs": 60000,
        "vars": getattr(SiteSetting, "mcp_grafana_vars", None) or {},
    }
    try:
        mcp_ws_url = str((grafana_entry or {}).get("mcp_ws_url", "")).strip()
        if mcp_ws_url:
            result = run_mcp_tool_url(mcp_ws_url, tool, params)
        else:
            # Prefer managed Grafana runner when ws URL is not set.
            try:
                from app.mcp.managed import run_managed_mcp_grafana_tool  # local import
                result = run_managed_mcp_grafana_tool(grafana_name, tool, params)
            except Exception:
                result = run_mcp_tool(tool, params, host_name=grafana_name)
    except Exception as e:
        logger.exception(
            "Grafana MCP query failed: host=%s tool=%s params=%s",
            grafana_name,
            tool,
            {"uid": dashboard_uid, "panel_id": panel_id},
        )
        return f"Grafana Duration analysis:\n\n- Grafana panel query failed: {e}"

    series = result.get("series") if isinstance(result, dict) else None
    if not isinstance(series, list) or not series:
        return "Grafana Duration analysis:\n\n- No series data returned."

    values = []
    for s in series:
        if isinstance(s, dict):
            data = s.get("data") or s
            result_items = data.get("data", {}).get("result") if isinstance(data, dict) else None
            if isinstance(result_items, list):
                for item in result_items:
                    vals = item.get("values") if isinstance(item, dict) else None
                    if isinstance(vals, list):
                        for v in vals:
                            if isinstance(v, (list, tuple)) and len(v) >= 2:
                                try:
                                    values.append(float(v[1]))
                                except Exception:
                                    continue
    if not values:
        return "Grafana Duration analysis:\n\n- No data points found."
    avg = sum(values) / len(values)
    max_v = max(values)
    return (
        "Grafana Duration analysis:\n\n"
        f"- avg: {avg:.6f}\n"
        f"- max: {max_v:.6f}\n"
        f"- points: {len(values)}"
    )
