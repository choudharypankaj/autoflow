import logging
from datetime import datetime, UTC

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


def _build_grafana_storage_capacity(
    grafana_entry: dict,
    grafana_name: str | None,
    logger: logging.Logger,
    start_ms: int,
    end_ms: int,
) -> str:
    dashboard_uid = str(getattr(SiteSetting, "mcp_grafana_panels_uid", "") or "").strip()
    panel_id = int(getattr(SiteSetting, "mcp_grafana_storage_panel_id", 0) or 0)
    if not dashboard_uid or not panel_id:
        return ""
    params = {
        "uid": dashboard_uid,
        "panel_id": panel_id,
        "panel_title": "Storage capacity",
        "from": start_ms,
        "to": end_ms,
        "intervalMs": 60000,
        "vars": getattr(SiteSetting, "mcp_grafana_vars", None) or {},
    }
    try:
        result = _run_grafana_tool(grafana_entry, grafana_name, "grafana_panel_query_range", params)
    except Exception as e:
        logger.exception("Grafana storage capacity query failed: %s", e)
        return f"Grafana Storage capacity:\n\n- Grafana panel query failed: {e}"
    series = result.get("series") if isinstance(result, dict) else None
    if not isinstance(series, list) or not series:
        return "Grafana Storage capacity:\n\n- No series data returned."
    values = _extract_series_values(series)
    if not values:
        return "Grafana Storage capacity:\n\n- No data points found."
    avg = sum(values) / len(values)
    max_v = max(values)
    return (
        "Grafana Storage capacity:\n\n"
        f"- avg: {avg:.6f}\n"
        f"- max: {max_v:.6f}\n"
        f"- points: {len(values)}"
    )

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

    values = _extract_series_values(series)
    if not values:
        storage_text = _build_grafana_storage_capacity(
            grafana_entry,
            grafana_name,
            logger,
            start_ms,
            end_ms,
        )
        if storage_text:
            return "Grafana Duration analysis:\n\n- No data points found.\n\n" + storage_text
        return "Grafana Duration analysis:\n\n- No data points found."
    avg = sum(values) / len(values)
    max_v = max(values)
    storage_text = _build_grafana_storage_capacity(
        grafana_entry,
        grafana_name,
        logger,
        start_ms,
        end_ms,
    )
    text = (
        "Grafana Duration analysis:\n\n"
        f"- avg: {avg:.6f}\n"
        f"- max: {max_v:.6f}\n"
        f"- points: {len(values)}"
    )
    if storage_text:
        text = f"{text}\n\n{storage_text}"
    return text
