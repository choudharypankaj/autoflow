import logging
import re
from datetime import UTC, datetime
from typing import Any

import requests

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


def build_grafana_tidb_metrics_analysis(
    start_time: str,
    end_time: str,
    grafana_host: str | None,
    logger: logging.Logger,
    cluster_hint: str | None = None,
    session: Any | None = None,
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

    def _coerce_panel_datasource(value: object) -> dict:
        if isinstance(value, dict):
            return {
                "datasource_uid": str(value.get("uid") or ""),
                "datasource_name": str(value.get("name") or ""),
                "datasource_type": str(value.get("type") or ""),
            }
        if isinstance(value, str):
            return {
                "datasource_uid": "",
                "datasource_name": value,
                "datasource_type": "",
            }
        return {"datasource_uid": "", "datasource_name": "", "datasource_type": ""}

    def _coerce_target_datasource(value: object) -> dict:
        if isinstance(value, dict):
            return {
                "datasource_uid": str(value.get("uid") or ""),
                "datasource_name": str(value.get("name") or ""),
                "datasource_type": str(value.get("type") or ""),
            }
        if isinstance(value, str):
            return {"datasource_uid": "", "datasource_name": value, "datasource_type": ""}
        return {"datasource_uid": "", "datasource_name": "", "datasource_type": ""}

    def _simplify_targets(items: list) -> list[dict]:
        simplified = []
        for target in items or []:
            if not isinstance(target, dict):
                continue
            target_ds = _coerce_target_datasource(target.get("datasource"))
            simplified.append(
                {
                    "ref_id": target.get("refId", ""),
                    "expr": target.get("expr", ""),
                    "query": target.get("query", ""),
                    "legend": target.get("legendFormat", ""),
                    "format": target.get("format", ""),
                    "interval": target.get("interval", ""),
                    "datasource_uid": target_ds.get("datasource_uid", ""),
                    "datasource_name": target_ds.get("datasource_name", ""),
                    "datasource_type": target_ds.get("datasource_type", ""),
                }
            )
        return simplified

    def _iter_panels(items: list) -> list[dict]:
        rows = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            rows.append(item)
            if item.get("type") == "row":
                rows.extend([p for p in item.get("panels") or [] if isinstance(p, dict)])
        return rows

    def _fetch_dashboards(entry: dict, name: str | None) -> list[dict]:
        base = str((entry or {}).get("grafana_url", "")).strip().rstrip("/")
        api_key = str((entry or {}).get("grafana_api_key", "")).strip()
        if not base or not api_key:
            return []
        try:
            resp = requests.get(
                base + "/api/search",
                headers={"Authorization": f"Bearer {api_key}"},
                params={"type": "dash-db"},
                timeout=10,
            )
        except Exception as e:
            logger.exception("Grafana list dashboards failed: host=%s error=%s", name, e)
            return []
        if resp.status_code >= 400:
            logger.info("Grafana list dashboards failed: host=%s status=%s", name, resp.status_code)
            return []
        items = resp.json() or []
        dashboards = []
        for it in items:
            if not isinstance(it, dict):
                continue
            dashboards.append(
                {
                    "host": name or "",
                    "uid": str(it.get("uid") or ""),
                    "title": str(it.get("title") or ""),
                    "folder": str(it.get("folderTitle") or ""),
                    "uri": str(it.get("uri") or ""),
                    "url": str(it.get("url") or ""),
                }
            )
        return dashboards

    def _fetch_panels(entry: dict, name: str | None, dashboards: list[dict]) -> list[dict]:
        base = str((entry or {}).get("grafana_url", "")).strip().rstrip("/")
        api_key = str((entry or {}).get("grafana_api_key", "")).strip()
        if not base or not api_key:
            return []
        panels = []
        for d in dashboards:
            uid = str((d or {}).get("uid") or "")
            title = str((d or {}).get("title") or "")
            if not uid:
                continue
            try:
                resp = requests.get(
                    base + f"/api/dashboards/uid/{uid}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10,
                )
            except Exception as e:
                logger.exception("Grafana fetch dashboard failed: host=%s uid=%s error=%s", name, uid, e)
                continue
            if resp.status_code >= 400:
                logger.info("Grafana fetch dashboard failed: host=%s uid=%s status=%s", name, uid, resp.status_code)
                continue
            payload = resp.json() or {}
            dashboard = payload.get("dashboard") or {}
            items = _iter_panels(dashboard.get("panels") or [])
            dashboard_ds = _coerce_panel_datasource(dashboard.get("datasource"))
            for p in items:
                panel_ds = _coerce_panel_datasource(p.get("datasource") or dashboard.get("datasource"))
                panels.append(
                    {
                        "host": name or "",
                        "dashboard_uid": uid,
                        "dashboard_title": title,
                        "panel_id": p.get("id", ""),
                        "panel_title": p.get("title", ""),
                        "panel_type": p.get("type", ""),
                        "datasource_uid": panel_ds.get("datasource_uid", "") or dashboard_ds.get("datasource_uid", ""),
                        "datasource_name": panel_ds.get("datasource_name", "") or dashboard_ds.get("datasource_name", ""),
                        "datasource_type": panel_ds.get("datasource_type", "") or dashboard_ds.get("datasource_type", ""),
                        "targets": _simplify_targets(p.get("targets") or []),
                    }
                )
        return panels

    def _maybe_sync_panels() -> list[dict]:
        dashboards = _fetch_dashboards(grafana_entry, grafana_name)
        panels_fetched = _fetch_panels(grafana_entry, grafana_name, dashboards)
        if session is not None and grafana_name:
            try:
                SiteSetting.update_db_cache()
                existing_dash = getattr(SiteSetting, "mcp_grafana_dashboards", None) or []
                kept_dash = [
                    d for d in existing_dash
                    if str((d or {}).get("host", "")).strip().lower() != grafana_name.strip().lower()
                ]
                SiteSetting.update_setting(session, "mcp_grafana_dashboards", kept_dash + dashboards)
                existing_panels = getattr(SiteSetting, "mcp_grafana_panels", None) or []
                kept_panels = [
                    p for p in existing_panels
                    if str((p or {}).get("host", "")).strip().lower() != grafana_name.strip().lower()
                ]
                SiteSetting.update_setting(session, "mcp_grafana_panels", kept_panels + panels_fetched)
            except Exception:
                logger.exception("Grafana on-demand sync failed to persist: host=%s", grafana_name)
        return panels_fetched

    panels = getattr(SiteSetting, "mcp_grafana_panels", None) or []
    if grafana_name:
        panels = [p for p in panels if str((p or {}).get("host", "")).strip().lower() == grafana_name.strip().lower()]
    if not panels:
        panels = _maybe_sync_panels()
    if not panels:
        return "Grafana TiDB metrics:\n\n- Grafana panels not synced; sync dashboards and panels first."

    summary_panel = _pick_panel_entry(
        panels,
        dashboard_title_pattern=r"\btidb\b",
        panel_title_pattern=r"\bduration\b",
    )
    cpu_panel = _pick_panel_entry(panels, dashboard_title_pattern=r"\btidb\b", panel_title_pattern=r"\bcpu\b")

    vars_map = _build_grafana_vars(cluster_hint)
    metrics = []
    for label, panel in [("Duration (TiDB)", summary_panel), ("CPU", cpu_panel)]:
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
