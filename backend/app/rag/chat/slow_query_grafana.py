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


def _extract_series_values_from_entry(entry: dict) -> list[float]:
    values = []
    if not isinstance(entry, dict):
        return values
    data = entry.get("data") or entry
    result_items = None
    if isinstance(data, dict):
        if isinstance(data.get("result"), list):
            result_items = data.get("result")
        elif isinstance(data.get("data"), dict) and isinstance(data["data"].get("result"), list):
            result_items = data["data"].get("result")
    if not isinstance(result_items, list):
        return values
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


def _extract_entry_values_by_label(entry: dict, label_key: str) -> dict[str, list[float]]:
    values_by_label: dict[str, list[float]] = {}
    if not isinstance(entry, dict):
        return values_by_label
    data = entry.get("data") or entry
    result_items = None
    if isinstance(data, dict):
        if isinstance(data.get("result"), list):
            result_items = data.get("result")
        elif isinstance(data.get("data"), dict) and isinstance(data["data"].get("result"), list):
            result_items = data["data"].get("result")
    if not isinstance(result_items, list):
        return values_by_label
    for item in result_items:
        if not isinstance(item, dict):
            continue
        metric = item.get("metric") if isinstance(item.get("metric"), dict) else {}
        label = str(metric.get(label_key) or "").strip() if isinstance(metric, dict) else ""
        if not label:
            label = "unknown"
        bucket = values_by_label.setdefault(label, [])
        vals = item.get("values")
        if isinstance(vals, list):
            for v in vals:
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    try:
                        bucket.append(float(v[1]))
                    except Exception:
                        continue
        else:
            single = item.get("value")
            if isinstance(single, (list, tuple)) and len(single) >= 2:
                try:
                    bucket.append(float(single[1]))
                except Exception:
                    continue
    return values_by_label


def _extract_entry_values_by_best_label(
    entry: dict,
    candidate_keys: list[str],
    logger: logging.Logger,
    context: str,
) -> tuple[str | None, dict[str, list[float]]]:
    best_key = None
    best_values: dict[str, list[float]] = {}
    for key in candidate_keys:
        values = _extract_entry_values_by_label(entry, key)
        if len(values) > len(best_values):
            best_values = values
            best_key = key
    logger.info(
        "%s metrics: best_label_key=%s distinct=%s",
        context,
        best_key,
        len(best_values),
    )
    return best_key, best_values


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


def _extract_series_values_by_label(series: list, label_key: str) -> dict[str, list[float]]:
    values_by_label: dict[str, list[float]] = {}
    for s in series:
        if not isinstance(s, dict):
            continue
        data = s.get("data") or s
        result_items = None
        if isinstance(data, dict):
            if isinstance(data.get("result"), list):
                result_items = data.get("result")
            elif isinstance(data.get("data"), dict) and isinstance(data["data"].get("result"), list):
                result_items = data["data"].get("result")
        if not isinstance(result_items, list):
            continue
        for item in result_items:
            if not isinstance(item, dict):
                continue
            metric = item.get("metric") if isinstance(item.get("metric"), dict) else {}
            label = str(metric.get(label_key) or "").strip() if isinstance(metric, dict) else ""
            if not label:
                label = "unknown"
            bucket = values_by_label.setdefault(label, [])
            vals = item.get("values")
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        try:
                            bucket.append(float(v[1]))
                        except Exception:
                            continue
            else:
                single = item.get("value")
                if isinstance(single, (list, tuple)) and len(single) >= 2:
                    try:
                        bucket.append(float(single[1]))
                    except Exception:
                        continue
    return values_by_label


def _summarize_duration_series(series: list, targets: list | None) -> str:
    if not series:
        return "- No data points found."
    if not isinstance(targets, list) or not targets:
        values = _extract_series_values(series)
        if not values:
            return "- No data points found."
        avg = (sum(values) / len(values)) * 1000.0
        max_v = max(values) * 1000.0
        return f"- avg: {avg:.2f} ms\n- max: {max_v:.2f} ms"

    quantile_re = re.compile(r"histogram_quantile\(\s*([0-9.]+)")
    lines = []
    for idx, target in enumerate(targets):
        if idx >= len(series):
            break
        expr = str((target or {}).get("expr") or (target or {}).get("query") or "")
        match = quantile_re.search(expr)
        if not match:
            continue
        q = match.group(1)
        label = q
        if q == "0.95":
            label = "P95"
        elif q == "0.99":
            label = "P99"
        elif q == "0.999":
            label = "P999"
        values = _extract_series_values_from_entry(series[idx])
        if not values:
            lines.append(f"- {label}: no data")
            continue
        avg = (sum(values) / len(values)) * 1000.0
        max_v = max(values) * 1000.0
        lines.append(f"- {label}: avg {avg:.2f} ms, max {max_v:.2f} ms")
    if lines:
        return "\n".join(lines)
    values = _extract_series_values(series)
    if not values:
        return "- No data points found."
    avg = (sum(values) / len(values)) * 1000.0
    max_v = max(values) * 1000.0
    return f"- avg: {avg:.2f} ms\n- max: {max_v:.2f} ms"


def _summarize_cpu_series(series: list, targets: list | None, logger: logging.Logger) -> str:
    if not series:
        return "- No data points found."
    if not isinstance(targets, list) or not targets:
        logger.info("CPU metrics: targets missing; falling back to per-instance raw stats")
        return _summarize_panel_series(series, per_instance=True)

    quota_idx = None
    actual_idx = None
    for idx, target in enumerate(targets):
        if not isinstance(target, dict):
            continue
        legend = str(target.get("legendFormat") or target.get("legend") or "").strip().lower()
        expr = str(target.get("expr") or target.get("query") or "").strip().lower()
        logger.info("CPU metrics: target[%s] legend=%s expr=%s", idx, legend, expr)
        if (
            "quota-" in legend
            and ("{{instance}}" in legend or "${instance}" in legend or "$instance" in legend)
        ) or ("quota" in legend) or ("maxprocs" in expr):
            quota_idx = idx
        elif actual_idx is None:
            actual_idx = idx
    logger.info(
        "CPU metrics: targets=%s actual_idx=%s quota_idx=%s",
        len(targets),
        actual_idx,
        quota_idx,
    )

    if actual_idx is None:
        actual_idx = 0
    if actual_idx >= len(series):
        logger.info(
            "CPU metrics: actual_idx out of range actual_idx=%s series_len=%s",
            actual_idx,
            len(series),
        )
        return "- No data points found."

    _, actual_by_instance = _extract_entry_values_by_best_label(
        series[actual_idx],
        ["instance", "tidb_instance", "pod", "pod_name", "instance_addr"],
        logger,
        "CPU actual",
    )
    logger.info("CPU metrics: actual_instances=%s", len(actual_by_instance))

    quota_by_instance: dict[str, list[float]] = {}
    if quota_idx is not None and quota_idx < len(series):
        _, quota_by_instance = _extract_entry_values_by_best_label(
            series[quota_idx],
            ["instance", "tidb_instance", "pod", "pod_name", "instance_addr"],
            logger,
            "CPU quota",
        )
    logger.info("CPU metrics: quota_instances=%s", len(quota_by_instance))

    if not actual_by_instance:
        return "- No data points found."

    lines = []
    table_rows = []
    all_pcts: list[float] = []
    for instance, values in sorted(actual_by_instance.items()):
        if not values:
            table_rows.append((instance, "-", "-", "-", "-"))
            continue
        avg = sum(values) / len(values)
        max_v = max(values)
        quota_vals = quota_by_instance.get(instance, [])
        if quota_vals:
            quota_max = max(quota_vals)
            if quota_max > 0:
                pct_values = [(v / quota_max * 100.0) for v in values]
                avg_pct = sum(pct_values) / len(pct_values)
                max_pct = max(pct_values)
                logger.info(
                    "CPU metrics: instance=%s avg=%s max=%s quota_max=%s avg_pct=%.2f max_pct=%.2f",
                    instance,
                    avg,
                    max_v,
                    quota_max,
                    avg_pct,
                    max_pct,
                )
                all_pcts.extend(pct_values)
                table_rows.append(
                    (instance, f"{avg_pct:.2f}%", f"{max_pct:.2f}%", f"{avg:.6f}", f"{quota_max:.2f}", pct_values)
                )
                continue
        logger.info(
            "CPU metrics: instance=%s avg=%s max=%s quota_missing_or_zero=%s",
            instance,
            avg,
            max_v,
            not quota_vals or (max(quota_vals) if quota_vals else 0) <= 0,
        )
        table_rows.append((instance, "-", "-", f"{avg:.6f}", "-", []))

    if not table_rows:
        return "- No data points found."
    if all_pcts:
        overall_avg = sum(all_pcts) / len(all_pcts)
        threshold = overall_avg + 5.0
        filtered = []
        for r in table_rows:
            if r[1] == "-":
                continue
            pct_values = r[5] if len(r) > 5 else []
            if pct_values:
                rolling_avgs = [
                    (pct_values[i] + pct_values[i + 1]) / 2.0
                    for i in range(len(pct_values) - 1)
                ]
                if rolling_avgs and max(rolling_avgs) > threshold:
                    filtered.append(r)
        table_rows = filtered
        if not table_rows:
            return "- No instances above avg + 5%."
    header = "| instance | avg_pct | max_pct |"
    sep = "|---|---:|---:|"
    body = "\n".join(f"| {r[0]} | {r[1]} | {r[2]} |" for r in table_rows)
    return "\n".join([header, sep, body])


def _summarize_panel_series(series: list, *, per_instance: bool = False) -> str:
    if per_instance:
        values_by_instance = _extract_series_values_by_label(series, "instance")
        if not values_by_instance:
            values_by_instance = _extract_series_values_by_label(series, "tidb_instance")
        if not values_by_instance:
            values = _extract_series_values(series)
            if not values:
                return "- No data points found."
            avg = sum(values) / len(values)
            max_v = max(values)
            return f"- avg: {avg:.6f}\n- max: {max_v:.6f}"
        lines = []
        for instance, values in sorted(values_by_instance.items()):
            if not values:
                lines.append(f"- {instance}: no data")
                continue
            avg = sum(values) / len(values)
            max_v = max(values)
            lines.append(f"- {instance}: avg {avg:.6f}, max {max_v:.6f}")
        return "\n".join(lines)

    values = _extract_series_values(series)
    if not values:
        return "- No data points found."
    avg = sum(values) / len(values)
    max_v = max(values)
    return f"- avg: {avg:.6f}\n- max: {max_v:.6f}"


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
    tikv_cpu_panel = _pick_panel_entry(panels, dashboard_title_pattern=r"\btikv\b", panel_title_pattern=r"\bcpu\b")

    vars_map = _build_grafana_vars(cluster_hint)
    metrics = []
    for label, panel in [
        ("Duration (TiDB)", summary_panel),
        ("CPU (TiDB)", cpu_panel),
        ("CPU (TiKV)", tikv_cpu_panel),
    ]:
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
        if isinstance(panel, dict) and isinstance(panel.get("targets"), list):
            params["targets"] = panel.get("targets")
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
        per_instance = label.lower().startswith("cpu")
        if label.lower().startswith("duration"):
            targets = panel.get("targets") if isinstance(panel, dict) else None
            metrics.append(f"{label} (panel: {panel_title}):\n{_summarize_duration_series(series, targets)}")
        elif label.lower().startswith("cpu"):
            targets = panel.get("targets") if isinstance(panel, dict) else None
            metrics.append(
                f"{label} (panel: {panel_title}):\n{_summarize_cpu_series(series, targets, logger)}"
            )
        else:
            metrics.append(
                f"{label} (panel: {panel_title}):\n{_summarize_panel_series(series, per_instance=per_instance)}"
            )

    return "Grafana TiDB metrics:\n\n" + "\n\n".join(metrics)
