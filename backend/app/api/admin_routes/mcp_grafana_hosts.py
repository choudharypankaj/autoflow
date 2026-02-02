from typing import Dict, List, Optional

import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import CurrentSuperuserDep, SessionDep
from app.site_settings import SiteSetting

router = APIRouter()


class GrafanaMCPHostItem(BaseModel):
    name: str = Field(..., description="Display name for Grafana MCP host")
    grafana_url: str = Field(..., description="Grafana base URL (http/https)")
    grafana_api_key: str = Field(..., description="Grafana service account API key")
    mcp_ws_url: Optional[str] = Field(
        default=None, description="Optional ws:// or wss:// MCP server URL"
    )


class CreateGrafanaMCPHostRequest(GrafanaMCPHostItem):
    pass


def _get_grafana_hosts() -> List[Dict[str, str]]:
    SiteSetting.update_db_cache()
    return getattr(SiteSetting, "mcp_grafana_hosts", None) or []


def _save_grafana_hosts(session: SessionDep, hosts: List[Dict[str, str]]) -> None:
    SiteSetting.update_setting(session, "mcp_grafana_hosts", hosts)


def _verify_grafana(grafana_url: str, api_key: str) -> None:
    if not (grafana_url.startswith("http://") or grafana_url.startswith("https://")):
        raise HTTPException(status_code=400, detail="grafana_url must start with http:// or https://")
    base = grafana_url.rstrip("/")
    url = base + "/api/user"
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=6,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to reach Grafana: {e}")
    if resp.status_code == 404:
        resp = requests.get(
            base + "/api/org",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=6,
        )
    if resp.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Grafana auth failed: {resp.status_code} {resp.text}")


def _sync_grafana_dashboards(
    session: SessionDep,
    *,
    host_name: str,
    grafana_url: str,
    api_key: str,
) -> List[Dict[str, str]]:
    base = grafana_url.rstrip("/")
    try:
        resp = requests.get(
            base + "/api/search",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"type": "dash-db"},
            timeout=10,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to list Grafana dashboards: {e}")
    if resp.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Grafana list dashboards failed: {resp.status_code} {resp.text}")
    items = resp.json() or []
    dashboards: List[Dict[str, str]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        dashboards.append(
            {
                "host": host_name,
                "uid": str(it.get("uid") or ""),
                "title": str(it.get("title") or ""),
                "folder": str(it.get("folderTitle") or ""),
                "uri": str(it.get("uri") or ""),
                "url": str(it.get("url") or ""),
            }
        )
    SiteSetting.update_db_cache()
    existing = getattr(SiteSetting, "mcp_grafana_dashboards", None) or []
    kept = [d for d in existing if str((d or {}).get("host", "")).strip().lower() != host_name.strip().lower()]
    SiteSetting.update_setting(session, "mcp_grafana_dashboards", kept + dashboards)
    return dashboards


def _sync_grafana_panels(
    session: SessionDep,
    *,
    host_name: str,
    grafana_url: str,
    api_key: str,
    dashboards: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    base = grafana_url.rstrip("/")
    panels: List[Dict[str, str]] = []

    def _coerce_panel_datasource(value: object) -> Dict[str, str]:
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

    def _coerce_target_datasource(value: object) -> Dict[str, str]:
        if isinstance(value, dict):
            return {
                "datasource_uid": str(value.get("uid") or ""),
                "datasource_name": str(value.get("name") or ""),
                "datasource_type": str(value.get("type") or ""),
            }
        if isinstance(value, str):
            return {"datasource_uid": "", "datasource_name": value, "datasource_type": ""}
        return {"datasource_uid": "", "datasource_name": "", "datasource_type": ""}

    def _simplify_targets(items: List[Dict]) -> List[Dict[str, object]]:
        simplified: List[Dict[str, object]] = []
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

    def _iter_panels(items: List[Dict]) -> List[Dict]:
        rows = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            rows.append(item)
            if item.get("type") == "row":
                rows.extend([p for p in item.get("panels") or [] if isinstance(p, dict)])
        return rows

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
            raise HTTPException(status_code=400, detail=f"Failed to fetch Grafana dashboard panels: {e}")
        if resp.status_code >= 400:
            raise HTTPException(status_code=400, detail=f"Grafana panels fetch failed: {resp.status_code} {resp.text}")
        payload = resp.json() or {}
        dashboard = payload.get("dashboard") or {}
        items = dashboard.get("panels") or []
        items = _iter_panels(items)
        dashboard_ds = _coerce_panel_datasource(dashboard.get("datasource"))
        for p in items:
            panel_ds = _coerce_panel_datasource(p.get("datasource") or dashboard.get("datasource"))
            panels.append(
                {
                    "host": host_name,
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

    SiteSetting.update_db_cache()
    existing = getattr(SiteSetting, "mcp_grafana_panels", None) or []
    kept = [p for p in existing if str((p or {}).get("host", "")).strip().lower() != host_name.strip().lower()]
    SiteSetting.update_setting(session, "mcp_grafana_panels", kept + panels)
    return panels


def _upsert_mcp_host(session: SessionDep, name: str, href: str) -> None:
    SiteSetting.update_db_cache()
    hosts: List[Dict[str, str]] = getattr(SiteSetting, "mcp_hosts", None) or []
    updated: List[Dict[str, str]] = []
    found = False
    for it in hosts:
        if str((it or {}).get("text", "")).strip().lower() == name.strip().lower():
            updated.append({"text": name, "href": href})
            found = True
        else:
            updated.append(it)
    if not found:
        updated.append({"text": name, "href": href})
    SiteSetting.update_setting(session, "mcp_hosts", updated)


@router.get("/admin/mcp/grafana/hosts", response_model=List[GrafanaMCPHostItem])
def list_grafana_hosts(user: CurrentSuperuserDep) -> List[GrafanaMCPHostItem]:
    items = _get_grafana_hosts()
    return [
        GrafanaMCPHostItem.model_validate(
            {
                "name": (it or {}).get("name", ""),
                "grafana_url": (it or {}).get("grafana_url", ""),
                "grafana_api_key": (it or {}).get("grafana_api_key", ""),
                "mcp_ws_url": (it or {}).get("mcp_ws_url", None),
            }
        )
        for it in items
    ]


@router.get("/admin/mcp/grafana/dashboards", response_model=List[Dict[str, str]])
def list_grafana_dashboards(
    user: CurrentSuperuserDep,
    host: Optional[str] = Query(default=None, description="Filter by Grafana host name"),
) -> List[Dict[str, str]]:
    SiteSetting.update_db_cache()
    dashboards = getattr(SiteSetting, "mcp_grafana_dashboards", None) or []
    if host:
        return [d for d in dashboards if str((d or {}).get("host", "")).strip().lower() == host.strip().lower()]
    return dashboards


@router.get("/admin/mcp/grafana/panels", response_model=List[Dict[str, str]])
def list_grafana_panels(
    user: CurrentSuperuserDep,
    host: Optional[str] = Query(default=None, description="Filter by Grafana host name"),
    dashboard_uid: Optional[str] = Query(default=None, description="Filter by dashboard UID"),
    panel_title: Optional[str] = Query(default=None, description="Filter by panel title"),
) -> List[Dict[str, str]]:
    SiteSetting.update_db_cache()
    panels = getattr(SiteSetting, "mcp_grafana_panels", None) or []
    if host:
        panels = [p for p in panels if str((p or {}).get("host", "")).strip().lower() == host.strip().lower()]
    if dashboard_uid:
        panels = [p for p in panels if str((p or {}).get("dashboard_uid", "")).strip() == dashboard_uid.strip()]
    if panel_title:
        panels = [p for p in panels if str((p or {}).get("panel_title", "")).strip().lower() == panel_title.strip().lower()]
    return panels


@router.post("/admin/mcp/grafana/hosts", response_model=GrafanaMCPHostItem)
def create_grafana_host(
    session: SessionDep, user: CurrentSuperuserDep, req: CreateGrafanaMCPHostRequest
) -> GrafanaMCPHostItem:
    _verify_grafana(req.grafana_url, req.grafana_api_key)
    if req.mcp_ws_url:
        if not (req.mcp_ws_url.startswith("ws://") or req.mcp_ws_url.startswith("wss://")):
            raise HTTPException(status_code=400, detail="mcp_ws_url must be ws:// or wss://")
        _upsert_mcp_host(session, req.name, req.mcp_ws_url)
    else:
        _upsert_mcp_host(session, req.name, f"managed-grafana://{req.name}")
    hosts = _get_grafana_hosts()
    updated: List[Dict[str, str]] = []
    found = False
    for it in hosts:
        if str((it or {}).get("name", "")).strip().lower() == req.name.strip().lower():
            updated.append(req.model_dump())
            found = True
        else:
            updated.append(it)
    if not found:
        updated.append(req.model_dump())
    _save_grafana_hosts(session, updated)
    dashboards = _sync_grafana_dashboards(
        session,
        host_name=req.name,
        grafana_url=req.grafana_url,
        api_key=req.grafana_api_key,
    )
    _sync_grafana_panels(
        session,
        host_name=req.name,
        grafana_url=req.grafana_url,
        api_key=req.grafana_api_key,
        dashboards=dashboards,
    )
    return GrafanaMCPHostItem.model_validate(req.model_dump())
