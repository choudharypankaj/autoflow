from typing import Dict, List, Optional

import requests
from fastapi import APIRouter, HTTPException
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
    url = grafana_url.rstrip("/") + "/api/health"
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=6,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to reach Grafana: {e}")
    if resp.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Grafana auth failed: {resp.status_code} {resp.text}")


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


@router.post("/admin/mcp/grafana/hosts", response_model=GrafanaMCPHostItem)
def create_grafana_host(
    session: SessionDep, user: CurrentSuperuserDep, req: CreateGrafanaMCPHostRequest
) -> GrafanaMCPHostItem:
    _verify_grafana(req.grafana_url, req.grafana_api_key)
    if req.mcp_ws_url:
        if not (req.mcp_ws_url.startswith("ws://") or req.mcp_ws_url.startswith("wss://")):
            raise HTTPException(status_code=400, detail="mcp_ws_url must be ws:// or wss://")
        _upsert_mcp_host(session, req.name, req.mcp_ws_url)
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
    return GrafanaMCPHostItem.model_validate(req.model_dump())
