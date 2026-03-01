"""
Admin API for Prometheus hosts (site setting prometheus_hosts).
"""
from typing import Dict, List, Optional

import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import CurrentSuperuserDep, SessionDep
from app.site_settings import SiteSetting

router = APIRouter()


class PrometheusHostItem(BaseModel):
    name: str = Field(..., description="Display name for this Prometheus host")
    prometheus_url: str = Field(..., description="Prometheus base URL (http/https)")
    bearer_token: Optional[str] = Field(default=None, description="Optional Bearer token for auth")


class CreatePrometheusHostRequest(PrometheusHostItem):
    pass


def _get_prometheus_hosts() -> List[Dict[str, str]]:
    SiteSetting.update_db_cache()
    return getattr(SiteSetting, "prometheus_hosts", None) or []


def _save_prometheus_hosts(session: SessionDep, hosts: List[Dict[str, str]]) -> None:
    SiteSetting.update_setting(session, "prometheus_hosts", hosts)


def _verify_prometheus(prometheus_url: str, bearer_token: Optional[str] = None) -> None:
    if not (prometheus_url.startswith("http://") or prometheus_url.startswith("https://")):
        raise HTTPException(
            status_code=400,
            detail="prometheus_url must start with http:// or https://",
        )
    base = prometheus_url.rstrip("/")
    url = base + "/api/v1/query"
    headers = {}
    if bearer_token and bearer_token.strip():
        headers["Authorization"] = f"Bearer {bearer_token.strip()}"
    try:
        resp = requests.get(
            url,
            params={"query": "1"},
            headers=headers or None,
            timeout=6,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to reach Prometheus: {e}") from e
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=400,
            detail=f"Prometheus API error: {resp.status_code} {resp.text[:500]}",
        )


@router.get("/admin/prometheus/hosts", response_model=List[PrometheusHostItem])
def list_prometheus_hosts(user: CurrentSuperuserDep) -> List[PrometheusHostItem]:
    items = _get_prometheus_hosts()
    return [
        PrometheusHostItem.model_validate(
            {
                "name": (it or {}).get("name", ""),
                "prometheus_url": (it or {}).get("prometheus_url", ""),
                "bearer_token": (it or {}).get("bearer_token"),
            }
        )
        for it in items
    ]


@router.post("/admin/prometheus/hosts", response_model=PrometheusHostItem)
def create_prometheus_host(
    session: SessionDep,
    user: CurrentSuperuserDep,
    req: CreatePrometheusHostRequest,
) -> PrometheusHostItem:
    _verify_prometheus(req.prometheus_url, req.bearer_token)
    hosts = _get_prometheus_hosts()
    updated: List[Dict[str, str]] = []
    found = False
    for it in hosts:
        if str((it or {}).get("name", "")).strip().lower() == req.name.strip().lower():
            updated.append(req.model_dump(exclude_none=True))
            found = True
        else:
            updated.append(it)
    if not found:
        updated.append(req.model_dump(exclude_none=True))
    _save_prometheus_hosts(session, updated)
    return PrometheusHostItem.model_validate(req.model_dump(exclude_none=True))
