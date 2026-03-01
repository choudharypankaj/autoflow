"""
Admin API for Prometheus: run flexible queries via HTTP API.
"""
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import CurrentSuperuserDep
from app.prometheus.client import prometheus_query, prometheus_query_range
from app.site_settings import SiteSetting

router = APIRouter()


class PrometheusQueryRequest(BaseModel):
    host_name: Optional[str] = Field(None, description="Prometheus host name from prometheus_hosts")
    query: str = Field(..., description="PromQL expression")
    time_sec: Optional[float] = Field(None, description="Instant query time (unix seconds)")
    start_sec: Optional[float] = Field(None, description="Range query start (unix seconds)")
    end_sec: Optional[float] = Field(None, description="Range query end (unix seconds)")
    step_sec: Optional[int] = Field(60, description="Range query step in seconds")


class PrometheusQueryResponse(BaseModel):
    result: Any


def _get_prometheus_entry(host_name: Optional[str]) -> tuple[dict, str | None]:
    SiteSetting.update_db_cache()
    hosts = getattr(SiteSetting, "prometheus_hosts", None) or []
    if host_name:
        for it in hosts:
            if str((it or {}).get("name", "")).strip().lower() == host_name.strip().lower():
                return it, str((it or {}).get("name", "")).strip()
    if hosts:
        entry = hosts[0]
        return entry, str((entry or {}).get("name", "")).strip()
    return {}, None


@router.post("/admin/prometheus/query", response_model=PrometheusQueryResponse)
def run_prometheus_query(user: CurrentSuperuserDep, request: PrometheusQueryRequest):
    """Run a Prometheus instant or range query via the HTTP API."""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    entry, name = _get_prometheus_entry(request.host_name)
    if not entry:
        raise HTTPException(status_code=400, detail="No Prometheus host configured. Add prometheus_hosts in site settings.")
    base_url = str((entry or {}).get("prometheus_url", "")).strip().rstrip("/")
    bearer_token = str((entry or {}).get("bearer_token", "")).strip() or None
    if not base_url or not base_url.startswith("http"):
        raise HTTPException(status_code=400, detail="Prometheus URL missing or invalid for this host.")
    try:
        if request.start_sec is not None and request.end_sec is not None:
            result = prometheus_query_range(
                base_url,
                request.query,
                start_sec=request.start_sec,
                end_sec=request.end_sec,
                step_sec=request.step_sec or 60,
                bearer_token=bearer_token,
            )
        else:
            result = prometheus_query(
                base_url,
                request.query,
                time_sec=request.time_sec,
                bearer_token=bearer_token,
            )
        return PrometheusQueryResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prometheus API call failed: {e}") from e
