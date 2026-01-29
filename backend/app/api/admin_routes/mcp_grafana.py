from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import CurrentSuperuserDep
from app.site_settings import SiteSetting
from app.mcp.client import MCPNotConfigured, run_mcp_tool

router = APIRouter()


class GrafanaMCPQueryRequest(BaseModel):
    tool: str = Field("grafana_query_range", description="MCP Grafana tool name")
    params: Dict[str, Any] = Field(..., description="Tool params for Grafana MCP")
    host_name: Optional[str] = Field(None, description="Optional MCP host name")


class GrafanaMCPQueryResponse(BaseModel):
    result: Any


@router.post("/admin/mcp/grafana/query", response_model=GrafanaMCPQueryResponse)
def run_grafana_query(user: CurrentSuperuserDep, request: GrafanaMCPQueryRequest):
    if not request.tool:
        raise HTTPException(status_code=400, detail="tool is required")
    try:
        if request.host_name:
            SiteSetting.update_db_cache()
            hosts = getattr(SiteSetting, "mcp_grafana_hosts", None) or []
            for it in hosts:
                if str((it or {}).get("name", "")).strip().lower() == request.host_name.strip().lower():
                    params = dict(request.params or {})
                    if "grafana_url" not in params:
                        params["grafana_url"] = (it or {}).get("grafana_url", "")
                    if "grafana_api_key" not in params:
                        params["grafana_api_key"] = (it or {}).get("grafana_api_key", "")
                    request = GrafanaMCPQueryRequest(tool=request.tool, params=params, host_name=request.host_name)
                    break
        result = run_mcp_tool(request.tool, request.params, host_name=request.host_name)
        return GrafanaMCPQueryResponse(result=result)
    except MCPNotConfigured as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCP call failed: {e}")
