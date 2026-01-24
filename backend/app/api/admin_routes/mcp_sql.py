import re
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import SessionDep, CurrentSuperuserDep
from app.mcp.client import MCPNotConfigured, run_mcp_db_query


router = APIRouter()


class MCPAdhocQueryRequest(BaseModel):
    sql: str = Field(..., description="Read-only SQL query (SELECT/CTE only)")
    host_name: str | None = Field(None, description="Optional MCP host name (matches mcp_hosts[].text)")


class MCPAdhocQueryResponse(BaseModel):
    result: Any


_FORBIDDEN_KEYWORDS = [
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
    "replace",
    "merge",
    "call",
    "execute",
    "into",
    "begin",
    "commit",
    "rollback",
]


def _validate_readonly_sql(sql: str) -> None:
    if len(sql) > 20000:
        raise HTTPException(status_code=400, detail="SQL too long")
    cleaned = sql.strip().rstrip(";")
    if not cleaned:
        raise HTTPException(status_code=400, detail="Empty SQL")
    lowered = cleaned.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise HTTPException(status_code=400, detail="Only SELECT/CTE queries are allowed")
    for kw in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\\b{kw}\\b", lowered):
            raise HTTPException(status_code=400, detail=f"Keyword '{kw}' is not allowed")
    if sql.count(";") > 1:
        raise HTTPException(status_code=400, detail="Multiple statements are not allowed")


@router.post("/admin/mcp/sql/run", response_model=MCPAdhocQueryResponse)
def run_mcp_sql(user: CurrentSuperuserDep, request: MCPAdhocQueryRequest):
    _validate_readonly_sql(request.sql)
    try:
        result = run_mcp_db_query(request.sql, host_name=request.host_name)
        return MCPAdhocQueryResponse(result=result)
    except MCPNotConfigured as e:
        # try managed agent if provided
        if request.host_name:
            try:
                from app.mcp.managed import run_managed_mcp_db_query
                result = run_managed_mcp_db_query(request.host_name, request.sql)
                return MCPAdhocQueryResponse(result=result)
            except Exception as e2:
                raise HTTPException(status_code=400, detail=str(e2))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # generic failure; attempt managed fallback if named
        if request.host_name:
            try:
                from app.mcp.managed import run_managed_mcp_db_query
                result = run_managed_mcp_db_query(request.host_name, request.sql)
                return MCPAdhocQueryResponse(result=result)
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"MCP call failed: {e2}")
        raise HTTPException(status_code=500, detail=f"MCP call failed: {e}")

