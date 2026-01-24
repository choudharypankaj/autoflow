from datetime import datetime
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.api.deps import CurrentSuperuserDep
from app.mcp.client import MCPNotConfigured, run_mcp_db_query

router = APIRouter()


AllowedOrderBy = Literal["rocksdb_key_skipped_count", "query_time", "Time"]


class MCPSlowQueriesRequest(BaseModel):
    start_time: datetime = Field(..., description="Start time (UTC).")
    end_time: datetime = Field(..., description="End time (UTC).")
    limit: int = Field(20, ge=1, le=1000)
    order_by: AllowedOrderBy = Field(
        "rocksdb_key_skipped_count",
        description="Column to order by.",
    )
    host_name: Optional[str] = Field(
        None, description="Optional MCP host/agent name to target."
    )

    @field_validator("end_time")
    @classmethod
    def _validate_time_range(cls, v: datetime, info):
        start_time: datetime = info.data.get("start_time")  # type: ignore[assignment]
        if start_time and v <= start_time:
            raise ValueError("end_time must be greater than start_time")
        return v

    def to_sql(self) -> str:
        # All fields are validated; format timestamps safely
        start = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        end = self.end_time.strftime("%Y-%m-%d %H:%M:%S")
        order_col = self.order_by
        limit = self.limit
        # Note: The MCP db_query tool executes raw SQL; we strictly control injected parts.
        return (
            "SELECT Time, INSTANCE, query_time, query, rocksdb_key_skipped_count "
            "FROM information_schema.CLUSTER_SLOW_QUERY "
            "WHERE is_internal = false "
            f"AND Time BETWEEN '{start}' AND '{end}' "
            f"ORDER BY {order_col} DESC "
            f"LIMIT {limit};"
        )


class MCPSlowQueriesResponse(BaseModel):
    result: Any


@router.post("/admin/mcp/slow-queries", response_model=MCPSlowQueriesResponse)
def get_slow_queries(user: CurrentSuperuserDep, request: MCPSlowQueriesRequest):
    sql = request.to_sql()
    try:
        result = run_mcp_db_query(sql, host_name=request.host_name)
        return MCPSlowQueriesResponse(result=result)
    except MCPNotConfigured as e:
        # Try managed agent if name provided
        if request.host_name:
            try:
                from app.mcp.managed import run_managed_mcp_db_query

                result = run_managed_mcp_db_query(request.host_name, sql)
                return MCPSlowQueriesResponse(result=result)
            except Exception as e2:
                raise HTTPException(status_code=400, detail=str(e2))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Generic failure; attempt managed fallback if named
        if request.host_name:
            try:
                from app.mcp.managed import run_managed_mcp_db_query

                result = run_managed_mcp_db_query(request.host_name, sql)
                return MCPSlowQueriesResponse(result=result)
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"MCP call failed: {e2}")
        raise HTTPException(status_code=500, detail=f"MCP call failed: {e}")

