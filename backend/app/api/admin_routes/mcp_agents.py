from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import CurrentSuperuserDep, SessionDep
from app.site_settings import SiteSetting

router = APIRouter()


class CreateAgentRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Agent name (used in prompts)")
    tidb_host: str = Field(..., description="MySQL-compatible host")
    tidb_port: int = Field(..., ge=1, le=65535, description="Port")
    tidb_username: str = Field(..., min_length=1, description="Username")
    tidb_password: str = Field(..., min_length=1, description="Password")
    tidb_database: str = Field(..., min_length=1, description="Database name")
    # Optional: if user also provides a WS MCP host to register
    mcp_ws_url: Optional[str] = Field(
        default=None, description="Optional ws:// or wss:// MCP server URL"
    )


class CreateAgentResponse(BaseModel):
    success: bool
    verified_db: bool
    verified_ws: bool
    message: str = ""
    agent: Dict[str, Any] = {}


def _verify_db_conn(req: CreateAgentRequest) -> None:
    try:
        import pymysql  # type: ignore
    except Exception:
        raise HTTPException(status_code=400, detail="pymysql is not installed on server")
    try:
        conn = pymysql.connect(  # type: ignore
            host=req.tidb_host,
            port=req.tidb_port,
            user=req.tidb_username,
            password=req.tidb_password,
            database=req.tidb_database,
            connect_timeout=6,
            read_timeout=6,
            write_timeout=6,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.Cursor,  # type: ignore[attr-defined]
        )
        try:
            with conn.cursor() as cur:  # type: ignore
                cur.execute("SELECT 1")
                cur.fetchone()
        finally:
            conn.close()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to connect to database: {e}",
        )


async def _verify_ws(url: str) -> None:
    if not (url.startswith("ws://") or url.startswith("wss://")):
        raise HTTPException(status_code=400, detail="mcp_ws_url must be ws:// or wss://")
    try:
        from modelcontextprotocol.client.websocket import WebSocketClient  # type: ignore
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="modelcontextprotocol is not installed on server",
        )
    async with WebSocketClient(url) as client:  # type: ignore
        await client.initialize()


def _upsert_managed_agent(session: SessionDep, req: CreateAgentRequest) -> Dict[str, Any]:
    SiteSetting.update_db_cache()
    agents: List[Dict[str, Any]] = getattr(SiteSetting, "managed_mcp_agents", None) or []
    new_item = {
        "name": req.name,
        "tidb_host": req.tidb_host,
        "tidb_port": str(req.tidb_port),
        "tidb_username": req.tidb_username,
        "tidb_password": req.tidb_password,
        "tidb_database": req.tidb_database,
    }
    updated: List[Dict[str, Any]] = []
    found = False
    for item in agents:
        if str(item.get("name", "")).strip().lower() == req.name.strip().lower():
            updated.append(new_item)
            found = True
        else:
            updated.append(item)
    if not found:
        updated.append(new_item)
    SiteSetting.update_setting(session, "managed_mcp_agents", updated)
    return new_item


def _maybe_upsert_ws_host(session: SessionDep, name: str, url: str) -> Dict[str, Any]:
    SiteSetting.update_db_cache()
    hosts: List[Dict[str, Any]] = getattr(SiteSetting, "mcp_hosts", None) or []
    new_item = {"text": name, "href": url}
    updated: List[Dict[str, Any]] = []
    found = False
    for item in hosts:
        if str(item.get("text", "")).strip().lower() == name.strip().lower():
            updated.append(new_item)
            found = True
        else:
            updated.append(item)
    if not found:
        updated.append(new_item)
    SiteSetting.update_setting(session, "mcp_hosts", updated)
    return new_item


def _upsert_managed_virtual_host(session: SessionDep, name: str) -> Dict[str, Any]:
    """
    Create a virtual MCP host entry that points to the managed agent by name.
    href uses managed://<name> scheme, which the MCP client understands.
    """
    return _maybe_upsert_ws_host(session, name, f"managed://{name}")


@router.post("/admin/mcp/agents", response_model=CreateAgentResponse)
async def create_managed_agent(session: SessionDep, user: CurrentSuperuserDep, req: CreateAgentRequest):
    # a) Verify connectivity to the database
    _verify_db_conn(req)
    verified_db = True

    verified_ws = False
    # b) Optionally verify and add a WS MCP host; otherwise create a virtual managed host
    if req.mcp_ws_url:
        await _verify_ws(req.mcp_ws_url)
        _maybe_upsert_ws_host(session, req.name, req.mcp_ws_url)
        verified_ws = True
    else:
        _upsert_managed_virtual_host(session, req.name)

    # Always upsert managed agent
    agent = _upsert_managed_agent(session, req)

    return CreateAgentResponse(
        success=True,
        verified_db=verified_db,
        verified_ws=verified_ws,
        message="Agent created and verified",
        agent=agent,
    )
