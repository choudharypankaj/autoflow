from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import CurrentSuperuserDep, SessionDep
from app.site_settings import SiteSetting

router = APIRouter()


class CreateAgentRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Agent name (used in prompts)")
    # Either provide inline credentials, or reference a saved MCP database by name/id.
    tidb_host: Optional[str] = Field(None, description="MySQL-compatible host")
    tidb_port: Optional[int] = Field(None, ge=1, le=65535, description="Port")
    tidb_username: Optional[str] = Field(None, min_length=1, description="Username")
    tidb_password: Optional[str] = Field(None, min_length=1, description="Password")
    tidb_database: Optional[str] = Field(None, min_length=1, description="Database name")
    db_name: Optional[str] = Field(None, description="Reference MCP Database by name")
    db_id: Optional[int] = Field(None, description="Reference MCP Database by id")
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


def _verify_db_conn(req: CreateAgentRequest) -> Dict[str, Any]:
    try:
        import pymysql  # type: ignore
    except Exception:
        raise HTTPException(status_code=400, detail="pymysql is not installed on server")
    # Prefer inline credentials; otherwise resolve from referenced MCP database via public endpoint
    creds: Dict[str, Any] = {}
    if req.tidb_host and req.tidb_port and req.tidb_username and req.tidb_password and req.tidb_database:
        creds = {
            "host": req.tidb_host,
            "port": req.tidb_port,
            "user": req.tidb_username,
            "password": req.tidb_password,
            "database": req.tidb_database,
        }
    elif req.db_name or req.db_id:
        # Lazy import to avoid circular deps
        from sqlmodel import Session, create_engine
        from app.core.config import settings
        from app.repositories.mcp_database import mcp_database_repo
        engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))
        with Session(engine) as session:
            db_obj = mcp_database_repo.get(session, int(req.db_id)) if req.db_id else mcp_database_repo.get_by_name(session, str(req.db_name))
            if not db_obj:
                raise HTTPException(status_code=400, detail="Referenced MCP database not found")
            c = db_obj.credentials or {}
            creds = {
                "host": str(c.get("tidb_host", "")),
                "port": int(str(c.get("tidb_port", "0")) or "0"),
                "user": str(c.get("tidb_username", "")),
                "password": str(c.get("tidb_password", "")),
                "database": str(c.get("tidb_database", "")),
            }
    else:
        raise HTTPException(status_code=400, detail="Provide either inline DB credentials or db_name/db_id")
    try:
        conn = pymysql.connect(  # type: ignore
            host=creds["host"],
            port=creds["port"],
            user=creds["user"],
            password=creds["password"],
            database=creds["database"],
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
    return creds


async def _verify_ws(url: str) -> None:
    if not (url.startswith("ws://") or url.startswith("wss://")):
        raise HTTPException(status_code=400, detail="mcp_ws_url must be ws:// or wss://")
    try:
        from mcp.client.session import ClientSession  # type: ignore
        from mcp.transport.websocket import WebSocketClientTransport  # type: ignore
    except Exception:
        try:
            from mcp.client.websocket import WebSocketClient  # type: ignore
            async with WebSocketClient(url) as client:  # type: ignore
                await client.initialize()
            return
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="MCP Python SDK not available (need mcp.client.session+mcp.transport.websocket or mcp.client.websocket).",
            )
    async with WebSocketClientTransport(url) as transport:  # type: ignore
        async with ClientSession(transport) as session:  # type: ignore
            await session.initialize()


def _upsert_managed_agent(session: SessionDep, req: CreateAgentRequest, resolved_creds: Dict[str, Any]) -> Dict[str, Any]:
    SiteSetting.update_db_cache()
    agents: List[Dict[str, Any]] = getattr(SiteSetting, "managed_mcp_agents", None) or []
    # Prefer reference storage if db_name/db_id provided, otherwise keep inline for backward compatibility
    if req.db_name or req.db_id:
        new_item = {
            "name": req.name,
            "db_name": req.db_name,
            "db_id": req.db_id,
        }
    else:
        new_item = {
            "name": req.name,
            "tidb_host": str(resolved_creds.get("host", "")),
            "tidb_port": str(resolved_creds.get("port", "")),
            "tidb_username": str(resolved_creds.get("user", "")),
            "tidb_password": str(resolved_creds.get("password", "")),
            "tidb_database": str(resolved_creds.get("database", "")),
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
    resolved = _verify_db_conn(req)
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
    agent = _upsert_managed_agent(session, req, resolved)

    return CreateAgentResponse(
        success=True,
        verified_db=verified_db,
        verified_ws=verified_ws,
        message="Agent created and verified",
        agent=agent,
    )
