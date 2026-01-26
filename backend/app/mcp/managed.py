import asyncio
import os
import sys
from typing import Any, Dict, Optional

from app.site_settings import SiteSetting
from sqlmodel import Session, create_engine
from app.core.config import settings
from app.repositories.mcp_database import mcp_database_repo
from app.models.mcp_database import MCPDatabase


class ManagedMCPAgentNotFound(Exception):
    pass


def _get_agent_config(name: str) -> Dict[str, Any]:
    SiteSetting.update_db_cache()
    agents = getattr(SiteSetting, "managed_mcp_agents", None) or []
    for item in agents:
        if str(item.get("name", "")).lower() == name.lower():
            return item
    raise ManagedMCPAgentNotFound(f"Managed MCP agent '{name}' not found")


def _resolve_db_credentials(agent: Dict[str, Any]) -> Dict[str, str]:
    """
    Resolve DB credentials from either inline fields or referenced MCPDatabase by name/id.
    """
    # Inline credentials (backward compatible)
    inline = {
        "TIDB_HOST": str(agent.get("tidb_host", "")),
        "TIDB_PORT": str(agent.get("tidb_port", "")),
        "TIDB_USERNAME": str(agent.get("tidb_username", "")),
        "TIDB_PASSWORD": str(agent.get("tidb_password", "")),
        "TIDB_DATABASE": str(agent.get("tidb_database", "")),
    }
    if all(inline.values()):
        return inline

    # Reference by name or id
    ref_name = str(agent.get("db_name", "")).strip()
    ref_id = int(agent.get("db_id", 0) or 0)
    if not (ref_name or ref_id):
        return inline

    # Create a short-lived session to fetch encrypted credentials
    engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))
    with Session(engine) as session:
        db_obj: MCPDatabase | None
        if ref_id:
            db_obj = mcp_database_repo.get(session, ref_id)
        else:
            db_obj = mcp_database_repo.get_by_name(session, ref_name)
        if not db_obj:
            return inline
        cred = db_obj.credentials or {}
        return {
            "TIDB_HOST": str(cred.get("tidb_host", "")),
            "TIDB_PORT": str(cred.get("tidb_port", "")),
            "TIDB_USERNAME": str(cred.get("tidb_username", "")),
            "TIDB_PASSWORD": str(cred.get("tidb_password", "")),
            "TIDB_DATABASE": str(cred.get("tidb_database", "")),
        }


def _run_direct_db_query(env: Dict[str, str], sql: str) -> Any:
    """
    Fallback path when modelcontextprotocol isn't available:
    run the SQL directly using PyMySQL with provided env credentials.
    """
    try:
        import pymysql  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pymysql is not installed. Install it in the backend image to enable direct DB fallback."
        ) from e

    host = env.get("TIDB_HOST", "")
    port = int(env.get("TIDB_PORT", "0") or 0)
    user = env.get("TIDB_USERNAME", "")
    password = env.get("TIDB_PASSWORD", "")
    database = env.get("TIDB_DATABASE", "")
    if not (host and port and user and password and database):
        raise RuntimeError("Invalid DB credentials for direct query fallback.")

    conn = pymysql.connect(  # type: ignore
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        connect_timeout=8,
        read_timeout=30,
        write_timeout=30,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,  # type: ignore[attr-defined]
    )
    try:
        with conn.cursor() as cur:  # type: ignore
            cur.execute(sql)
            rows = cur.fetchall()
            return rows
    finally:
        conn.close()


async def _run_stdio_tool(env: Dict[str, str], tool: str, params: Dict[str, Any]) -> Any:
    # Extract once so fallback never references an undefined variable
    query_sql = str((params or {}).get("sql", ""))
    try:
        from modelcontextprotocol.client.stdio import StdioClient  # type: ignore
    except Exception as e:
        # Graceful fallback: if we're running a db_query, execute directly via PyMySQL
        if tool == "db_query":
            return _run_direct_db_query(env, query_sql)
        raise RuntimeError(
            "modelcontextprotocol is not installed or incompatible. "
            "Please install 'modelcontextprotocol' in the backend environment."
        ) from e

    cmd = [sys.executable, "-m", "pytidb.ext.mcp"]
    async with StdioClient(cmd, env=env) as client:  # type: ignore
        await client.initialize()
        return await client.call_tool(tool, params)


def run_managed_mcp_db_query(agent_name: str, sql: str) -> Any:
    agent = _get_agent_config(agent_name)

    creds = _resolve_db_credentials(agent)
    env = {**creds, **os.environ}

    return asyncio.run(_run_stdio_tool(env, "db_query", {"sql": sql}))

