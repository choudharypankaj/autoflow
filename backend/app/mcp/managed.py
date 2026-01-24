import asyncio
import os
from typing import Any, Dict, Optional

from app.site_settings import SiteSetting


class ManagedMCPAgentNotFound(Exception):
    pass


def _get_agent_config(name: str) -> Dict[str, Any]:
    SiteSetting.update_db_cache()
    agents = getattr(SiteSetting, "managed_mcp_agents", None) or []
    for item in agents:
        if str(item.get("name", "")).lower() == name.lower():
            return item
    raise ManagedMCPAgentNotFound(f"Managed MCP agent '{name}' not found")


async def _run_stdio_tool(env: Dict[str, str], tool: str, params: Dict[str, Any]) -> Any:
    try:
        from modelcontextprotocol.client.stdio import StdioClient  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "modelcontextprotocol is not installed or incompatible. "
            "Please install 'modelcontextprotocol' in the backend environment."
        ) from e

    cmd = ["python", "-m", "pytidb.ext.mcp"]
    async with StdioClient(cmd, env=env) as client:  # type: ignore
        await client.initialize()
        return await client.call_tool(tool, params)


def run_managed_mcp_db_query(agent_name: str, sql: str) -> Any:
    agent = _get_agent_config(agent_name)

    env = {
        "TIDB_HOST": str(agent.get("tidb_host", "")),
        "TIDB_PORT": str(agent.get("tidb_port", "")),
        "TIDB_USERNAME": str(agent.get("tidb_username", "")),
        "TIDB_PASSWORD": str(agent.get("tidb_password", "")),
        "TIDB_DATABASE": str(agent.get("tidb_database", "")),
        **os.environ,
    }

    return asyncio.run(_run_stdio_tool(env, "db_query", {"sql": sql}))

