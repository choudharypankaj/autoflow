import asyncio
import logging
from typing import Any, Dict

from app.site_settings import SiteSetting

logger = logging.getLogger(__name__)


class MCPNotConfigured(Exception):
    pass


async def _run_ws_tool(mcp_url: str, tool: str, params: Dict[str, Any]) -> Any:
    try:
        # Lazy import to avoid hard dependency at import time
        from mcp.client.session import ClientSession  # type: ignore
        from mcp.transport.websocket import WebSocketClientTransport  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "MCP Python SDK is not installed or incompatible. "
            "Please install 'mcp' in the backend environment."
        ) from e

    async with WebSocketClientTransport(mcp_url) as transport:  # type: ignore
        async with ClientSession(transport) as session:  # type: ignore
            await session.initialize()
            return await session.call_tool(tool, params)


def _select_mcp_host(preferred_name: str | None = None) -> str:
    """
    Select an MCP host URL from site settings.
    If preferred_name is provided, match against items in mcp_hosts (by .text, case-insensitive).
    Fallback to mcp_host, then first mcp_hosts entry.
    """
    # Ensure latest values
    SiteSetting.update_db_cache()

    # Prefer named host
    hosts = getattr(SiteSetting, "mcp_hosts", None) or []
    if preferred_name:
        for item in hosts:
            try:
                if str(item.get("text", "")).lower() == preferred_name.lower():
                    href = str(item.get("href", "")).strip()
                    if href:
                        return href
            except Exception:
                continue
        # If a preferred name was provided but not found among ws hosts,
        # do not silently fall back to a different host. Force empty to let callers decide.
        return ""
    # Single host fallback
    single = str(getattr(SiteSetting, "mcp_host", "") or "").strip()
    if single:
        return single
    # First in list
    if hosts:
        href = str((hosts[0] or {}).get("href", "")).strip()
        if href:
            return href
    return ""


def run_mcp_tool(tool: str, params: Dict[str, Any], *, host_name: str | None = None) -> Any:
    """
    Run an MCP tool against the configured MCP host.
    Only WebSocket/WSS MCP host is supported (e.g., ws://..., wss://...).
    """
    mcp_url = _select_mcp_host(host_name)
    if not mcp_url:
        raise MCPNotConfigured("mcp_host is not configured. Set it in Site Settings.")

    # Support virtual managed hosts: managed://<agent_name>
    if mcp_url.startswith("managed://"):
        agent_name = mcp_url[len("managed://") :].strip()
        if not agent_name:
            raise ValueError("Invalid managed MCP URL")
        # For now we support db_query tool via managed agent
        if tool == "db_query":
            from app.mcp.managed import run_managed_mcp_db_query  # local import

            return run_managed_mcp_db_query(agent_name, params.get("sql", ""))
        raise ValueError(f"Tool '{tool}' is not supported via managed MCP host")

    if not (mcp_url.startswith("ws://") or mcp_url.startswith("wss://")):
        raise ValueError("Only ws:// or wss:// MCP host URLs are supported.")

    # Run async client in a fresh event loop (sync entry)
    return asyncio.run(_run_ws_tool(mcp_url, tool, params))


def run_mcp_db_query(sql: str, *, host_name: str | None = None) -> Any:
    """
    Convenience wrapper to call the TiDB MCP Server 'db_query' tool.
    """
    return run_mcp_tool("db_query", {"sql": sql}, host_name=host_name)

