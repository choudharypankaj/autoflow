import asyncio
import logging
import importlib
from importlib import metadata as importlib_metadata
import json
import ast
import re
from typing import Any, Dict

from app.site_settings import SiteSetting

logger = logging.getLogger(__name__)
def _unwrap_mcp_result(result: Any) -> Any:
    def _parse_text(text: str) -> Any:
        match = re.search(r"text='((?:\\'|[^'])*?)'|text=\"((?:\\\"|[^\"])*?)\"", text, flags=re.DOTALL)
        if match:
            raw = match.group(1) or match.group(2) or ""
            try:
                text = ast.literal_eval(f"'{raw}'" if match.group(1) else f'\"{raw}\"')
            except Exception:
                pass
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(text)
            except Exception:
                continue
        for opener, closer in [("{", "}"), ("[", "]")]:
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end != -1 and end > start:
                chunk = text[start : end + 1]
                for parser in (json.loads, ast.literal_eval):
                    try:
                        return parser(chunk)
                    except Exception:
                        continue
        return text

    if isinstance(result, str):
        logger.info("MCP unwrap (ws) input=str preview=%s", result[:200].replace("\n", "\\n"))
        parsed = _parse_text(result)
        logger.info("MCP unwrap (ws) parsed type=%s", type(parsed).__name__)
        return parsed
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            logger.info("MCP unwrap (ws) content list size=%s", len(content))
            parsed_items = []
            for item in content:
                text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                if not text:
                    continue
                if isinstance(text, str):
                    logger.info("MCP unwrap (ws) text preview=%s", text[:200].replace("\n", "\\n"))
                    parsed_items.append(_parse_text(text))
            if len(parsed_items) == 1:
                return parsed_items[0]
            if parsed_items:
                return parsed_items
    content = getattr(result, "content", None)
    if isinstance(content, list):
        logger.info("MCP unwrap (ws) object content list size=%s", len(content))
        parsed_items = []
        for item in content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if not text:
                continue
            if isinstance(text, str):
                logger.info("MCP unwrap (ws) object text preview=%s", text[:200].replace("\n", "\\n"))
                parsed_items.append(_parse_text(text))
        if len(parsed_items) == 1:
            return parsed_items[0]
        if parsed_items:
            return parsed_items
    logger.info("MCP unwrap (ws) passthrough type=%s", type(result).__name__)
    return result



class MCPNotConfigured(Exception):
    pass


async def _run_ws_tool(mcp_url: str, tool: str, params: Dict[str, Any]) -> Any:
    try:
        ws_mod = importlib.import_module("mcp.client.websocket")
    except Exception as e:
        diagnostics: Dict[str, Any] = {"mcp_version": None, "mcp_client_websocket_attrs": None}
        try:
            diagnostics["mcp_version"] = importlib_metadata.version("mcp")
        except Exception:
            diagnostics["mcp_version"] = "unknown"
        try:
            ws_mod = importlib.import_module("mcp.client.websocket")
            diagnostics["mcp_client_websocket_attrs"] = sorted(
                {name for name in dir(ws_mod) if not name.startswith("_")}
            )
        except Exception as e2:
            diagnostics["mcp_client_websocket_attrs"] = f"import_failed: {e2}"
        logger.error("MCP websocket import failed: %s", diagnostics)
        raise RuntimeError(
            "MCP Python SDK not available (expected mcp.client.websocket). "
            "Install the official SDK into the app venv, e.g.: "
            "/app/.venv/bin/python -m pip install 'mcp[client] @ "
            "git+https://github.com/modelcontextprotocol/python-sdk@v0.1.0'"
        ) from e
    # Path A: WebSocketClient class
    if hasattr(ws_mod, "WebSocketClient"):
        WebSocketClient = getattr(ws_mod, "WebSocketClient")
        async with WebSocketClient(mcp_url) as client:  # type: ignore
            await client.initialize()
            if tool == "db_query" and "sql" in params and "sql_stmt" not in params:
                params = {**params, "sql_stmt": params["sql"]}
            result = await client.call_tool(tool, params)
            return _unwrap_mcp_result(result)
    # Path B: websocket_client + ClientSession
    if hasattr(ws_mod, "websocket_client"):
        websocket_client = getattr(ws_mod, "websocket_client")
        session_mod = importlib.import_module("mcp.client.session")
        ClientSession = getattr(session_mod, "ClientSession")
        async with websocket_client(mcp_url) as (read_stream, write_stream):  # type: ignore
            async with ClientSession(read_stream, write_stream) as session:  # type: ignore
                await session.initialize()
                if tool == "db_query" and "sql" in params and "sql_stmt" not in params:
                    params = {**params, "sql_stmt": params["sql"]}
                result = await session.call_tool(tool, params)
                return _unwrap_mcp_result(result)
    logger.error(
        "MCP websocket module missing WebSocketClient/websocket_client: %s",
        sorted({name for name in dir(ws_mod) if not name.startswith("_")}),
    )
    raise RuntimeError(
        "MCP Python SDK not available (expected mcp.client.websocket WebSocketClient or websocket_client). "
        "Install the official SDK into the app venv, e.g.: "
        "/app/.venv/bin/python -m pip install 'mcp[client] @ "
        "git+https://github.com/modelcontextprotocol/python-sdk@v0.1.0'"
    )


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

    # Support virtual managed Grafana hosts: managed-grafana://<name>
    if mcp_url.startswith("managed-grafana://"):
        grafana_name = mcp_url[len("managed-grafana://") :].strip()
        if not grafana_name:
            raise ValueError("Invalid managed Grafana MCP URL")
        if not tool.startswith("grafana_"):
            raise ValueError(f"Tool '{tool}' is not supported via managed Grafana MCP host")
        from app.mcp.managed import run_managed_mcp_grafana_tool  # local import

        return run_managed_mcp_grafana_tool(grafana_name, tool, params)

    if not (mcp_url.startswith("ws://") or mcp_url.startswith("wss://")):
        raise ValueError("Only ws:// or wss:// MCP host URLs are supported.")

    # Run async client in a fresh event loop (sync entry)
    return asyncio.run(_run_ws_tool(mcp_url, tool, params))


def run_mcp_tool_url(mcp_url: str, tool: str, params: Dict[str, Any]) -> Any:
    """
    Run an MCP tool against a specific MCP URL (ws:// or wss://).
    """
    if not mcp_url:
        raise MCPNotConfigured("mcp_url is required.")
    if not (mcp_url.startswith("ws://") or mcp_url.startswith("wss://")):
        raise ValueError("Only ws:// or wss:// MCP host URLs are supported.")
    return asyncio.run(_run_ws_tool(mcp_url, tool, params))


def run_mcp_db_query(sql: str, *, host_name: str | None = None) -> Any:
    """
    Convenience wrapper to call the TiDB MCP Server 'db_query' tool.
    """
    return run_mcp_tool("db_query", {"sql": sql}, host_name=host_name)

