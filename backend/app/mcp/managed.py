import asyncio
import logging
import importlib
from importlib import metadata as importlib_metadata
import os
import shutil
import sys
import json
import ast
import re
from typing import Any, Dict, Optional

from app.site_settings import SiteSetting
from sqlmodel import Session, create_engine
from app.core.config import settings
from app.repositories.mcp_database import mcp_database_repo
from app.models.mcp_database import MCPDatabase

logger = logging.getLogger(__name__)
def _unwrap_mcp_result(result: Any) -> Any:
    def _parse_text(text: str) -> Any:
        # Extract wrapper content when the whole object is stringified
        match = re.search(r"text='((?:\\'|[^'])*?)'|text=\"((?:\\\"|[^\"])*?)\"", text, flags=re.DOTALL)
        if match:
            raw = match.group(1) or match.group(2) or ""
            try:
                text = ast.literal_eval(f"'{raw}'" if match.group(1) else f'\"{raw}\"')
            except Exception:
                pass
        # Try JSON / literal eval
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(text)
            except Exception:
                continue
        # Try to extract JSON substring
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
        logger.info("MCP unwrap (managed) input=str preview=%s", result[:200].replace("\n", "\\n"))
        parsed = _parse_text(result)
        logger.info("MCP unwrap (managed) parsed type=%s", type(parsed).__name__)
        return parsed
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            logger.info("MCP unwrap (managed) content list size=%s", len(content))
            for item in content:
                text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                if not text:
                    continue
                if isinstance(text, str):
                    logger.info("MCP unwrap (managed) text preview=%s", text[:200].replace("\n", "\\n"))
                    return _parse_text(text)
    logger.info("MCP unwrap (managed) passthrough type=%s", type(result).__name__)
    return result


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


async def _run_stdio_tool(env: Dict[str, str], tool: str, params: Dict[str, Any]) -> Any:
    cmd = [sys.executable, "-m", "pytidb.ext.mcp"]
    try:
        stdio_mod = importlib.import_module("mcp.client.stdio")
    except Exception as e:
        diagnostics: Dict[str, Any] = {"mcp_version": None, "mcp_client_stdio_attrs": None}
        try:
            diagnostics["mcp_version"] = importlib_metadata.version("mcp")
        except Exception:
            diagnostics["mcp_version"] = "unknown"
        try:
            stdio_mod = importlib.import_module("mcp.client.stdio")
            diagnostics["mcp_client_stdio_attrs"] = sorted(
                {name for name in dir(stdio_mod) if not name.startswith("_")}
            )
        except Exception as e2:
            diagnostics["mcp_client_stdio_attrs"] = f"import_failed: {e2}"
        logger.error("MCP stdio import failed: %s", diagnostics)
        raise RuntimeError(
            "MCP Python SDK not available (expected mcp.client.stdio). "
            "Install the official SDK into the app venv, e.g.: "
            "/app/.venv/bin/python -m pip install 'mcp[client] @ "
            "git+https://github.com/modelcontextprotocol/python-sdk@v0.1.0'"
        ) from e
    # Path A: StdioClient class
    if hasattr(stdio_mod, "StdioClient"):
        StdioClient = getattr(stdio_mod, "StdioClient")
        async with StdioClient(cmd, env=env) as client:  # type: ignore
            await client.initialize()
            if tool == "db_query" and "sql" in params and "sql_stmt" not in params:
                params = {**params, "sql_stmt": params["sql"]}
            result = await client.call_tool(tool, params)
            return _unwrap_mcp_result(result)
    # Path B: stdio_client + ClientSession
    if hasattr(stdio_mod, "stdio_client") and hasattr(stdio_mod, "StdioServerParameters"):
        stdio_client = getattr(stdio_mod, "stdio_client")
        StdioServerParameters = getattr(stdio_mod, "StdioServerParameters")
        session_mod = importlib.import_module("mcp.client.session")
        ClientSession = getattr(session_mod, "ClientSession")
        # Build StdioServerParameters with correct shape for this SDK version
        field_names = set(getattr(StdioServerParameters, "model_fields", {}).keys())
        preferred_exe = "/app/.venv/bin/python"
        command_exe = preferred_exe if os.path.exists(preferred_exe) else sys.executable
        if not os.path.exists(command_exe):
            resolved = shutil.which(command_exe) if command_exe else None
            logger.error("MCP stdio command not found: %s (resolved=%s)", command_exe, resolved)
        server_params_data = {"command": command_exe, "env": env}
        if "args" in field_names:
            server_params_data["args"] = ["-m", "pytidb.ext.mcp"]
        elif "command_args" in field_names:
            server_params_data["command_args"] = ["-m", "pytidb.ext.mcp"]
        else:
            # Fallback: single string command
            server_params_data["command"] = f"{command_exe} -m pytidb.ext.mcp"
        logger.info("MCP stdio params: %s", server_params_data)
        server_params = StdioServerParameters(**server_params_data)
        async with stdio_client(server_params) as (read_stream, write_stream):  # type: ignore
            async with ClientSession(read_stream, write_stream) as session:  # type: ignore
                await session.initialize()
                tool_params = params
                if tool == "db_query" and "sql" in tool_params and "sql_stmt" not in tool_params:
                    tool_params = {**tool_params, "sql_stmt": tool_params["sql"]}
                result = await session.call_tool(tool, tool_params)
                return _unwrap_mcp_result(result)
    logger.error(
        "MCP stdio module missing StdioClient/stdio_client: %s",
        sorted({name for name in dir(stdio_mod) if not name.startswith("_")}),
    )
    raise RuntimeError(
        "MCP Python SDK not available (expected mcp.client.stdio StdioClient or stdio_client). "
        "Install the official SDK into the app venv, e.g.: "
        "/app/.venv/bin/python -m pip install 'mcp[client] @ "
        "git+https://github.com/modelcontextprotocol/python-sdk@v0.1.0'"
    )


def run_managed_mcp_db_query(agent_name: str, sql: str) -> Any:
    agent = _get_agent_config(agent_name)

    creds = _resolve_db_credentials(agent)
    env = {**creds, **os.environ}

    return asyncio.run(_run_stdio_tool(env, "db_query", {"sql": sql}))

