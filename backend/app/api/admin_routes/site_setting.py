import asyncio
import os
import sys
from typing import Any, Dict, List
from pydantic import BaseModel
from http import HTTPStatus
from fastapi import APIRouter, HTTPException

from app.api.deps import CurrentSuperuserDep, SessionDep
from app.site_settings import SiteSetting, SettingValue, SettingType

router = APIRouter()


@router.get("/admin/site-settings", response_model=Dict[str, SettingValue])
def site_settings(user: CurrentSuperuserDep):
    return SiteSetting.get_all_settings(force_check_db_cache=True)


class SettingUpdate(BaseModel):
    value: SettingType


@router.put(
    "/admin/site-settings/{setting_name}",
    status_code=HTTPStatus.NO_CONTENT,
    responses={
        HTTPStatus.BAD_REQUEST: {
            "content": {
                "application/json": {
                    "examples": {
                        "invalid_data_type": {
                            "summary": "Invalid data type",
                            "value": {"detail": "title must be of type `str`"},
                        },
                    }
                }
            },
        },
        HTTPStatus.NOT_FOUND: {
            "content": {
                "application/json": {
                    "examples": {
                        "setting_not_found": {
                            "summary": "Setting not found",
                            "value": {"detail": "Setting not found"},
                        },
                    }
                }
            },
        },
    },
)
def update_site_setting(
    session: SessionDep,
    user: CurrentSuperuserDep,
    setting_name: str,
    request: SettingUpdate,
):
    if not SiteSetting.setting_exists(setting_name):
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="Setting not found"
        )

    # Special validation for MCP-related settings:
    # - mcp_hosts: verify each ws(s):// URL can be connected with MCP handshake
    # - managed_mcp_agents: verify credentials by running a quick stdio MCP 'SELECT 1'
    if setting_name == "mcp_hosts":
        hosts = request.value or []
        if not isinstance(hosts, list):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="mcp_hosts must be a list of {text, href}",
            )

        async def _check_ws(href: str) -> None:
            try:
                from modelcontextprotocol.client.websocket import WebSocketClient  # type: ignore
            except Exception:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail="modelcontextprotocol is not installed on server",
                )
            async with WebSocketClient(href) as client:
                await client.initialize()

        # Validate each host with a short timeout
        for item in hosts:
            href = str((item or {}).get("href", "")).strip()
            text = str((item or {}).get("text", "")).strip() or href
            if not href or not (href.startswith("ws://") or href.startswith("wss://")):
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f"Invalid MCP host URL for '{text}': {href}",
                )
            try:
                asyncio.run(asyncio.wait_for(_check_ws(href), timeout=8))
            except Exception as e:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f"Failed to connect MCP host '{text}' ({href}): {e}",
                )

    if setting_name == "managed_mcp_agents":
        agents = request.value or []
        if not isinstance(agents, list):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="managed_mcp_agents must be a list of agent configs",
            )

        async def _check_stdio_agent(agent: Dict[str, Any]) -> None:
            try:
                # Local import to avoid hard dependency if unused
                from modelcontextprotocol.client.stdio import StdioClient  # type: ignore
            except Exception:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail="modelcontextprotocol is not installed on server",
                )
            env = {
                "TIDB_HOST": str(agent.get("tidb_host", "")),
                "TIDB_PORT": str(agent.get("tidb_port", "")),
                "TIDB_USERNAME": str(agent.get("tidb_username", "")),
                "TIDB_PASSWORD": str(agent.get("tidb_password", "")),
                "TIDB_DATABASE": str(agent.get("tidb_database", "")),
                **os.environ,  # inherit PATH, VIRTUAL_ENV, etc. so subprocess can find deps
            }
            # Basic sanity
            if not all(env.values()):
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail="managed_mcp_agents entries must include tidb_host, tidb_port, tidb_username, tidb_password, tidb_database",
                )
            # Use the same interpreter running this process to ensure deps are available
            cmd = [sys.executable, "-m", "pytidb.ext.mcp"]
            async with StdioClient(cmd, env=env) as client:  # type: ignore
                await client.initialize()
                # Quick connectivity test
                await client.call_tool("db_query", {"sql": "SELECT 1"})

        for agent in agents:
            name = str((agent or {}).get("name", "")).strip() or "<unnamed>"
            try:
                asyncio.run(asyncio.wait_for(_check_stdio_agent(agent), timeout=12))
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f"Failed to validate managed MCP agent '{name}': {e}",
                )

    try:
        SiteSetting.update_setting(session, setting_name, request.value)
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))
