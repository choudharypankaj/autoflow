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
            # Try modern mcp SDK
            try:
                from mcp.client.session import ClientSession  # type: ignore
                from mcp.transport.websocket import WebSocketClientTransport  # type: ignore
                async with WebSocketClientTransport(href) as transport:  # type: ignore
                    async with ClientSession(transport) as session:  # type: ignore
                        await session.initialize()
                return
            except Exception:
                pass
            # Try legacy mcp client
            try:
                from mcp.client.websocket import WebSocketClient  # type: ignore
                async with WebSocketClient(href) as client:  # type: ignore
                    await client.initialize()
                return
            except Exception:
                pass
            # Try old modelcontextprotocol
            try:
                from modelcontextprotocol.client.websocket import WebSocketClient  # type: ignore
                async with WebSocketClient(href) as client:  # type: ignore
                    await client.initialize()
                return
            except Exception:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail="mcp/modelcontextprotocol is not installed on server",
                )

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

        async def _check_tidb_connection(agent: Dict[str, Any]) -> None:
            # Validate TiDB connectivity/credentials directly using PyMySQL.
            # Avoid requiring modelcontextprotocol just for settings validation.
            tidb_host = str(agent.get("tidb_host", ""))
            tidb_port = int(agent.get("tidb_port", 0)) if str(agent.get("tidb_port", "")).isdigit() else 0
            tidb_username = str(agent.get("tidb_username", ""))
            tidb_password = str(agent.get("tidb_password", ""))
            tidb_database = str(agent.get("tidb_database", ""))
            if not (tidb_host and tidb_port and tidb_username and tidb_password and tidb_database):
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail="managed_mcp_agents entries must include tidb_host, tidb_port, tidb_username, tidb_password, tidb_database",
                )
            try:
                import pymysql  # type: ignore
            except Exception:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail="pymysql is not installed on server",
                )
            try:
                conn = pymysql.connect(
                    host=tidb_host,
                    port=tidb_port,
                    user=tidb_username,
                    password=tidb_password,
                    database=tidb_database,
                    connect_timeout=6,
                    read_timeout=6,
                    write_timeout=6,
                    charset="utf8mb4",
                    cursorclass=pymysql.cursors.Cursor,  # type: ignore[attr-defined]
                )
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        cur.fetchone()
                finally:
                    conn.close()
            except Exception as e:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f"Failed to connect to TiDB with provided credentials: {e}",
                )

        for agent in agents:
            name = str((agent or {}).get("name", "")).strip() or "<unnamed>"
            try:
                asyncio.run(asyncio.wait_for(_check_tidb_connection(agent), timeout=12))
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
