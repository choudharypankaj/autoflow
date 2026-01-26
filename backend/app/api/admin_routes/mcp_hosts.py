from typing import List, Dict, Optional
import asyncio
from http import HTTPStatus
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from app.api.deps import CurrentSuperuserDep, SessionDep
from app.site_settings import SiteSetting

router = APIRouter()


class MCPHostItem(BaseModel):
    text: str = Field(..., description="Display name of the MCP host")
    href: str = Field(..., description="ws:// or wss:// URL")


class CreateMCPHostRequest(MCPHostItem):
    pass


class UpdateMCPHostRequest(BaseModel):
    text: Optional[str] = Field(None, description="New display name")
    href: Optional[str] = Field(None, description="New ws:// or wss:// URL")


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


def _get_hosts() -> List[Dict[str, str]]:
    SiteSetting.update_db_cache()
    return getattr(SiteSetting, "mcp_hosts", None) or []


def _save_hosts(session: SessionDep, hosts: List[Dict[str, str]]) -> None:
    SiteSetting.update_setting(session, "mcp_hosts", hosts)


@router.get("/admin/mcp-hosts", response_model=List[MCPHostItem])
def list_mcp_hosts(user: CurrentSuperuserDep) -> List[MCPHostItem]:
    items = _get_hosts()
    return [MCPHostItem.model_validate({"text": (it or {}).get("text", ""), "href": (it or {}).get("href", "")}) for it in items]


@router.post("/admin/mcp-hosts", response_model=MCPHostItem, status_code=HTTPStatus.CREATED)
def create_mcp_host(session: SessionDep, user: CurrentSuperuserDep, req: CreateMCPHostRequest) -> MCPHostItem:
    name = req.text.strip()
    href = req.href.strip()
    if not href.startswith("ws://") and not href.startswith("wss://"):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="href must start with ws:// or wss://")
    # validate connectivity
    try:
        asyncio.run(asyncio.wait_for(_check_ws(href), timeout=8))
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Failed to connect MCP host '{name}' ({href}): {e}")
    # upsert by name (case-insensitive)
    hosts = _get_hosts()
    updated: List[Dict[str, str]] = []
    replaced = False
    for it in hosts:
        if str((it or {}).get("text", "")).strip().lower() == name.lower():
            updated.append({"text": name, "href": href})
            replaced = True
        else:
            updated.append(it)
    if not replaced:
        updated.append({"text": name, "href": href})
    _save_hosts(session, updated)
    return MCPHostItem(text=name, href=href)


@router.get("/admin/mcp-hosts/{name}", response_model=MCPHostItem)
def get_mcp_host(user: CurrentSuperuserDep, name: str) -> MCPHostItem:
    name_norm = name.strip().lower()
    for it in _get_hosts():
        if str((it or {}).get("text", "")).strip().lower() == name_norm:
            return MCPHostItem(text=str(it.get("text", "")), href=str(it.get("href", "")))
    raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="MCP host not found")


@router.put("/admin/mcp-hosts/{name}", response_model=MCPHostItem)
def update_mcp_host(session: SessionDep, user: CurrentSuperuserDep, name: str, req: UpdateMCPHostRequest) -> MCPHostItem:
    name_norm = name.strip().lower()
    hosts = _get_hosts()
    found: Optional[Dict[str, str]] = None
    for it in hosts:
        if str((it or {}).get("text", "")).strip().lower() == name_norm:
            found = {"text": str(it.get("text", "")), "href": str(it.get("href", ""))}
            break
    if not found:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="MCP host not found")
    new_text = (req.text or found["text"]).strip()
    new_href = (req.href or found["href"]).strip()
    if not new_href.startswith("ws://") and not new_href.startswith("wss://"):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="href must start with ws:// or wss://")
    # validate connectivity (only if href changed)
    if new_href != found["href"]:
        try:
            asyncio.run(asyncio.wait_for(_check_ws(new_href), timeout=8))
        except Exception as e:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=f"Failed to connect MCP host '{new_text}' ({new_href}): {e}")
    # apply update
    updated: List[Dict[str, str]] = []
    for it in hosts:
        if str((it or {}).get("text", "")).strip().lower() == name_norm:
            updated.append({"text": new_text, "href": new_href})
        else:
            updated.append(it)
    _save_hosts(session, updated)
    return MCPHostItem(text=new_text, href=new_href)


@router.delete("/admin/mcp-hosts/{name}", status_code=HTTPStatus.NO_CONTENT)
def delete_mcp_host(session: SessionDep, user: CurrentSuperuserDep, name: str):
    name_norm = name.strip().lower()
    hosts = _get_hosts()
    next_hosts = [it for it in hosts if str((it or {}).get("text", "")).strip().lower() != name_norm]
    if len(next_hosts) == len(hosts):
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="MCP host not found")
    _save_hosts(session, next_hosts)
    return None

