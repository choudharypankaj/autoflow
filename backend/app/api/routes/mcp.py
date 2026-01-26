from fastapi import APIRouter
from app.site_settings import SiteSetting

router = APIRouter()


@router.get("/mcp/agents")
def list_mcp_agents() -> dict:
    """
    Returns available MCP agent names from both ws hosts and managed agents.
    No secrets included.
    """
    SiteSetting.update_db_cache()
    ws = getattr(SiteSetting, "mcp_hosts", None) or []
    managed = getattr(SiteSetting, "managed_mcp_agents", None) or []
    ws_names = []
    for item in ws:
        try:
            name = str(item.get("text", "")).strip()
            if name:
                ws_names.append(name)
        except Exception:
            continue
    managed_names = []
    for item in managed:
        try:
            name = str(item.get("name", "")).strip()
            if name:
                managed_names.append(name)
        except Exception:
            continue
    # de-duplicate while preserving order
    names = []
    for n in ws_names + managed_names:
        if n not in names:
            names.append(n)
    return {"agents": names}


@router.get("/mcp/hosts")
def list_mcp_hosts() -> dict:
    """
    Returns MCP WebSocket hosts with names and URLs from site settings.
    """
    SiteSetting.update_db_cache()
    ws = getattr(SiteSetting, "mcp_hosts", None) or []
    hosts = []
    for item in ws:
        try:
            name = str(item.get("text", "")).strip()
            href = str(item.get("href", "")).strip()
            if name and href:
                hosts.append({"text": name, "href": href})
        except Exception:
            continue
    return {"hosts": hosts}

