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

import requests

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
            parsed_items = []
            for item in content:
                text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                if not text:
                    continue
                if isinstance(text, str):
                    logger.info("MCP unwrap (managed) text preview=%s", text[:200].replace("\n", "\\n"))
                    parsed_items.append(_parse_text(text))
            if len(parsed_items) == 1:
                return parsed_items[0]
            if parsed_items:
                return parsed_items
    content = getattr(result, "content", None)
    if isinstance(content, list):
        logger.info("MCP unwrap (managed) object content list size=%s", len(content))
        parsed_items = []
        for item in content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if not text:
                continue
            if isinstance(text, str):
                logger.info("MCP unwrap (managed) object text preview=%s", text[:200].replace("\n", "\\n"))
                parsed_items.append(_parse_text(text))
        if len(parsed_items) == 1:
            return parsed_items[0]
        if parsed_items:
            return parsed_items
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


def _get_grafana_config(name: str) -> Dict[str, Any]:
    SiteSetting.update_db_cache()
    hosts = getattr(SiteSetting, "mcp_grafana_hosts", None) or []
    for item in hosts:
        if str(item.get("name", "")).strip().lower() == name.lower():
            return item
    raise ManagedMCPAgentNotFound(f"Grafana MCP host '{name}' not found")


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


def run_managed_mcp_grafana_tool(name: str, tool: str, params: Dict[str, Any]) -> Any:
    config = _get_grafana_config(name)
    grafana_url = str(config.get("grafana_url", "")).strip().rstrip("/")
    api_key = str(config.get("grafana_api_key", "")).strip()
    if not grafana_url or not api_key:
        raise RuntimeError("Grafana MCP host missing shown grafana_url or grafana_api_key")

    headers = {"Authorization": f"Bearer {api_key}"}
    logger.info("Grafana MCP request tool=%s params=%s", tool, params)
    if tool in {"grafana_query_range", "grafana_query"}:
        queries = params.get("queries") if isinstance(params, dict) else None
        if isinstance(queries, list):
            has_datasource = any(isinstance(q, dict) and q.get("datasource") for q in queries)
            if not has_datasource:
                try:
                    ds_resp = requests.get(grafana_url + "/api/datasources", headers=headers, timeout=10)
                    if ds_resp.status_code < 400:
                        ds_list = ds_resp.json() or []
                        default_ds = next((d for d in ds_list if d.get("isDefault")), None) or (ds_list[0] if ds_list else None)
                        if isinstance(default_ds, dict) and default_ds.get("uid"):
                            ds_info = {"uid": default_ds.get("uid"), "type": default_ds.get("type", "prometheus")}
                            params = dict(params)
                            params["queries"] = [
                                ({**q, "datasource": ds_info} if isinstance(q, dict) and "datasource" not in q else q)
                                for q in queries
                            ]
                except Exception:
                    pass
        logger.info("Grafana MCP request params_resolved=%s", params)
    if tool == "grafana_list_dashboards":
        resp = requests.get(
            grafana_url + "/api/search",
            headers=headers,
            params={"type": "dash-db"},
            timeout=10,
        )
    elif tool == "grafana_list_panels":
        uid = str(params.get("uid", "")).strip() if isinstance(params, dict) else ""
        if not uid:
            raise RuntimeError("Grafana panel list requires dashboard uid")
        resp = requests.get(
            grafana_url + f"/api/dashboards/uid/{uid}",
            headers=headers,
            timeout=10,
        )
    elif tool == "grafana_panel_query_range":
        if not isinstance(params, dict):
            raise RuntimeError("Grafana panel query requires params")
        uid = str(params.get("uid", "")).strip()
        panel_id = int(params.get("panel_id") or 0)
        panel_title = str(params.get("panel_title") or "").strip().lower()
        start_ms = int(params.get("from") or 0)
        end_ms = int(params.get("to") or 0)
        interval_ms = int(params.get("intervalMs") or 60000)
        vars_map = params.get("vars") if isinstance(params.get("vars"), dict) else {}
        if not uid or (not panel_id and not panel_title):
            raise RuntimeError("Grafana panel query requires uid and panel_id")
        dash_resp = requests.get(
            grafana_url + f"/api/dashboards/uid/{uid}",
            headers=headers,
            timeout=10,
        )
        if dash_resp.status_code >= 400:
            raise RuntimeError(f"Grafana dashboard fetch failed: {dash_resp.status_code} {dash_resp.text}")
        dash = dash_resp.json() or {}
        dashboard = dash.get("dashboard") or {}
        panels = dashboard.get("panels") or []
        rows = dashboard.get("rows") or []
        if not isinstance(vars_map, dict):
            vars_map = {}
        def _iter_panels(items):
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                yield item
                if item.get("type") == "row":
                    for child in item.get("panels") or []:
                        if isinstance(child, dict):
                            yield child

        flat_panels = list(_iter_panels(panels))
        for row in rows:
            if isinstance(row, dict):
                flat_panels.extend([p for p in row.get("panels") or [] if isinstance(p, dict)])

        panel = None
        if panel_id:
            panel = next((p for p in flat_panels if p.get("id") == panel_id), None)
        if not panel and panel_title:
            panel = next(
                (p for p in flat_panels if str(p.get("title", "")).strip().lower() == panel_title),
                None,
            )
        if not panel:
            missing = f"id {panel_id}" if panel_id else f"title '{panel_title}'"
            raise RuntimeError(f"Grafana panel {missing} not found")
        targets = panel.get("targets") or []
        if not isinstance(targets, list) or not targets:
            lib_uid = None
            lib = panel.get("libraryPanel")
            if isinstance(lib, dict):
                lib_uid = lib.get("uid")
            lib_uid = lib_uid or panel.get("libraryPanelUid")
            if lib_uid:
                lib_uid = str(lib_uid).strip()
            if lib_uid:
                lib_resp = requests.get(
                    grafana_url + f"/api/library-elements/{lib_uid}",
                    headers=headers,
                    timeout=10,
                )
                if lib_resp.status_code == 404:
                    lib_resp = requests.get(
                        grafana_url + f"/api/library-elements/uid/{lib_uid}",
                        headers=headers,
                        timeout=10,
                    )
                if lib_resp.status_code < 400:
                    lib_payload = lib_resp.json() or {}
                    lib_model = lib_payload.get("model") or {}
                    lib_targets = lib_model.get("targets") or []
                    if isinstance(lib_targets, list) and lib_targets:
                        targets = lib_targets
        if (not isinstance(targets, list) or not targets) and isinstance(params.get("targets"), list):
            targets = params.get("targets") or []
        if not isinstance(targets, list) or not targets:
            raise RuntimeError("Grafana panel has no targets")
        # Resolve datasource uid for each target
        ds_info = None
        ds_resp = requests.get(grafana_url + "/api/datasources", headers=headers, timeout=10)
        if ds_resp.status_code < 400:
            ds_list = ds_resp.json() or []
            default_ds = next((d for d in ds_list if d.get("isDefault")), None) or (ds_list[0] if ds_list else None)
            if isinstance(default_ds, dict) and default_ds.get("uid"):
                ds_info = {"uid": default_ds.get("uid"), "type": default_ds.get("type", "prometheus")}
        series = []
        for t in targets:
            if not isinstance(t, dict):
                continue
            expr = str(t.get("expr") or t.get("query") or "").strip()
            if not expr:
                continue
            for k, v in (vars_map or {}).items():
                expr = expr.replace(f"${k}", str(v))
                expr = expr.replace(f"${{{k}}}", str(v))
            raw_expr = expr
            expr = re.sub(r'tidb_cluster\s*=\s*"(?:[^"\\]|\\.)*"', 'tidb_cluster="tidb-cluster-basic"', expr)
            expr = re.sub(r'\s*,\s*k8s_cluster\s*=\s*"(?:[^"\\]|\\.)*"', "", expr)
            expr = re.sub(r'k8s_cluster\s*=\s*"(?:[^"\\]|\\.)*"\s*,\s*', "", expr)
            expr = re.sub(r'instance\s*=~\s*"(?:[^"\\]|\\.)*"', 'instance=~".*"', expr)
            if expr != raw_expr:
                logger.info("Grafana panel query_range expr_rewrite raw=%s rewritten=%s", raw_expr, expr)
            ds_uid = None
            if isinstance(t.get("datasource"), dict):
                ds_uid = t["datasource"].get("uid")
            ds_uid = ds_uid or (ds_info.get("uid") if ds_info else None)
            if not ds_uid:
                continue
            step = max(1, int(interval_ms / 1000))
            logger.info(
                "Grafana panel query_range expr=%s panel_id=%s ds_uid=%s start=%s end=%s step=%s",
                expr,
                panel_id,
                ds_uid,
                int(start_ms / 1000),
                int(end_ms / 1000),
                step,
            )
            proxy_url = f"{grafana_url}/api/datasources/proxy/uid/{ds_uid}/api/v1/query_range"
            r = requests.get(
                proxy_url,
                headers=headers,
                params={
                    "query": expr,
                    "start": start_ms / 1000,
                    "end": end_ms / 1000,
                    "step": step,
                },
                timeout=10,
            )
            if r.status_code >= 400:
                raise RuntimeError(f"Grafana query_range failed: {r.status_code} {r.text}")
            try:
                series.append(r.json())
            except Exception:
                series.append({"raw": r.text})
        return {"panel": {"id": panel_id, "title": panel.get("title", "")}, "series": series}
    elif tool in {"grafana_query_range", "grafana_query"}:
        queries = params.get("queries") if isinstance(params, dict) else None
        ds_uid = None
        if isinstance(queries, list):
            for q in queries:
                if isinstance(q, dict):
                    ds = q.get("datasource") or {}
                    ds_uid = ds.get("uid") or ds_uid
                    break
        # If we have a Prometheus datasource uid, use Grafana proxy to avoid ds/query schema issues.
        if ds_uid:
            expr = ""
            if isinstance(queries, list) and queries and isinstance(queries[0], dict):
                expr = str(queries[0].get("expr", "") or "")
            interval_ms = 60000
            if isinstance(params, dict):
                interval_ms = int(params.get("intervalMs") or interval_ms)
            start_ms = int(params.get("from") or 0)
            end_ms = int(params.get("to") or 0)
            if not start_ms or not end_ms:
                if isinstance(params.get("range"), dict):
                    start_ms = int(params["range"].get("from") or start_ms)
                    end_ms = int(params["range"].get("to") or end_ms)
            step = max(1, int(interval_ms / 1000))
            logger.info(
                "Grafana query_range expr=%s ds_uid=%s start=%s end=%s step=%s",
                expr,
                ds_uid,
                int(start_ms / 1000),
                int(end_ms / 1000),
                step,
            )
            proxy_url = f"{grafana_url}/api/datasources/proxy/uid/{ds_uid}/api/v1/query_range"
            resp = requests.get(
                proxy_url,
                headers=headers,
                params={
                    "query": expr,
                    "start": start_ms / 1000,
                    "end": end_ms / 1000,
                    "step": step,
                },
                timeout=10,
            )
        else:
            resp = requests.post(
                grafana_url + "/api/ds/query",
                headers=headers,
                json=params,
                timeout=10,
            )
    else:
        raise RuntimeError(f"Grafana MCP tool '{tool}' not supported by managed host")

    if resp.status_code >= 400:
        logger.error("Grafana MCP response status=%s body=%s", resp.status_code, resp.text)
        raise RuntimeError(f"Grafana API error: {resp.status_code} {resp.text}")
    try:
        return resp.json()
    except Exception:
        return resp.text

