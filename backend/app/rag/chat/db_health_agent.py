"""
Agent loop for database health / slow-query / Prometheus RCA.
Replaces the fixed pipeline: the agent chooses which tools to call and synthesizes the answer.
"""
import json
import logging
import re
from typing import Any, Optional

from app.mcp.client import run_mcp_db_query
from app.rag.chat.slow_query_db import (
    normalize_rows,
    parse_time_window_from_question,
)
from app.rag.chat.slow_query_prometheus import (
    build_prometheus_tidb_metrics_analysis,
    build_rca_summary_from_metrics,
)
from app.site_settings import SiteSetting

logger = logging.getLogger(__name__)

DB_HEALTH_AGENT_SYSTEM = """You are a TiDB database health and performance assistant. You have access to tools to run SQL queries and fetch Prometheus metrics. Use them to answer the user's question and provide RCA (root cause analysis) or recommendations.

Available tools:
- parse_time_window: Input: {"question": "user question"}. Returns {"start_ts": "...", "end_ts": "..."} (UTC) or {"error": "..."}. Use this to get a time range from phrases like "last 30 minutes" or "2026-01-14 16:00:00 to 2026-01-14 17:00:00".
- list_hosts: Input: {}. Returns available DB and Prometheus host names. Use one of these as host_name when calling other tools.
- run_sql: Input: {"sql": "SELECT ...", "host_name": "optional host name"}. Runs a SQL query via MCP. Use for CLUSTER_SLOW_QUERY, statement summary, or other TiDB queries.
- get_prometheus_metrics_and_rca: Input: {"start_ts": "YYYY-MM-DD HH:MM:SS", "end_ts": "YYYY-MM-DD HH:MM:SS", "user_question": "optional", "prometheus_host": "optional"}. Fetches Prometheus metrics and RCA summary for the time window.

Respond in this exact format. Use one of the tools or give the final answer.
Thought: <your reasoning>
Action: <tool name>
Action Input: <JSON object with tool parameters>
Observation: (you will receive this after the tool runs)

When you have enough information to answer, respond with:
Thought: I have enough to answer.
Final Answer: <your full response to the user, including metrics summary, RCA, and recommendations>
"""


def _run_parse_time_window(question: str) -> str:
    out = parse_time_window_from_question(question)
    return json.dumps(out, ensure_ascii=False)


def _run_list_hosts() -> str:
    SiteSetting.update_db_cache()
    ws = getattr(SiteSetting, "mcp_hosts", None) or []
    managed = getattr(SiteSetting, "managed_mcp_agents", None) or []
    prometheus = getattr(SiteSetting, "prometheus_hosts", None) or []
    db_names = set()
    for it in ws:
        name = str((it or {}).get("text", "")).strip()
        href = str((it or {}).get("href", "")).strip()
        if name and href and (href.startswith("ws://") or href.startswith("wss://") or href.startswith("managed://")):
            db_names.add(name)
    for it in managed:
        name = str((it or {}).get("name", "")).strip()
        if name:
            db_names.add(name)
    prom_names = [str((it or {}).get("name", "")).strip() for it in prometheus if (it or {}).get("name")]
    return json.dumps({
        "db_hosts": sorted(db_names),
        "prometheus_hosts": prom_names,
    }, ensure_ascii=False)


def _run_sql(chat_flow: Any, sql: str, host_name: Optional[str] = None) -> str:
    if not sql or not sql.strip():
        return json.dumps({"error": "sql is required"})
    try:
        SiteSetting.update_db_cache()
        managed = getattr(SiteSetting, "managed_mcp_agents", None) or []
        managed_names = {str((it or {}).get("name", "")).strip().lower() for it in managed if it}
        ws_list = getattr(SiteSetting, "mcp_hosts", None) or []
        ws_names = {
            str((it or {}).get("text", "")).strip().lower()
            for it in ws_list
            if it and (href := str((it or {}).get("href", "")).strip())
            and (href.startswith("ws://") or href.startswith("wss://") or href.startswith("managed://"))
        }
        if host_name and host_name.lower() in managed_names and host_name.lower() not in ws_names:
            from app.mcp.managed import run_managed_mcp_db_query
            result = run_managed_mcp_db_query(host_name, sql)
        else:
            result = run_mcp_db_query(sql, host_name=host_name)
        rows = normalize_rows(result, logger=logger)
        if isinstance(rows, list) and rows:
            return json.dumps({"row_count": len(rows), "rows": rows[:50]}, default=str, ensure_ascii=False)
        return json.dumps({"row_count": 0, "raw": str(result)[:2000]}, ensure_ascii=False)
    except Exception as e:
        logger.exception("run_sql failed: %s", e)
        return json.dumps({"error": str(e)})


def _run_get_prometheus_metrics_and_rca(
    chat_flow: Any,
    start_ts: str,
    end_ts: str,
    user_question: Optional[str] = None,
    prometheus_host: Optional[str] = None,
) -> str:
    if not start_ts or not end_ts:
        return json.dumps({"error": "start_ts and end_ts are required (YYYY-MM-DD HH:MM:SS UTC)"})
    try:
        metrics_text = build_prometheus_tidb_metrics_analysis(
            start_ts,
            end_ts,
            prometheus_host,
            logger,
            cluster_hint=None,
            session=chat_flow.db_session,
            user_question=user_question,
        )
        rca_summary = build_rca_summary_from_metrics(metrics_text, user_question)
        return json.dumps({
            "metrics_text": metrics_text,
            "rca_summary": rca_summary,
        }, ensure_ascii=False)
    except Exception as e:
        logger.exception("get_prometheus_metrics_and_rca failed: %s", e)
        return json.dumps({"error": str(e)})


def _execute_tool(name: str, action_input: dict, chat_flow: Any) -> str:
    if name == "parse_time_window":
        return _run_parse_time_window(str((action_input or {}).get("question", "")))
    if name == "list_hosts":
        return _run_list_hosts()
    if name == "run_sql":
        return _run_sql(
            chat_flow,
            str((action_input or {}).get("sql", "")),
            action_input.get("host_name"),
        )
    if name == "get_prometheus_metrics_and_rca":
        return _run_get_prometheus_metrics_and_rca(
            chat_flow,
            str((action_input or {}).get("start_ts", "")),
            str((action_input or {}).get("end_ts", "")),
            action_input.get("user_question"),
            action_input.get("prometheus_host"),
        )
    return json.dumps({"error": f"Unknown tool: {name}"})


def _parse_agent_response(text: str) -> tuple[Optional[str], Optional[dict], Optional[str]]:
    """Returns (thought, action_and_input, final_answer)."""
    thought = None
    action = None
    action_input = None
    final_answer = None
    if "Final Answer:" in text:
        idx = text.find("Final Answer:")
        final_answer = text[idx + len("Final Answer:"):].strip()
        text = text[:idx]
    if "Action:" in text:
        m = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
        if m:
            action = m.group(1).strip()
        rest = text
        if "Thought:" in text:
            thought = text.split("Action:")[0].replace("Thought:", "").strip()
        if "Action Input:" in text:
            inp = re.search(r"Action Input:\s*(\{[\s\S]*?\})(?=\s*Observation:|\s*$)", text)
            if inp:
                try:
                    action_input = json.loads(inp.group(1).strip())
                except json.JSONDecodeError:
                    action_input = {"raw": inp.group(1).strip()}
    return thought, ({"action": action, "action_input": action_input} if action else None), final_answer


def run_db_health_agent_loop(
    chat_flow: Any,
    user_question: str,
    *,
    max_steps: int = 6,
    max_response_chars: int = 55000,
) -> str:
    """
    Run the ReAct-style agent loop for database health / slow-query / Prometheus.
    Returns the final answer string.
    """
    llm = getattr(chat_flow, "_llm", None)
    if not llm:
        return "Agent error: no LLM configured."
    conversation = f"User question: {user_question}\n\n"
    for step in range(max_steps):
        prompt = (
            DB_HEALTH_AGENT_SYSTEM
            + "\n\n"
            + conversation
            + "\n\nRespond with Thought, then either Action + Action Input, or Final Answer."
        )
        try:
            response = str(llm.predict(prompt)).strip()
        except Exception as e:
            logger.exception("Agent LLM step failed: %s", e)
            return f"Agent error: {e}"
        thought, action_info, final_answer = _parse_agent_response(response)
        if final_answer:
            if len(final_answer) > max_response_chars:
                final_answer = final_answer[:max_response_chars] + "\n\n[truncated]"
            return final_answer
        if not action_info or not action_info.get("action"):
            conversation += f"Assistant: {response}\n\nObservation: No valid Action or Final Answer in the response. Please use a tool or provide Final Answer.\n\n"
            continue
        action_name = action_info.get("action", "").strip()
        action_input = action_info.get("action_input") or {}
        if not isinstance(action_input, dict):
            action_input = {"raw": str(action_input)}
        try:
            observation = _execute_tool(action_name, action_input, chat_flow)
        except Exception as e:
            observation = json.dumps({"error": str(e)})
        if len(observation) > 8000:
            observation = observation[:8000] + "\n...[truncated]"
        conversation += f"Assistant: {response}\n\nObservation: {observation}\n\n"
    return (
        "Reached maximum agent steps. Here is what was gathered so far.\n\n"
        + conversation[-12000:]
        + "\n\nPlease ask again with a more specific question or time window."
    )


def should_use_db_health_agent(user_question: str) -> bool:
    """True if the user question should be handled by the DB health agent instead of the fixed pipeline."""
    if not user_question or not isinstance(user_question, str):
        return False
    q = user_question.strip()
    if not q:
        return False
    trigger = re.search(
        r"\b(slow\s+queries?|CLUSTER_SLOW_QUERY|rocksdb_key_skipped_count|analy(?:s|z)e|summary|prometheus|metrics?|monitoring|health|rca|root\s+cause)\b",
        q,
        flags=re.IGNORECASE,
    )
    return bool(trigger)
