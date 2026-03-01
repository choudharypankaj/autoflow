"""
Agent loop for database health / Prometheus-only analysis.
Uses only Prometheus metrics; the chat engine's configured LLM runs a ReAct loop with tools.
"""
import json
import logging
import re
from typing import Any, Optional

from llama_index.core import PromptTemplate

from app.rag.chat.slow_query_db import parse_time_window_from_question
from app.rag.chat.slow_query_prometheus import (
    get_prometheus_metadata,
    list_prometheus_metric_names,
    run_promql_range,
)
from app.site_settings import SiteSetting

logger = logging.getLogger(__name__)

DB_HEALTH_AGENT_SYSTEM = """You are a TiDB cluster health and performance assistant. You perform analysis using only Prometheus metrics. Do not run SQL or slow-query tools—use only the Prometheus-related tools below.

Available tools:
- parse_time_window: Input: {"question": "user question"}. Returns {"start_ts": "...", "end_ts": "..."} (UTC) or {"error": "..."}. Use for "last 6 hours", "last 30 minutes", or "YYYY-MM-DD HH:MM:SS" start/end.
- list_hosts: Input: {}. Returns available Prometheus host names (use as prometheus_host in other tools).
- list_prometheus_metric_names: Input: {"prometheus_host": "optional"}. Returns list of metric names from Prometheus. Use to discover what to query.
- get_prometheus_metadata: Input: {"prometheus_host": "optional"}. Returns metadata (type, help) for each metric. Use to build correct PromQL (e.g. rate() for counters, histogram_quantile for histograms).
- run_promql: Input: {"prometheus_host": "optional", "query": "PromQL expression", "start_ts": "YYYY-MM-DD HH:MM:SS", "end_ts": "YYYY-MM-DD HH:MM:SS"}. Runs one range query and returns a text summary. Call multiple times for different metrics.

Respond in this exact format. Use tools then give the final answer.
Thought: <your reasoning>
Action: <tool name>
Action Input: <JSON object with tool parameters>
Observation: (you will receive this after the tool runs)

When you have enough information:
Thought: I have enough to answer.
Final Answer: <your full response to the user. Synthesize the Prometheus metrics you gathered and provide root cause analysis (RCA), recommendations, and a clear summary.>
"""


def _run_parse_time_window(question: str) -> str:
    out = parse_time_window_from_question(question)
    return json.dumps(out, ensure_ascii=False)


def _run_list_hosts() -> str:
    SiteSetting.update_db_cache()
    prometheus = getattr(SiteSetting, "prometheus_hosts", None) or []
    prom_names = [str((it or {}).get("name", "")).strip() for it in prometheus if (it or {}).get("name")]
    return json.dumps({"prometheus_hosts": prom_names}, ensure_ascii=False)


def _run_get_prometheus_metadata(prometheus_host: Optional[str] = None) -> str:
    try:
        meta = get_prometheus_metadata(prometheus_host)
        if not meta:
            return json.dumps({"error": "No Prometheus host or no metadata."})
        out = {name: info for name, info in list(meta.items())[:200]}
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:
        logger.exception("get_prometheus_metadata failed: %s", e)
        return json.dumps({"error": str(e)})


def _run_list_prometheus_metric_names(prometheus_host: Optional[str] = None) -> str:
    try:
        names = list_prometheus_metric_names(prometheus_host)
        if not names:
            return json.dumps({"error": "No Prometheus host or no metric names."})
        return json.dumps({"metric_names": names[:500]}, ensure_ascii=False)
    except Exception as e:
        logger.exception("list_prometheus_metric_names failed: %s", e)
        return json.dumps({"error": str(e)})


def _run_run_promql(
    chat_flow: Any,
    prometheus_host: Optional[str],
    query: str,
    start_ts: str,
    end_ts: str,
) -> str:
    if not query or not query.strip():
        return json.dumps({"error": "query is required"})
    if not start_ts or not end_ts:
        return json.dumps({"error": "start_ts and end_ts are required (YYYY-MM-DD HH:MM:SS UTC)"})
    try:
        text = run_promql_range(prometheus_host, query.strip(), start_ts, end_ts, logger=logger)
        return json.dumps({"summary": text}, ensure_ascii=False)
    except Exception as e:
        logger.exception("run_promql failed: %s", e)
        return json.dumps({"error": str(e)})


def _execute_tool(name: str, action_input: dict, chat_flow: Any) -> str:
    if name == "parse_time_window":
        return _run_parse_time_window(str((action_input or {}).get("question", "")))
    if name == "list_hosts":
        return _run_list_hosts()
    if name == "list_prometheus_metric_names":
        return _run_list_prometheus_metric_names(action_input.get("prometheus_host"))
    if name == "get_prometheus_metadata":
        return _run_get_prometheus_metadata(action_input.get("prometheus_host"))
    if name == "run_promql":
        return _run_run_promql(
            chat_flow,
            action_input.get("prometheus_host"),
            str((action_input or {}).get("query", "")),
            str((action_input or {}).get("start_ts", "")),
            str((action_input or {}).get("end_ts", "")),
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
    Run the ReAct-style DB health agent using the chat engine's configured LLM.
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
            template = PromptTemplate(prompt)
            response = str(llm.predict(template)).strip()
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
