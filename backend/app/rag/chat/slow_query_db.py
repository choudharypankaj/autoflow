"""
Helpers for the DB health agent: time-window parsing and MCP result normalization.
All slow-query / metrics flows are driven by the agent in db_health_agent.py.
"""
import ast
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any, Optional


def coerce_text_payload(text: str) -> Any:
    if "TextContent" in text and "text=" in text:
        try:
            start = text.index("text=") + len("text=")
            snippet = text[start:].lstrip()
            if snippet and snippet[0] in ("'", '"'):
                quote = snippet[0]
                i = 1
                escaped = False
                while i < len(snippet):
                    ch = snippet[i]
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == quote:
                        break
                    i += 1
                raw_text = snippet[: i + 1]
                extracted = ast.literal_eval(raw_text)
                return coerce_text_payload(str(extracted))
        except Exception:
            pass
    for pattern, wrap in [
        (r"text='((?:\\'|[^'])*?)'", "'"),
        (r'text="((?:\\"|[^"])*?)"', '"'),
    ]:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            try:
                literal = f"{wrap}{match.group(1)}{wrap}"
                extracted = ast.literal_eval(literal)
                return coerce_text_payload(str(extracted))
            except Exception:
                continue
    if "meta=None content=" in text and "{" in text:
        text = text[text.find("{") :]
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        pass
    for opener, closer in [("[", "]"), ("{", "}")]:
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            chunk = text[start : end + 1]
            try:
                return json.loads(chunk)
            except Exception:
                try:
                    return ast.literal_eval(chunk)
                except Exception:
                    continue
    return text


def parse_mcp_text_result(result: Any, logger: logging.Logger | None = None) -> Any:
    content = None
    if isinstance(result, dict):
        content = result.get("content")
    else:
        content = getattr(result, "content", None)
    if not isinstance(content, list):
        return result
    parsed_items = []
    for item in content:
        text = None
        if isinstance(item, dict):
            text = item.get("text")
        else:
            text = getattr(item, "text", None)
        if not text:
            continue
        parsed = coerce_text_payload(text)
        if logger:
            logger.info("MCP text parsed type=%s", type(parsed).__name__)
        parsed_items.append(parsed)
    if len(parsed_items) == 1:
        return parsed_items[0]
    if parsed_items:
        return parsed_items
    return result


def normalize_rows(result: Any, logger: logging.Logger | None = None) -> list:
    parsed = parse_mcp_text_result(result, logger=logger)
    if isinstance(parsed, str):
        parsed = coerce_text_payload(parsed)
    if logger:
        logger.info(
            "MCP normalize_rows type=%s preview=%s",
            type(parsed).__name__,
            str(parsed)[:200].replace("\n", "\\n"),
        )
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("rows", "data", "result"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
        if any(k in parsed for k in ("digest", "sample_query", "INSTANCE", "Time", "query_time")):
            return [parsed]
    return []


def parse_time_window_from_question(user_question: str) -> dict:
    """
    Parse a UTC time window from the user question. Used by the DB health agent.
    Returns {"start_ts": str, "end_ts": str} or {"error": str} if no window found.
    Supports: "last N minutes/hours", or explicit "YYYY-MM-DD HH:MM:SS" start/end.
    """
    if not user_question or not isinstance(user_question, str):
        return {"error": "No question provided."}
    q = user_question.strip()
    ts_pattern = r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"
    matches = re.findall(ts_pattern, q)
    if len(matches) >= 2:
        return {"start_ts": matches[0], "end_ts": matches[1]}
    rel = re.search(r"\blast\s+(\d+)\s+hour(s)?\b", q, flags=re.IGNORECASE)
    if rel:
        hours = int(rel.group(1))
        end_dt = datetime.now(UTC)
        start_dt = end_dt - timedelta(hours=hours)
        return {
            "start_ts": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_ts": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        }
    rel = re.search(r"\blast\s+(\d+)\s+(?:min|mins|minute|minutes)\b", q, flags=re.IGNORECASE)
    if rel:
        minutes = int(rel.group(1))
        end_dt = datetime.now(UTC)
        start_dt = end_dt - timedelta(minutes=minutes)
        return {
            "start_ts": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end_ts": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        }
    return {
        "error": "Could not parse time window. Provide 'last N minutes/hours' or explicit start and end in 'YYYY-MM-DD HH:MM:SS' (UTC).",
    }
