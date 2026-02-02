import ast
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

from llama_index.core.base.llms.types import MessageRole
from llama_index.core.prompts.rich import RichPromptTemplate

from app.mcp.client import run_mcp_db_query
from app.rag.chat.slow_query_grafana import (
    build_grafana_duration_analysis,
    build_grafana_tidb_metrics_analysis,
)
from app.repositories import chat_repo
from app.site_settings import SiteSetting


def coerce_meta(meta_value: Any) -> Optional[dict]:
    if isinstance(meta_value, dict):
        return meta_value
    if isinstance(meta_value, str):
        try:
            parsed = json.loads(meta_value)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def extract_tables_from_sql(sql_text: str) -> list[str]:
    """
    Best-effort extractor for table names from SQL using FROM/JOIN patterns.
    Returns lowercased table identifiers without schema/quotes.
    """
    if not isinstance(sql_text, str):
        return []
    matches = re.findall(r"\b(?:from|join|into|update|delete\s+from)\s+([`\"\w\.]+)", sql_text, flags=re.IGNORECASE)
    tables: list[str] = []
    for m in matches:
        t = m.strip('`"').split(".")[-1].lower()
        if t:
            tables.append(t)
    return tables


def build_statement_summary_query(start_time: str, end_time: str) -> str:
    return (
        "SELECT "
        "ANY_VALUE(digest_text) AS agg_digest_text, "
        "ANY_VALUE(digest) AS agg_digest, "
        "SUM(exec_count) AS agg_exec_count, "
        "SUM(sum_latency) AS agg_sum_latency, "
        "MAX(max_latency) AS agg_max_latency, "
        "MIN(min_latency) AS agg_min_latency, "
        "CAST(SUM(exec_count * avg_latency) / SUM(exec_count) AS SIGNED) AS agg_avg_latency, "
        "ANY_VALUE(schema_name) AS agg_schema_name, "
        "COUNT(DISTINCT plan_digest) AS agg_plan_count "
        "FROM `INFORMATION_SCHEMA`.`CLUSTER_STATEMENTS_SUMMARY_HISTORY` "
        f"WHERE summary_begin_time <= '{end_time}' "
        f"AND summary_end_time >= '{start_time}' "
        "GROUP BY schema_name, digest "
        "ORDER BY agg_plan_count DESC "
        "LIMIT 20"
    )


def latency_to_seconds(value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    if num >= 1e9:
        return num / 1e9
    if num >= 1e6:
        return num / 1e6
    if num >= 1e3:
        return num / 1e3
    return num


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


def clean_cell(value: Any) -> str:
    text = str(value) if value is not None else ""
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    text = text.replace("|", r"\|")
    return " ".join(text.split()) or "-"


def rows_to_markdown(result: Any, columns: list[str], *, logger: logging.Logger | None = None) -> str:
    rows = normalize_rows(result, logger=logger)
    if not rows:
        return "(no data)"
    lines = []
    header = " | ".join(columns)
    sep = " | ".join(["---"] * len(columns))
    lines.append(header)
    lines.append(sep)
    for r in rows[:10]:
        if isinstance(r, dict):
            values = [clean_cell(r.get(c, "")) for c in columns]
        elif isinstance(r, (list, tuple)):
            values = [clean_cell(x) for x in r[: len(columns)]]
        else:
            values = [clean_cell(r)]
        lines.append(" | ".join(values))
    return "\n".join(lines)


def build_summary_from_rows(raw_rows: list) -> tuple[list[dict], list[dict], list[dict]]:
    digest_agg: dict[str, dict] = {}
    instance_agg: dict[str, dict] = {}
    for r in raw_rows:
        if not isinstance(r, dict):
            continue
        digest = str(r.get("digest") or "")
        plan_digest = r.get("plan_digest") or "-"
        q = r.get("query") or ""
        inst = r.get("INSTANCE") or ""
        qt = float(r.get("query_time") or 0.0)
        skipped = float(r.get("rocksdb_key_skipped_count") or 0.0)
        if digest:
            agg = digest_agg.setdefault(
                digest,
                {
                    "digest": digest,
                    "sample_query": str(q)[:200],
                    "plan_digest": plan_digest,
                    "plan": str(r.get("plan") or "")[:800],
                    "exec_count": 0,
                    "avg_s": 0.0,
                    "max_s": 0.0,
                    "skipped_sum": 0.0,
                    "_sum_s": 0.0,
                },
            )
            agg["exec_count"] += 1
            agg["_sum_s"] += qt
            agg["max_s"] = max(agg["max_s"], qt)
            agg["skipped_sum"] += skipped
            agg["avg_s"] = round(agg["_sum_s"] / agg["exec_count"], 3)
        if inst:
            inst_agg = instance_agg.setdefault(
                str(inst),
                {"INSTANCE": inst, "exec_count": 0, "avg_s": 0.0, "total_s": 0.0, "_sum_s": 0.0},
            )
            inst_agg["exec_count"] += 1
            inst_agg["_sum_s"] += qt
            inst_agg["total_s"] = round(inst_agg["_sum_s"], 3)
            inst_agg["avg_s"] = round(inst_agg["_sum_s"] / inst_agg["exec_count"], 3)
    digest_rows = sorted(digest_agg.values(), key=lambda x: x["skipped_sum"], reverse=True)[:10]
    instance_rows = sorted(instance_agg.values(), key=lambda x: x["total_s"], reverse=True)[:10]
    tables_agg: dict[str, dict] = {}
    for r in digest_rows:
        if not isinstance(r, dict):
            continue
        sample = r.get("sample_query") or ""
        exec_count = float(r.get("exec_count") or 0)
        avg_s = float(r.get("avg_s") or 0.0)
        total_s = exec_count * avg_s
        for t in extract_tables_from_sql(str(sample)):
            agg = tables_agg.setdefault(t, {"table": t, "exec_count": 0, "total_s": 0.0})
            agg["exec_count"] += int(exec_count)
            agg["total_s"] += total_s
    tables_rows = sorted(tables_agg.values(), key=lambda x: x["total_s"], reverse=True)[:10]
    return digest_rows, instance_rows, tables_rows


def build_ai_recommendations(
    raw_rows: list,
    llm: Any,
    logger: logging.Logger,
) -> tuple[str, str, str]:
    ai_recommendations_text = ""
    ai_examples_json = ""
    try:
        examples = []
        for r in raw_rows:
            if not isinstance(r, dict):
                continue
            try:
                skipped = float(r.get("rocksdb_key_skipped_count") or 0.0)
            except (TypeError, ValueError):
                skipped = 0.0
            if skipped <= 0:
                continue
            plan_text = str(r.get("plan") or "").strip()
            if not plan_text:
                continue
            examples.append({"plan": plan_text})
            if len(examples) >= 3:
                break
        if examples:
            plan_only = [{"plan": str(item.get("plan") or "")} for item in examples if item.get("plan")]
            ai_examples_json = json.dumps(plan_only, ensure_ascii=False)
            logger.info(
                "AI recommendation input plan_count=%d plan_chars=%d plans=%s",
                len(plan_only),
                sum(len(item.get("plan") or "") for item in plan_only),
                plan_only,
            )
            plan_previews = [(item.get("plan") or "") for item in plan_only]
            logger.info("AI recommendation input plan_previews=%s", plan_previews)
            prompt_text = (
                "You are a TiDB performance expert. Analyze the execution plans "
                "to suggest concrete index or query changes.\n"
                "For each plan, output in this exact format:\n"
                "Plan: <short summary of plan>\n"
                "Recommendation: <specific action>\n"
                "Reason: <why needed>\n"
                "Details: <tables/columns/index names; if unknown say 'unknown'>\n"
                "---\n"
                "Do not ask for more data. Use only the provided plans.\n\n"
                f"Plans (JSON): {ai_examples_json}\n"
            )
            prompt = RichPromptTemplate(prompt_text)
            logger.info("AI recommendation prompt=%s", prompt_text)
            ai_recommendations_text = str(llm.predict(prompt)).strip()
            logger.info(
                "AI recommendation response_len=%d response=%s",
                len(ai_recommendations_text),
                ai_recommendations_text,
            )
            if re.search(r"please\s+provide.*plans?|don't\s+see\s+any\s+actual\s+data", ai_recommendations_text, flags=re.IGNORECASE):
                retry_prompt = RichPromptTemplate(
                    "You have all required execution plans below. Do not ask for more data. "
                    "Provide recommendations in the required format.\n\n"
                    f"Plans (JSON): {ai_examples_json}\n"
                )
                ai_recommendations_text = str(llm.predict(retry_prompt)).strip()
                logger.info(
                    "AI recommendation retry_response_len=%d response=%s",
                    len(ai_recommendations_text),
                    ai_recommendations_text,
                )
            if re.search(r"please\s+provide.*plans?|don't\s+see\s+any\s+actual\s+data", ai_recommendations_text, flags=re.IGNORECASE):
                ai_recommendations_text = ""
    except Exception as e:
        logger.exception("AI recommendation error: %s", e)
        logger.exception("AI recommendation generation failed: %s", e)

    if ai_recommendations_text:
        return ai_examples_json or "[]", "AI status: success", ai_recommendations_text
    if not ai_examples_json:
        return "[]", "AI status: no plans available", "- AI analysis unavailable; no execution plans were returned in CLUSTER_SLOW_QUERY for this window."
    return ai_examples_json, "AI status: failed to generate recommendations", "- AI analysis unavailable; model did not return recommendations for the provided plans."


def build_rule_recommendations(digest_rows: list, stmt_rows: list) -> str:
    recommendations: list[dict] = []
    top_digest = digest_rows[0] if isinstance(digest_rows, list) and digest_rows else {}
    if isinstance(top_digest, dict):
        max_s = float(top_digest.get("max_s") or 0.0)
        skipped_sum = float(top_digest.get("skipped_sum") or 0.0)
        exec_count = int(top_digest.get("exec_count") or 0)
        rules = [
            {
                "score": 3 if max_s >= 2.0 else 2 if max_s >= 1.0 else 0,
                "text": "Investigate the slowest query; review execution plan and indexes.",
            },
            {
                "score": 3 if skipped_sum >= 50000 else 2 if skipped_sum >= 10000 else 0,
                "text": "High rocksdb_key_skipped_count; check for inefficient range scans or missing indexes.",
            },
            {
                "score": 2 if exec_count >= 100 else 1 if exec_count >= 50 else 0,
                "text": "High exec_count; consider caching, batching, or throttling repeated queries.",
            },
        ]
        recommendations = [r for r in rules if r["score"] > 0]
        recommendations.sort(key=lambda r: r["score"], reverse=True)
    if isinstance(stmt_rows, list) and stmt_rows:
        top_stmt = stmt_rows[0] if isinstance(stmt_rows[0], dict) else {}
        try:
            plan_count = int((top_stmt or {}).get("agg_plan_count") or 0)
        except (TypeError, ValueError):
            plan_count = 0
        max_latency_s = latency_to_seconds((top_stmt or {}).get("agg_max_latency"))
        avg_latency_s = latency_to_seconds((top_stmt or {}).get("agg_avg_latency"))
        sum_latency_s = latency_to_seconds((top_stmt or {}).get("agg_sum_latency"))
        if plan_count >= 5 and (max_latency_s >= 1.0 or avg_latency_s >= 0.5):
            recommendations.append({
                "score": 3,
                "text": "High plan count with elevated latency; review plan stability, update stats, and consider plan bindings.",
            })
        elif plan_count >= 5 and sum_latency_s > 0:
            recommendations.append({
                "score": 2,
                "text": "Multiple plans detected for this digest; check for plan cache instability and inconsistent parameter patterns.",
            })
        recommendations.sort(key=lambda r: r.get("score", 0), reverse=True)
    if not recommendations:
        return "- No obvious hotspots detected; consider widening the time window."
    return "\n".join(f"- {r['text']}" for r in recommendations)


def maybe_run_db_slow_query(
    chat_flow: Any,
    user_question: str,
    *,
    logger: logging.Logger,
    max_chars: int,
    json_safe: Any,
) -> Optional[str]:
    """
    Detects and runs the predefined slow query against TiDB via MCP when the user
    supplies UTC time window. Expected formats (UTC):
      - 2026-01-14 16:15:00 to 2026-01-14 16:47:00
      - start: 2026-01-14 16:15:00, end: 2026-01-14 16:47:00
    """
    # 0) Follow-up meta-only path: if user references last/summary/digest/instance/table,
    # try answering from the last assistant meta immediately, unless a time window is present.
    has_relative_window = bool(re.search(
        r"\blast\s+\d+\s+(?:min|mins|minute|minutes|hour|hours)\b",
        user_question,
        flags=re.IGNORECASE,
    ))
    if (
        re.search(r"\b(last|previous|summary|summarize|digest|instance|table)\b", user_question, flags=re.IGNORECASE)
        and not has_relative_window
    ):
        try:
            prior_messages = chat_repo.get_messages(chat_flow.db_session, chat_flow.db_chat_obj)
            meta = None
            for m in reversed(prior_messages):
                try:
                    if getattr(m, "role", "") == MessageRole.ASSISTANT.value:
                        candidate = coerce_meta(m.meta)
                        if not candidate:
                            continue
                        t = str(candidate.get("type", ""))
                        if t in {"slow_query_summary", "slow_query_rows"}:
                            meta = candidate
                            break
                except Exception:
                    continue
            if meta and meta.get("type") == "slow_query_summary":
                want_instance = bool(re.search(r"\binstance", user_question, flags=re.IGNORECASE))
                want_digest = bool(re.search(r"\bdigest", user_question, flags=re.IGNORECASE))
                want_table = bool(re.search(r"\btable", user_question, flags=re.IGNORECASE))
                want_sample = bool(re.search(r"\b(sample|examples?|queries?|execution)\b", user_question, flags=re.IGNORECASE))
                digests = meta.get("digests") or []
                instances = meta.get("instances") or []
                tables = meta.get("tables") or []
                chunks: list[str] = []
                if want_digest and isinstance(digests, list):
                    scored = []
                    for d in digests:
                        if isinstance(d, dict):
                            exec_count = float(d.get("exec_count") or 0)
                            avg_s = float(d.get("avg_s") or 0.0)
                            total_s = exec_count * avg_s
                            scored.append({
                                "digest": d.get("digest", ""),
                                "exec_count": int(exec_count),
                                "avg_s": avg_s,
                                "total_s": round(total_s, 3),
                            })
                    scored.sort(key=lambda x: x["total_s"], reverse=True)
                    chunks.append("Top digests by total time:\n\n" + rows_to_markdown(scored, ["digest", "exec_count", "avg_s", "total_s"]))
                if want_instance and isinstance(instances, list):
                    chunks.append("Instances (cached):\n\n" + rows_to_markdown(instances, ["INSTANCE", "exec_count", "avg_s", "total_s"]))
                if want_table and isinstance(tables, list):
                    chunks.append("Impacted tables (cached):\n\n" + rows_to_markdown(tables, ["table", "exec_count", "total_s"]))
                # Sample queries removed per UI preference
                if chunks:
                    text = "\n\n".join(chunks)
                    if len(text) > max_chars:
                        text = text[:max_chars] + "\n\n[truncated]"
                    chat_flow._cached_slow_query_meta = json_safe(meta)
                    return text
            if meta and meta.get("type") == "slow_query_rows":
                rows = meta.get("rows") or []
                want_instance = bool(re.search(r"\binstance", user_question, flags=re.IGNORECASE))
                want_digest = bool(re.search(r"\bdigest", user_question, flags=re.IGNORECASE))
                want_table = bool(re.search(r"\btable", user_question, flags=re.IGNORECASE))
                want_sample = bool(re.search(r"\b(sample|examples?|queries?|execution)\b", user_question, flags=re.IGNORECASE))
                digest_agg: dict[str, dict] = {}
                instance_agg: dict[str, dict] = {}
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    digest = str(r.get("digest") or "")
                    plan_digest = r.get("plan_digest") or "-"
                    plan_text = r.get("plan") or ""
                    q = r.get("query") or ""
                    inst = r.get("INSTANCE") or ""
                    qt = float(r.get("query_time") or 0.0)
                    skipped = float(r.get("rocksdb_key_skipped_count") or 0.0)
                    if digest:
                        agg = digest_agg.setdefault(
                            digest,
                            {
                                "digest": digest,
                                "sample_query": str(q)[:200],
                                "plan_digest": plan_digest,
                                "plan": str(plan_text)[:800],
                                "exec_count": 0,
                                "avg_s": 0.0,
                                "max_s": 0.0,
                                "skipped_sum": 0.0,
                                "_sum_s": 0.0,
                            },
                        )
                        agg["exec_count"] += 1
                        agg["_sum_s"] += qt
                        agg["max_s"] = max(agg["max_s"], qt)
                        agg["skipped_sum"] += skipped
                        agg["avg_s"] = round(agg["_sum_s"] / agg["exec_count"], 3)
                    if inst:
                        inst_agg = instance_agg.setdefault(
                            str(inst),
                            {"INSTANCE": inst, "exec_count": 0, "avg_s": 0.0, "total_s": 0.0, "_sum_s": 0.0},
                        )
                        inst_agg["exec_count"] += 1
                        inst_agg["_sum_s"] += qt
                        inst_agg["total_s"] = round(inst_agg["_sum_s"], 3)
                        inst_agg["avg_s"] = round(inst_agg["_sum_s"] / inst_agg["exec_count"], 3)
                digest_rows = sorted(digest_agg.values(), key=lambda x: x["skipped_sum"], reverse=True)[:10]
                instance_rows = sorted(instance_agg.values(), key=lambda x: x["total_s"], reverse=True)[:10]
                tables_agg: dict[str, dict] = {}
                for r in digest_rows:
                    if not isinstance(r, dict):
                        continue
                    sample = r.get("sample_query") or ""
                    exec_count = float(r.get("exec_count") or 0)
                    avg_s = float(r.get("avg_s") or 0.0)
                    total_s = exec_count * avg_s
                    for t in extract_tables_from_sql(str(sample)):
                        agg = tables_agg.setdefault(t, {"table": t, "exec_count": 0, "total_s": 0.0})
                        agg["exec_count"] += int(exec_count)
                        agg["total_s"] += total_s
                tables_rows = sorted(tables_agg.values(), key=lambda x: x["total_s"], reverse=True)[:10]
                chunks: list[str] = []
                if want_digest and digest_rows:
                    chunks.append("Top digests (cached rows):\n\n" + rows_to_markdown(digest_rows, ["digest", "exec_count", "avg_s", "max_s", "skipped_sum"]))
                if want_instance and instance_rows:
                    chunks.append("Instances (cached rows):\n\n" + rows_to_markdown(instance_rows, ["INSTANCE", "exec_count", "avg_s", "total_s"]))
                if want_table and tables_rows:
                    chunks.append("Impacted tables (cached rows):\n\n" + rows_to_markdown(tables_rows, ["table", "exec_count", "total_s"]))
                if want_sample:
                    samples = []
                    for r in rows[:5]:
                        if isinstance(r, dict):
                            sample = str(r.get("query") or "").strip()
                            if sample:
                                samples.append(sample[:200])
                    chunks.append("Sample queries (cached rows):\n\n" + ("\n".join(f"- {s}" for s in samples) or "(no data)"))
                if chunks:
                    text = "\n\n".join(chunks)
                    if len(text) > max_chars:
                        text = text[:max_chars] + "\n\n[truncated]"
                    chat_flow._cached_slow_query_meta = json_safe(meta)
                    return text
        except Exception:
            pass

    # Quick intent filter to avoid accidental triggers
    trigger_patterns = [
        r"\bslow\s+queries?\b",
        r"\bCLUSTER_SLOW_QUERY\b",
        r"\brocksdb_key_skipped_count\b",
    ]
    if not any(re.search(p, user_question, flags=re.IGNORECASE) for p in trigger_patterns):
        if not re.search(r"\b(analy(?:s|z)e|summary|slow)\b", user_question, flags=re.IGNORECASE):
            return None

    # Extract two UTC timestamps
    ts_pattern = r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"
    matches = re.findall(ts_pattern, user_question)
    if len(matches) < 2:
        rel = re.search(r"\blast\s+(\d+)\s+hour(s)?\b", user_question, flags=re.IGNORECASE)
        if rel:
            hours = int(rel.group(1))
            end_dt = datetime.now(UTC)
            start_dt = end_dt - timedelta(hours=hours)
            matches = [
                start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            ]
    if len(matches) < 2:
        rel = re.search(r"\blast\s+(\d+)\s+(?:min|mins|minute|minutes)\b", user_question, flags=re.IGNORECASE)
        if rel:
            minutes = int(rel.group(1))
            end_dt = datetime.now(UTC)
            start_dt = end_dt - timedelta(minutes=minutes)
            matches = [
                start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            ]
    if len(matches) < 2:
        force_fresh = bool(re.search(r"\b(fresh|re[-\s]?run|new\s+window|do\s+not\s+use\s+last|ignore\s+summary)\b", user_question, flags=re.IGNORECASE))
        if not force_fresh:
            try:
                prior_messages = chat_repo.get_messages(chat_flow.db_session, chat_flow.db_chat_obj)
                meta = None
                for m in reversed(prior_messages):
                    try:
                        if getattr(m, "role", "") == MessageRole.ASSISTANT.value:
                            candidate = coerce_meta(m.meta)
                            if not candidate:
                                continue
                            t = str(candidate.get("type", ""))
                            if t in {"slow_query_summary", "slow_query_rows"}:
                                meta = candidate
                                break
                    except Exception:
                        continue
                if meta and meta.get("type") == "slow_query_summary":
                    want_instance = bool(re.search(r"\binstance", user_question, flags=re.IGNORECASE))
                    want_digest = bool(re.search(r"\bdigest", user_question, flags=re.IGNORECASE))
                    want_table = bool(re.search(r"\btable", user_question, flags=re.IGNORECASE))
                    digests = meta.get("digests") or []
                    instances = meta.get("instances") or []
                    tables = meta.get("tables") or []
                    chunks: list[str] = []
                    if want_digest or (not want_instance and not want_table):
                        if isinstance(digests, list):
                            scored = []
                            for d in digests:
                                if isinstance(d, dict):
                                    exec_count = float(d.get("exec_count") or 0)
                                    avg_s = float(d.get("avg_s") or 0.0)
                                    total_s = exec_count * avg_s
                                    scored.append({
                                        "digest": d.get("digest", ""),
                                        "exec_count": int(exec_count),
                                        "avg_s": avg_s,
                                        "total_s": round(total_s, 3),
                                    })
                            scored.sort(key=lambda x: x["total_s"], reverse=True)
                            chunks.append("Top digests by total time (cached):\n\n" + rows_to_markdown(scored, ["digest", "exec_count", "avg_s", "total_s"]))
                    if want_instance and isinstance(instances, list):
                        chunks.append("Instances (cached):\n\n" + rows_to_markdown(instances, ["INSTANCE", "exec_count", "avg_s", "total_s"]))
                    if want_table and isinstance(tables, list):
                        chunks.append("Impacted tables (cached):\n\n" + rows_to_markdown(tables, ["table", "exec_count", "total_s"]))
                    if chunks:
                        text = "\n\n".join(chunks)
                        if len(text) > max_chars:
                            text = text[:max_chars] + "\n\n[truncated]"
                        chat_flow._cached_slow_query_meta = json_safe(meta)
                        return text
            except Exception:
                pass
        return (
            "Please provide UTC start and end times in the format "
            "'YYYY-MM-DD HH:MM:SS'. Example: start 2026-01-14 16:15:00, end 2026-01-14 16:47:00"
        )

    start_ts, end_ts = matches[0], matches[1]
    cluster_hint = None
    if re.search(r"\bprod(uction)?\s+cluster\b", user_question, flags=re.IGNORECASE):
        cluster_hint = "prod"

    host_name = chat_flow.mcp_host_name
    if not host_name:
        try:
            m = re.search(
                r"\b(?:for|using|on)\s+([A-Za-z0-9._-]+)\s+(?:mcp|db|database|cluster)\b",
                user_question,
                flags=re.IGNORECASE,
            )
            candidate = m.group(1).strip() if m else ""
            SiteSetting.update_db_cache()
            ws = getattr(SiteSetting, "mcp_hosts", None) or []
            managed = getattr(SiteSetting, "managed_mcp_agents", None) or []
            grafana_hosts = getattr(SiteSetting, "mcp_grafana_hosts", None) or []
            db_valid_names = set()
            for item in ws:
                try:
                    name = str(item.get("text", "")).strip()
                    href = str((item or {}).get("href", "")).strip()
                    if name and href and not href.startswith("managed-grafana://") and (
                        href.startswith("ws://") or href.startswith("wss://") or href.startswith("managed://")
                    ):
                        db_valid_names.add(name.lower())
                except Exception:
                    continue
            for item in managed:
                try:
                    name = str((item or {}).get("name", "")).strip()
                    if name:
                        db_valid_names.add(name.lower())
                except Exception:
                    continue
            grafana_names = {str((it or {}).get("name", "")).strip().lower() for it in grafana_hosts if it}
            if candidate:
                if candidate.lower() in grafana_names:
                    host_name = candidate
                elif candidate.lower() in db_valid_names:
                    host_name = candidate
            if not host_name and isinstance(managed, list) and len(managed) == 1:
                maybe_name = str((managed[0] or {}).get("name", "")).strip()
                if maybe_name:
                    host_name = maybe_name
        except Exception:
            pass

    grafana_host_name = None
    db_host_name = host_name
    db_host_ready = False
    default_mcp_ok = False
    try:
        SiteSetting.update_db_cache()
        grafana_hosts = getattr(SiteSetting, "mcp_grafana_hosts", None) or []
        grafana_names = {str((it or {}).get("name", "")).strip().lower() for it in grafana_hosts if it}
        ws_hosts = getattr(SiteSetting, "mcp_hosts", None) or []
        managed_agents = getattr(SiteSetting, "managed_mcp_agents", None) or []
        managed_names = {str((it or {}).get("name", "")).strip().lower() for it in managed_agents if it}
        default_mcp_url = str(getattr(SiteSetting, "mcp_host", "") or "").strip()
        default_mcp_ok = bool(
            default_mcp_url
            and not default_mcp_url.startswith("managed-grafana://")
            and (
                default_mcp_url.startswith("ws://")
                or default_mcp_url.startswith("wss://")
                or default_mcp_url.startswith("managed://")
            )
        )
        ws_names = {
            str((it or {}).get("text", "")).strip().lower()
            for it in ws_hosts
            if it
            and (href := str((it or {}).get("href", "")).strip())
            and not href.startswith("managed-grafana://")
            and (href.startswith("ws://") or href.startswith("wss://") or href.startswith("managed://"))
        }
        if host_name and host_name.lower() in grafana_names:
            grafana_host_name = host_name
            db_host_name = ""
        if not db_host_name:
            if managed_names:
                db_host_name = next(iter(managed_names))
            elif ws_names:
                db_host_name = next(iter(ws_names))
        if db_host_name:
            for it in ws_hosts:
                if str((it or {}).get("text", "")).strip().lower() == db_host_name.lower():
                    href = str((it or {}).get("href", "")).strip()
                    if href.startswith("managed-grafana://"):
                        logger.info("DB host_name points to Grafana MCP; clearing host_name=%s href=%s", db_host_name, href)
                        db_host_name = ""
                        break
        valid_db_host = False
        if db_host_name:
            if db_host_name.lower() in managed_names:
                valid_db_host = True
            elif db_host_name.lower() in ws_names:
                valid_db_host = True
        if not valid_db_host:
            if db_host_name:
                logger.info("DB MCP host invalid or unsupported; skipping DB queries host_name=%s", db_host_name)
            db_host_name = ""
            db_host_ready = bool(default_mcp_ok)
        else:
            db_host_ready = True
    except Exception:
        grafana_hosts = []
        db_host_ready = bool(db_host_name)
    if not grafana_host_name and grafana_hosts:
        grafana_host_name = str((grafana_hosts[0] or {}).get("name", "")).strip() or None
    display_host = db_host_name
    if not display_host:
        # Try to resolve default mcp_host to a configured DB host name.
        if default_mcp_url.startswith("managed://"):
            display_host = default_mcp_url[len("managed://") :].strip()
        elif default_mcp_url:
            for it in ws_hosts:
                href = str((it or {}).get("href", "")).strip()
                name = str((it or {}).get("text", "")).strip()
                if href and name and href == default_mcp_url:
                    display_host = name
                    break
    if not display_host and isinstance(managed_agents, list) and len(managed_agents) == 1:
        display_host = str((managed_agents[0] or {}).get("name", "")).strip() or ""

    summary_mode = any(
        re.search(p, user_question, flags=re.IGNORECASE)
        for p in [
            r"\bsummary\b",
            r"\bsummarize\b",
            r"\boverview\b",
            r"\banaly(?:s|z)e\b",
            r"\banalysis\b",
        ]
    )
    if re.search(r"\bcollect\s+slow\s+queries?\b", user_question, flags=re.IGNORECASE) and not re.search(
        r"\b(raw|rows|full|all)\b", user_question, flags=re.IGNORECASE
    ):
        summary_mode = True
    if re.search(r"\bcluster\b", user_question, flags=re.IGNORECASE) or (
        db_host_name and "cluster" in db_host_name.lower()
    ):
        summary_mode = True

    sql_query = (
        "select Time, digest, plan_digest, INSTANCE, query_time, plan, "
        "substring(query, 1, 2000) as query, "
        "rocksdb_key_skipped_count "
        "from information_schema.cluster_slow_query "
        f"where time >= '{start_ts}' and time <= '{end_ts}' "
        f"and Time BETWEEN '{start_ts}' AND '{end_ts}' "
        "order by rocksdb_key_skipped_count desc "
        "limit 20"
    )

    try:
        SiteSetting.update_db_cache()
        ws_list = getattr(SiteSetting, "mcp_hosts", None) or []
        managed_list = getattr(SiteSetting, "managed_mcp_agents", None) or []
        ws_names = {
            str((it or {}).get("text", "")).strip().lower()
            for it in ws_list
            if it
            and (href := str((it or {}).get("href", "")).strip())
            and (href.startswith("ws://") or href.startswith("wss://") or href.startswith("managed://"))
        }
        managed_names = {str((it or {}).get("name", "")).strip().lower() for it in managed_list if it}
    except Exception:
        ws_names, managed_names = set(), set()

    try:
        if summary_mode:
            result_rows = None
            cached = getattr(chat_flow, "_cached_slow_query_meta", None)
            if isinstance(cached, dict) and cached.get("type") == "slow_query_rows":
                same_host = (cached.get("host_name") or "") == (db_host_name or "")
                same_window = cached.get("start") == start_ts and cached.get("end") == end_ts
                if same_host and same_window and isinstance(cached.get("rows"), list):
                    result_rows = cached.get("rows")
            if result_rows is None:
                if not db_host_ready:
                    result_rows = []
                elif db_host_name and db_host_name.lower() in managed_names and db_host_name.lower() not in ws_names:
                    from app.mcp.managed import run_managed_mcp_db_query  # local import
                    result_rows = run_managed_mcp_db_query(db_host_name, sql_query)
                else:
                    result_rows = run_mcp_db_query(sql_query, host_name=db_host_name)

            raw_rows = normalize_rows(result_rows, logger=logger)
            if not raw_rows and isinstance(getattr(chat_flow, "_cached_slow_query_meta", None), dict):
                cached = chat_flow._cached_slow_query_meta or {}
                if cached.get("type") == "slow_query_rows" and isinstance(cached.get("rows"), list):
                    raw_rows = cached.get("rows") or []
            digest_rows, instance_rows, tables_rows = build_summary_from_rows(raw_rows)

            digest_md = rows_to_markdown(
                digest_rows,
                ["digest", "sample_query", "plan_digest", "exec_count", "avg_s", "max_s", "skipped_sum"],
            )
            tables_md = rows_to_markdown(tables_rows, ["table", "exec_count", "total_s"])
            plan_sections: list[str] = []
            plan_recos: list[dict] = []
            for r in raw_rows[:10]:
                if not isinstance(r, dict):
                    continue
                try:
                    skipped = float(r.get("rocksdb_key_skipped_count") or 0.0)
                except (TypeError, ValueError):
                    skipped = 0.0
                if skipped <= 0:
                    continue
                plan_text = str(r.get("plan") or "").strip()
                if not plan_text:
                    continue
                digest = str(r.get("digest") or "-")
                plan_sections.append(
                    "Plan (high rocksdb_key_skipped_count):\n\n"
                    f"- digest: {digest}\n"
                    f"- plan: {plan_text}"
                )
                if "TableFullScan" in plan_text:
                    plan_recos.append({
                        "score": 2,
                        "text": "Plan shows TableFullScan; add/selective indexes or narrow predicates to avoid full scans.",
                    })
                if "IndexFullScan" in plan_text:
                    plan_recos.append({
                        "score": 2,
                        "text": "Plan shows IndexFullScan; consider more selective indexes or range predicates.",
                    })
            plan_analysis_text = "\n\n".join(plan_sections) if plan_sections else ""
            ai_examples_json, ai_status_line, ai_recommendations_text = build_ai_recommendations(
                raw_rows,
                chat_flow._fast_llm,
                logger,
            )
            chat_flow._cached_slow_query_meta = json_safe({
                "type": "slow_query_summary",
                "host_name": db_host_name,
                "start": start_ts,
                "end": end_ts,
                "digests": digest_rows,
                "instances": instance_rows,
                "tables": tables_rows,
            })
            top_digest = digest_rows[0] if isinstance(digest_rows, list) and digest_rows else {}
            display_host = display_host or ("default" if default_mcp_ok else "not configured")
            summary_lines = [
                f"Time window (UTC): {start_ts} to {end_ts}",
                f"Host: {display_host}",
            ]
            summary_text = "\n".join(f"- {line}" for line in summary_lines)

            statement_sql = build_statement_summary_query(start_ts, end_ts)
            stmt_rows: list[dict] = []
            try:
                if not db_host_ready:
                    stmt_result = []
                else:
                    if db_host_name and db_host_name.lower() in managed_names and db_host_name.lower() not in ws_names:
                        from app.mcp.managed import run_managed_mcp_db_query  # local import
                        stmt_result = run_managed_mcp_db_query(db_host_name, statement_sql)
                    else:
                        stmt_result = run_mcp_db_query(statement_sql, host_name=db_host_name)
                normalized_stmt_rows = normalize_rows(stmt_result, logger=logger)
                if isinstance(normalized_stmt_rows, list):
                    stmt_rows = [r for r in normalized_stmt_rows if isinstance(r, dict)]
            except Exception as e:
                logger.exception("Statement summary query failed: %s", e)
                stmt_rows = []

            stmt_table_rows = []
            for r in stmt_rows[:10]:
                schema_name = str(r.get("agg_schema_name") or "")
                digest_text = r.get("agg_digest_text", "-")
                if schema_name.lower() == "information_schema":
                    digest_text = "-"
                stmt_table_rows.append({
                    "agg_digest_text": digest_text,
                    "agg_schema_name": r.get("agg_schema_name", "-"),
                    "agg_plan_count": r.get("agg_plan_count", "-"),
                    "agg_exec_count": r.get("agg_exec_count", "-"),
                    "agg_sum_latency_s": round(latency_to_seconds(r.get("agg_sum_latency")), 3),
                    "agg_max_latency_s": round(latency_to_seconds(r.get("agg_max_latency")), 3),
                    "agg_avg_latency_s": round(latency_to_seconds(r.get("agg_avg_latency")), 3),
                })
            stmt_md = rows_to_markdown(
                stmt_table_rows,
                [
                    "agg_digest_text",
                    "agg_schema_name",
                    "agg_plan_count",
                    "agg_exec_count",
                    "agg_sum_latency_s",
                    "agg_max_latency_s",
                    "agg_avg_latency_s",
                ],
            )

            grafana_sections = [
                build_grafana_duration_analysis(start_ts, end_ts, grafana_host_name, logger, cluster_hint),
                build_grafana_tidb_metrics_analysis(start_ts, end_ts, grafana_host_name, logger, cluster_hint),
            ]
            grafana_text = "\n\n".join(s for s in grafana_sections if s)

            recommendations: list[dict] = []
            if isinstance(top_digest, dict):
                max_s = float(top_digest.get("max_s") or 0.0)
                skipped_sum = float(top_digest.get("skipped_sum") or 0.0)
                exec_count = int(top_digest.get("exec_count") or 0)
                rules = [
                    {
                        "score": 3 if max_s >= 2.0 else 2 if max_s >= 1.0 else 0,
                        "text": "Investigate the slowest query; review execution plan and indexes.",
                    },
                    {
                        "score": 3 if skipped_sum >= 50000 else 2 if skipped_sum >= 10000 else 0,
                        "text": "High rocksdb_key_skipped_count; check for inefficient range scans or missing indexes.",
                    },
                    {
                        "score": 2 if exec_count >= 100 else 1 if exec_count >= 50 else 0,
                        "text": "High exec_count; consider caching, batching, or throttling repeated queries.",
                    },
                ]
                recommendations = [r for r in rules if r["score"] > 0]
                recommendations.sort(key=lambda r: r["score"], reverse=True)
            if plan_recos:
                recommendations.extend(plan_recos)
                recommendations.sort(key=lambda r: r.get("score", 0), reverse=True)
            if stmt_rows:
                top_stmt = stmt_rows[0]
                try:
                    plan_count = int(top_stmt.get("agg_plan_count") or 0)
                except (TypeError, ValueError):
                    plan_count = 0
                max_latency_s = latency_to_seconds(top_stmt.get("agg_max_latency"))
                avg_latency_s = latency_to_seconds(top_stmt.get("agg_avg_latency"))
                sum_latency_s = latency_to_seconds(top_stmt.get("agg_sum_latency"))
                if plan_count >= 5 and (max_latency_s >= 1.0 or avg_latency_s >= 0.5):
                    recommendations.append({
                        "score": 3,
                        "text": "High plan count with elevated latency; review plan stability, update stats, and consider plan bindings.",
                    })
                elif plan_count >= 5 and sum_latency_s > 0:
                    recommendations.append({
                        "score": 2,
                        "text": "Multiple plans detected for this digest; check for plan cache instability and inconsistent parameter patterns.",
                    })
                recommendations.sort(key=lambda r: r.get("score", 0), reverse=True)
            if not recommendations:
                recommendations_text = "- No obvious hotspots detected; consider widening the time window."
            else:
                recommendations_text = "\n".join(f"- {r['text']}" for r in recommendations)
            recommendations_text = (
                f"{recommendations_text}\n\nAI recommendations:\n{ai_recommendations_text}"
            )

            formatted_rows = []
            for r in raw_rows:
                if isinstance(r, dict):
                    formatted = dict(r)
                    if "query" in formatted:
                        formatted["query"] = str(formatted.get("query") or "")[:200]
                    formatted.setdefault("Time", "-")
                    formatted.setdefault("INSTANCE", "-")
                    formatted.setdefault("query_time", "-")
                    formatted.setdefault("digest", "-")
                    formatted.setdefault("plan_digest", "-")
                    formatted.setdefault("rocksdb_key_skipped_count", "-")
                    formatted_rows.append(formatted)
            query_output_md = rows_to_markdown(
                formatted_rows,
                [
                    "Time",
                    "INSTANCE",
                    "query_time",
                    "digest",
                    "plan_digest",
                    "query",
                    "rocksdb_key_skipped_count",
                ],
            )
            response_text = (
                "Slow query summary (high-level):\n\n"
                f"{summary_text}\n\n"
                "Query (slow query rows):\n\n"
                f"```sql\n{sql_query}\n```\n\n"
                "Query summary (statement history):\n\n"
                f"```sql\n{statement_sql}\n```\n\n"
                "Statement summary (by digest):\n\n"
                f"{stmt_md}\n\n"
                f"{grafana_text}\n\n"
                "Query output (raw rows):\n\n"
                f"{query_output_md}\n\n"
                "Slow query summary (by digest):\n\n"
                f"{digest_md}\n\n"
                "Impacted tables:\n\n"
                f"{tables_md}\n\n"
                "AI inputs (query + plan JSON):\n\n"
                f"```json\n{ai_examples_json}\n```\n\n"
                f"{ai_status_line}\n\n"
                "Recommendations:\n\n"
                f"{recommendations_text}"
            )
            if len(response_text) > max_chars:
                logger.info(
                    "Slow query summary truncated len=%d max=%d sections_present=%s",
                    len(response_text),
                    max_chars,
                    {
                        "grafana": "Grafana anomalies (window)" in response_text,
                        "ai_inputs": "AI inputs (query + plan JSON)" in response_text,
                        "ai_status": "AI status:" in response_text,
                        "recommendations": "Recommendations:" in response_text,
                    },
                )
                response_text = response_text[:max_chars] + "\n\n[truncated]"
            return response_text
        else:
            result = None
            cached = getattr(chat_flow, "_cached_slow_query_meta", None)
            if isinstance(cached, dict) and cached.get("type") == "slow_query_rows":
                same_host = (cached.get("host_name") or "") == (db_host_name or "")
                same_window = cached.get("start") == start_ts and cached.get("end") == end_ts
                if same_host and same_window and isinstance(cached.get("rows"), list):
                    result = cached.get("rows")
            if result is None:
                if not db_host_ready:
                    return "Database query skipped: no valid DB MCP host configured for this request."
                if db_host_name and db_host_name.lower() in managed_names and db_host_name.lower() not in ws_names:
                    from app.mcp.managed import run_managed_mcp_db_query  # local import
                    result = run_managed_mcp_db_query(db_host_name, sql_query)
                else:
                    result = run_mcp_db_query(sql_query, host_name=db_host_name)
            logger.info("Slow query raw result type=%s", type(result).__name__)
            parsed_result = parse_mcp_text_result(result, logger=logger)
            if isinstance(parsed_result, str):
                parsed_result = coerce_text_payload(parsed_result)
            logger.info(
                "Slow query parsed result type=%s preview=%s",
                type(parsed_result).__name__,
                str(parsed_result)[:200].replace("\n", "\\n"),
            )

            def _clean_value(value: Any) -> str:
                return clean_cell(value)

            rows = normalize_rows(parsed_result, logger=logger)
            if rows:
                columns = ["Time", "INSTANCE", "query_time", "query", "rocksdb_key_skipped_count"]
                lines = []
                header = " | ".join(columns)
                sep = " | ".join(["---"] * len(columns))
                lines.append(header)
                lines.append(sep)
                for r in rows[:20]:
                    if isinstance(r, dict):
                        values = [
                            _clean_value(r.get("Time")),
                            _clean_value(r.get("INSTANCE")),
                            _clean_value(r.get("query_time")),
                            _clean_value(r.get("query"))[:200],
                            _clean_value(r.get("rocksdb_key_skipped_count")),
                        ]
                    elif isinstance(r, (list, tuple)):
                        values = [_clean_value(x) for x in r[: len(columns)]]
                    else:
                        values = [_clean_value(r)]
                    lines.append(" | ".join(values))
                return "\n".join(lines)
            pretty = json.dumps(parsed_result, indent=2, ensure_ascii=False, default=str) if isinstance(parsed_result, (list, dict)) else str(parsed_result)
            response_text = f"{pretty}"
            if len(response_text) > max_chars:
                response_text = response_text[:max_chars] + "\n\n[truncated]"
            return response_text
    except Exception as e:
        logger.exception("DB slow query flow failed: %s", e)
    return None
