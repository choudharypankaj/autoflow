import ast
import json
import logging
import re
from typing import Any, Optional

from llama_index.core.prompts.rich import RichPromptTemplate


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
