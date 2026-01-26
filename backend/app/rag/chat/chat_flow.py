import json
import re
import logging
from datetime import datetime, UTC, date, time
from typing import List, Optional, Generator, Tuple, Any
from urllib.parse import urljoin
from uuid import UUID

import requests
from langfuse.llama_index import LlamaIndexInstrumentor
from langfuse.llama_index._context import langfuse_instrumentor_context
from llama_index.core import get_response_synthesizer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore
from llama_index.core.prompts.rich import RichPromptTemplate

from sqlmodel import Session
from app.core.config import settings
from app.exceptions import ChatNotFound
from app.models import (
    User,
    Chat as DBChat,
    ChatVisibility,
    ChatMessage as DBChatMessage,
)
from app.rag.chat.config import ChatEngineConfig
from app.rag.chat.retrieve.retrieve_flow import SourceDocument, RetrieveFlow
from app.rag.chat.stream_protocol import (
    ChatEvent,
    ChatStreamDataPayload,
    ChatStreamMessagePayload,
)
from app.rag.llms.dspy import get_dspy_lm_by_llama_llm
from app.rag.retrievers.knowledge_graph.schema import KnowledgeGraphRetrievalResult
from app.rag.types import ChatEventType, ChatMessageSate
from app.rag.utils import parse_goal_response_format
from app.repositories import chat_repo
from app.site_settings import SiteSetting
from app.utils.tracing import LangfuseContextManager
from app.mcp.client import run_mcp_db_query

logger = logging.getLogger(__name__)

MAX_CHAT_RESULT_CHARS = 60000

def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date, time)):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    try:
        # Last resort stringify
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)

def _extract_tables_from_sql(sql_text: str) -> list[str]:
    """
    Best-effort extractor for table names from SQL using FROM/JOIN patterns.
    Returns lowercased table identifiers without schema/quotes.
    """
    if not isinstance(sql_text, str):
        return []
    # Support FROM, JOIN, INSERT INTO, UPDATE, DELETE FROM
    matches = re.findall(r"\b(?:from|join|into|update|delete\s+from)\s+([`\"\w\.]+)", sql_text, flags=re.IGNORECASE)
    tables: list[str] = []
    for m in matches:
        t = m.strip('`"').split('.')[-1].lower()
        if t:
            tables.append(t)
    return tables

def _rows_to_markdown(rows: Any, columns: list[str]) -> str:
    if not isinstance(rows, list) or not rows:
        return "(no data)"
    lines = []
    header = " | ".join(columns)
    sep = " | ".join(["---"] * len(columns))
    lines.append(header)
    lines.append(sep)
    for r in rows[:10]:
        if isinstance(r, dict):
            values = [str(r.get(c, "")) for c in columns]
        elif isinstance(r, (list, tuple)):
            values = [str(x) for x in r[: len(columns)]]
        else:
            values = [str(r)]
        lines.append(" | ".join(values))
    return "\n".join(lines)


def parse_chat_messages(
    chat_messages: List[ChatMessage],
) -> tuple[str, List[ChatMessage]]:
    user_question = chat_messages[-1].content
    chat_history = chat_messages[:-1]
    return user_question, chat_history


class ChatFlow:
    _trace_manager: LangfuseContextManager

    def __init__(
        self,
        *,
        db_session: Session,
        user: User,
        browser_id: str,
        origin: str,
        chat_messages: List[ChatMessage],
        engine_name: str = "default",
        chat_id: Optional[UUID] = None,
        mcp_host_name: Optional[str] = None,
    ) -> None:
        self.chat_id = chat_id
        self.db_session = db_session
        self.user = user
        self.browser_id = browser_id
        self.engine_name = engine_name
        self.mcp_host_name = mcp_host_name

        # Load chat engine and chat session.
        self.user_question, self.chat_history = parse_chat_messages(chat_messages)
        if chat_id:
            # FIXME:
            #   only chat owner or superuser can access the chat,
            #   anonymous user can only access anonymous chat by track_id
            self.db_chat_obj = chat_repo.get(self.db_session, chat_id)
            if not self.db_chat_obj:
                raise ChatNotFound(chat_id)
            try:
                self.engine_config = ChatEngineConfig.load_from_db(
                    db_session, self.db_chat_obj.engine.name
                )
                self.db_chat_engine = self.engine_config.get_db_chat_engine()
            except Exception as e:
                logger.error(f"Failed to load chat engine config: {e}")
                self.engine_config = ChatEngineConfig.load_from_db(
                    db_session, engine_name
                )
                self.db_chat_engine = self.engine_config.get_db_chat_engine()
            logger.info(
                f"Init ChatFlow for chat {chat_id} (chat_engine: {self.db_chat_obj.engine.name})"
            )
            self.chat_history = [
                ChatMessage(role=m.role, content=m.content, additional_kwargs={})
                for m in chat_repo.get_messages(self.db_session, self.db_chat_obj)
            ]
        else:
            self.engine_config = ChatEngineConfig.load_from_db(db_session, engine_name)
            self.db_chat_engine = self.engine_config.get_db_chat_engine()
            self.db_chat_obj = chat_repo.create(
                self.db_session,
                DBChat(
                    # TODO: title should be generated by the LLM
                    title=self.user_question[:100],
                    engine_id=self.db_chat_engine.id,
                    engine_options=self.engine_config.screenshot(),
                    user_id=self.user.id if self.user else None,
                    browser_id=self.browser_id,
                    origin=origin,
                    visibility=(
                        ChatVisibility.PUBLIC
                        if not self.user
                        else ChatVisibility.PRIVATE
                    ),
                ),
            )
            chat_id = self.db_chat_obj.id

            # Notice: slack/discord bots may create a new chat with history messages.
            now = datetime.now(UTC)
            for i, m in enumerate(self.chat_history):
                chat_repo.create_message(
                    session=self.db_session,
                    chat=self.db_chat_obj,
                    chat_message=DBChatMessage(
                        role=m.role,
                        content=m.content,
                        ordinal=i + 1,
                        created_at=now,
                        updated_at=now,
                        finished_at=now,
                    ),
                )

        # Init Langfuse for tracing.
        enable_langfuse = (
            SiteSetting.langfuse_secret_key and SiteSetting.langfuse_public_key
        )
        instrumentor = LlamaIndexInstrumentor(
            host=SiteSetting.langfuse_host,
            secret_key=SiteSetting.langfuse_secret_key,
            public_key=SiteSetting.langfuse_public_key,
            enabled=enable_langfuse,
        )
        self._trace_manager = LangfuseContextManager(instrumentor)

        # Init LLM.
        self._llm = self.engine_config.get_llama_llm(self.db_session)
        self._fast_llm = self.engine_config.get_fast_llama_llm(self.db_session)
        self._fast_dspy_lm = get_dspy_lm_by_llama_llm(self._fast_llm)

        # Load knowledge bases.
        self.knowledge_bases = self.engine_config.get_knowledge_bases(self.db_session)
        self.knowledge_base_ids = [kb.id for kb in self.knowledge_bases]

        # Init retrieve flow.
        self.retrieve_flow = RetrieveFlow(
            db_session=self.db_session,
            engine_name=self.engine_name,
            engine_config=self.engine_config,
            llm=self._llm,
            fast_llm=self._fast_llm,
            knowledge_bases=self.knowledge_bases,
        )
        # Cached structured meta for slow query runs to power follow-ups
        self._cached_slow_query_meta: Optional[dict] = None

    def chat(self) -> Generator[ChatEvent | str, None, None]:
        try:
            with self._trace_manager.observe(
                trace_name="ChatFlow",
                user_id=(
                    self.user.email if self.user else f"anonymous-{self.browser_id}"
                ),
                metadata={
                    "is_external_engine": self.engine_config.is_external_engine,
                    "chat_engine_config": self.engine_config.screenshot(),
                },
                tags=[f"chat_engine:{self.engine_name}"],
                release=settings.ENVIRONMENT,
            ) as trace:
                trace.update(
                    input={
                        "user_question": self.user_question,
                        "chat_history": self.chat_history,
                    }
                )

                if self.engine_config.is_external_engine:
                    yield from self._external_chat()
                else:
                    response_text, source_documents = yield from self._builtin_chat()
                    trace.update(output=response_text)
        except Exception as e:
            logger.exception(e)
            yield ChatEvent(
                event_type=ChatEventType.ERROR_PART,
                payload="Encountered an error while processing the chat. Please try again later.",
            )

    def _builtin_chat(
        self,
    ) -> Generator[ChatEvent | str, None, Tuple[Optional[str], List[Any]]]:
        ctx = langfuse_instrumentor_context.get().copy()
        db_user_message, db_assistant_message = yield from self._chat_start()
        langfuse_instrumentor_context.get().update(ctx)

        # Special-case: DB slow query agent (user provides UTC start/end)
        response_text = self._maybe_run_db_slow_query(self.user_question)
        if response_text is not None:
            yield from self._chat_finish(
                db_assistant_message=db_assistant_message,
                db_user_message=db_user_message,
                response_text=response_text,
                knowledge_graph=KnowledgeGraphRetrievalResult(),
                source_documents=[],
                annotation_silent=True,
                extra_meta=self._cached_slow_query_meta,
            )
            return response_text, []

        # 1. Retrieve Knowledge graph related to the user question.
        (
            knowledge_graph,
            knowledge_graph_context,
        ) = yield from self._search_knowledge_graph(user_question=self.user_question)

        # 2. Refine the user question using knowledge graph and chat history.
        refined_question = yield from self._refine_user_question(
            user_question=self.user_question,
            chat_history=self.chat_history,
            knowledge_graph_context=knowledge_graph_context,
            refined_question_prompt=self.engine_config.llm.condense_question_prompt,
        )

        # 3. Check if the question provided enough context information or need to clarify.
        if self.engine_config.clarify_question:
            need_clarify, need_clarify_response = yield from self._clarify_question(
                user_question=refined_question,
                chat_history=self.chat_history,
                knowledge_graph_context=knowledge_graph_context,
            )
            if need_clarify:
                yield from self._chat_finish(
                    db_assistant_message=db_assistant_message,
                    db_user_message=db_user_message,
                    response_text=need_clarify_response,
                    knowledge_graph=knowledge_graph,
                    source_documents=[],
                )
                return None, []

        # 4. Use refined question to search for relevant chunks.
        relevant_chunks = yield from self._search_relevance_chunks(
            user_question=refined_question
        )

        # 5. Generate a response using the refined question and related chunks
        response_text, source_documents = yield from self._generate_answer(
            user_question=refined_question,
            knowledge_graph_context=knowledge_graph_context,
            relevant_chunks=relevant_chunks,
        )

        yield from self._chat_finish(
            db_assistant_message=db_assistant_message,
            db_user_message=db_user_message,
            response_text=response_text,
            knowledge_graph=knowledge_graph,
            source_documents=source_documents,
        )

        return response_text, source_documents

    def _maybe_run_db_slow_query(self, user_question: str) -> Optional[str]:
        """
        Detects and runs the predefined slow query against TiDB via MCP when the user
        supplies UTC time window. Expected formats (UTC):
          - 2026-01-14 16:15:00 to 2026-01-14 16:47:00
          - start: 2026-01-14 16:15:00, end: 2026-01-14 16:47:00
        """
        # 0) Follow-up meta-only path: if user references last/summary/digest/instance/table,
        # try answering from the last assistant meta immediately.
        if re.search(r"\b(last|previous|summary|summarize|digest|instance|table)\b", user_question, flags=re.IGNORECASE):
            try:
                prior_messages = chat_repo.get_messages(self.db_session, self.db_chat_obj)
                meta = None
                for m in reversed(prior_messages):
                    try:
                        if getattr(m, "role", "") == MessageRole.ASSISTANT.value and isinstance(m.meta, dict):
                            t = str(m.meta.get("type", ""))
                            if t in {"slow_query_summary", "slow_query_rows"}:
                                meta = m.meta
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
                        chunks.append("Top digests by total time:\n\n" + _rows_to_markdown(scored, ["digest", "exec_count", "avg_s", "total_s"]))
                    if want_instance and isinstance(instances, list):
                        chunks.append("Instances (cached):\n\n" + _rows_to_markdown(instances, ["INSTANCE", "exec_count", "avg_s", "total_s"]))
                    if want_table and isinstance(tables, list):
                        chunks.append("Impacted tables (cached):\n\n" + _rows_to_markdown(tables, ["table", "exec_count", "total_s"]))
                    if chunks:
                        text = "\n\n".join(chunks)
                        if len(text) > MAX_CHAT_RESULT_CHARS:
                            text = text[:MAX_CHAT_RESULT_CHARS] + "\n\n[truncated]"
                        self._cached_slow_query_meta = _json_safe(meta)
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
            return None

        # Extract two UTC timestamps
        # Accept "YYYY-MM-DD HH:MM:SS"
        ts_pattern = r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"
        matches = re.findall(ts_pattern, user_question)
        if len(matches) < 2:
            # Default to cached summary unless user explicitly asks for a fresh run
            force_fresh = bool(re.search(r"\b(fresh|re[-\s]?run|new\s+window|do\s+not\s+use\s+last|ignore\s+summary)\b", user_question, flags=re.IGNORECASE))
            if not force_fresh:
                try:
                    prior_messages = chat_repo.get_messages(self.db_session, self.db_chat_obj)
                    meta = None
                    for m in reversed(prior_messages):
                        try:
                            if getattr(m, "role", "") == MessageRole.ASSISTANT.value and isinstance(m.meta, dict):
                                t = str(m.meta.get("type", ""))
                                if t in {"slow_query_summary", "slow_query_rows"}:
                                    meta = m.meta
                                    break
                        except Exception:
                            continue
                    if meta and meta.get("type") == "slow_query_summary":
                        # Use cached summary for follow-up
                        want_instance = bool(re.search(r"\binstance", user_question, flags=re.IGNORECASE))
                        want_digest = bool(re.search(r"\bdigest", user_question, flags=re.IGNORECASE))
                        want_table = bool(re.search(r"\btable", user_question, flags=re.IGNORECASE))
                        digests = meta.get("digests") or []
                        instances = meta.get("instances") or []
                        tables = meta.get("tables") or []
                        chunks: list[str] = []
                        if want_digest or (not want_instance and not want_table):
                            # Default to digests if user didn't specify a facet
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
                                chunks.append("Top digests by total time (cached):\n\n" + _rows_to_markdown(scored, ["digest", "exec_count", "avg_s", "total_s"]))
                        if want_instance and isinstance(instances, list):
                            chunks.append("Instances (cached):\n\n" + _rows_to_markdown(instances, ["INSTANCE", "exec_count", "avg_s", "total_s"]))
                        if want_table and isinstance(tables, list):
                            chunks.append("Impacted tables (cached):\n\n" + _rows_to_markdown(tables, ["table", "exec_count", "total_s"]))
                        if chunks:
                            text = "\n\n".join(chunks)
                            if len(text) > MAX_CHAT_RESULT_CHARS:
                                text = text[:MAX_CHAT_RESULT_CHARS] + "\n\n[truncated]"
                            self._cached_slow_query_meta = _json_safe(meta)
                            return text
                except Exception:
                    pass
            # No cached context usable or user forced fresh; ask for window
            return (
                "Please provide UTC start and end times in the format "
                "'YYYY-MM-DD HH:MM:SS'. Example: start 2026-01-14 16:15:00, end 2026-01-14 16:47:00"
            )

        start_ts, end_ts = matches[0], matches[1]

        # Allow inline agent selection, e.g. "for prod mcp", "using prod database", "on prod db"
        host_name = self.mcp_host_name
        if not host_name:
            try:
                m = re.search(
                    r"\b(?:for|using|on)\s+([A-Za-z0-9._-]+)\s+(?:mcp|db|database)\b",
                    user_question,
                    flags=re.IGNORECASE,
                )
                candidate = m.group(1).strip() if m else ""
                # Validate against configured MCP hosts and managed agents
                SiteSetting.update_db_cache()
                ws = getattr(SiteSetting, "mcp_hosts", None) or []
                managed = getattr(SiteSetting, "managed_mcp_agents", None) or []
                valid_names = set()
                for item in ws:
                    try:
                        name = str(item.get("text", "")).strip()
                        if name:
                            valid_names.add(name.lower())
                    except Exception:
                        continue
                for item in managed:
                    try:
                        name = str(item.get("name", "")).strip()
                        if name:
                            valid_names.add(name.lower())
                    except Exception:
                        continue
                if candidate and candidate.lower() in valid_names:
                    host_name = candidate
                # If still no host_name provided, but exactly one managed agent exists, default to it
                if not host_name and isinstance(managed, list) and len(managed) == 1:
                    maybe_name = str((managed[0] or {}).get("name", "")).strip()
                    if maybe_name:
                        host_name = maybe_name
            except Exception:
                # Non-fatal; just skip inline selection on error
                pass

        # Determine if user asked for a summary/analysis rather than raw rows
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

        # Construct SQLs
        if summary_mode:
            sql_digest = (
                "select "
                "digest, "
                "any_value(substring(query, 1, 200)) as sample_query, "
                "any_value(plan_digest) as plan_digest, "
                "count(*) as exec_count, "
                "round(avg(query_time), 3) as avg_s, "
                "round(max(query_time), 3) as max_s, "
                "sum(rocksdb_key_skipped_count) as skipped_sum "
                "from information_schema.CLUSTER_SLOW_QUERY "
                "where is_internal = false "
                f"and Time BETWEEN '{start_ts}' AND '{end_ts}' "
                "group by digest "
                "order by sum(query_time) desc "
                "limit 10"
            )
            sql_instance = (
                "select "
                "INSTANCE, "
                "count(*) as exec_count, "
                "round(avg(query_time), 3) as avg_s, "
                "round(sum(query_time), 3) as total_s "
                "from information_schema.CLUSTER_SLOW_QUERY "
                "where is_internal = false "
                f"and Time BETWEEN '{start_ts}' AND '{end_ts}' "
                "group by INSTANCE "
                "order by total_s desc "
                "limit 10"
            )
        else:
            sql = (
                "select Time, INSTANCE, query_time, "
                "substring(query, 1, 2000) as query, "
                "rocksdb_key_skipped_count "
                "from information_schema.CLUSTER_SLOW_QUERY "
                "where is_internal = false "
                f"and Time BETWEEN '{start_ts}' AND '{end_ts}' "
                "order by rocksdb_key_skipped_count desc "
                "limit 20"
            )

        # If the selected name corresponds to a managed agent (and not a WS host),
        # call the managed path directly to avoid WS URL validation errors.
        try:
            SiteSetting.update_db_cache()
            ws_list = getattr(SiteSetting, "mcp_hosts", None) or []
            managed_list = getattr(SiteSetting, "managed_mcp_agents", None) or []
            ws_names = {str((it or {}).get("text", "")).strip().lower() for it in ws_list if it}
            managed_names = {str((it or {}).get("name", "")).strip().lower() for it in managed_list if it}
        except Exception:
            ws_names, managed_names = set(), set()

        try:
            if summary_mode:
                # Run both summaries
                if host_name and host_name.lower() in managed_names and host_name.lower() not in ws_names:
                    from app.mcp.managed import run_managed_mcp_db_query  # local import
                    result_digest = run_managed_mcp_db_query(host_name, sql_digest)
                    result_instance = run_managed_mcp_db_query(host_name, sql_instance)
                else:
                    result_digest = run_mcp_db_query(sql_digest, host_name=host_name)
                    result_instance = run_mcp_db_query(sql_instance, host_name=host_name)

                # Render concise summary
                def rows_to_markdown(rows: Any, columns: list[str]) -> str:
                    if not isinstance(rows, list) or not rows:
                        return "(no data)"
                    # normalize dict rows
                    lines = []
                    header = " | ".join(columns)
                    sep = " | ".join(["---"] * len(columns))
                    lines.append(header)
                    lines.append(sep)
                    for r in rows[:10]:
                        if isinstance(r, dict):
                            values = [str(r.get(c, "")) for c in columns]
                        elif isinstance(r, (list, tuple)):
                            values = [str(x) for x in r[: len(columns)]]
                        else:
                            values = [str(r)]
                        lines.append(" | ".join(values))
                    return "\n".join(lines)

                digest_md = rows_to_markdown(
                    result_digest,
                    ["digest", "sample_query", "plan_digest", "exec_count", "avg_s", "max_s", "skipped_sum"],
                )
                instance_md = rows_to_markdown(
                    result_instance,
                    ["INSTANCE", "exec_count", "avg_s", "total_s"],
                )
                # Impacted tables derived from digest sample_query
                tables_agg: dict[str, dict] = {}
                if isinstance(result_digest, list):
                    for r in result_digest:
                        if not isinstance(r, dict):
                            continue
                        sample = r.get("sample_query") or ""
                        exec_count = float(r.get("exec_count") or 0)
                        avg_s = float(r.get("avg_s") or 0.0)
                        total_s = exec_count * avg_s
                        for t in _extract_tables_from_sql(str(sample)):
                            agg = tables_agg.setdefault(t, {"table": t, "exec_count": 0, "total_s": 0.0})
                            agg["exec_count"] += int(exec_count)
                            agg["total_s"] += total_s
                tables_rows = sorted(tables_agg.values(), key=lambda x: x["total_s"], reverse=True)[:10]
                tables_md = rows_to_markdown(tables_rows, ["table", "exec_count", "total_s"])
                # Cache compact meta for follow-ups
                self._cached_slow_query_meta = _json_safe({
                    "type": "slow_query_summary",
                    "host_name": host_name,
                    "start": start_ts,
                    "end": end_ts,
                    "digests": result_digest if isinstance(result_digest, list) else [],
                    "instances": result_instance if isinstance(result_instance, list) else [],
                    "tables": tables_rows,
                })
                response_text = (
                    "Slow query summary (by digest):\n\n"
                    f"{digest_md}\n\n"
                    "Instance hotspots:\n\n"
                    f"{instance_md}\n\n"
                    "Impacted tables:\n\n"
                    f"{tables_md}"
                )
                if len(response_text) > MAX_CHAT_RESULT_CHARS:
                    response_text = response_text[:MAX_CHAT_RESULT_CHARS] + "\n\n[truncated]"
                return response_text
            else:
                # Raw rows path
                if host_name and host_name.lower() in managed_names and host_name.lower() not in ws_names:
                    # Directly use managed MCP
                    from app.mcp.managed import run_managed_mcp_db_query  # local import
                    result = run_managed_mcp_db_query(host_name, sql)
                else:
                    # Prefer WS if host_name maps to an MCP host; otherwise, default WS selection applies
                    result = run_mcp_db_query(sql, host_name=host_name)
                # Best-effort formatting
                if isinstance(result, (list, dict)):
                    pretty = json.dumps(result, indent=2, ensure_ascii=False, default=str)
                else:
                    pretty = str(result)
                # Cache compact meta for follow-ups (limit rows)
                compact_rows: list[dict] = []
                if isinstance(result, list):
                    for r in result[:20]:
                        if isinstance(r, dict):
                            compact_rows.append({
                                "Time": _json_safe(r.get("Time")),
                                "INSTANCE": _json_safe(r.get("INSTANCE")),
                                "query_time": _json_safe(r.get("query_time")),
                                "query": _json_safe(r.get("query")),
                                "rocksdb_key_skipped_count": _json_safe(r.get("rocksdb_key_skipped_count")),
                            })
                self._cached_slow_query_meta = _json_safe({
                    "type": "slow_query_rows",
                    "host_name": host_name,
                    "start": start_ts,
                    "end": end_ts,
                    "rows": compact_rows,
                })
                response_text = (
                    "Here are the top slow queries by rocksdb_key_skipped_count:\n\n"
                    f"{pretty}"
                )
                if len(response_text) > MAX_CHAT_RESULT_CHARS:
                    response_text = response_text[:MAX_CHAT_RESULT_CHARS] + "\n\n[truncated]"
                return response_text
        except Exception as e:
            # Fallback to managed agents if named or if exactly one managed agent is configured
            # First, if this is a WS scheme error and a host_name was provided, try that name as a managed agent directly.
            if host_name and isinstance(e, Exception) and "Only ws:// or wss://" in str(e):
                try:
                    from app.mcp.managed import run_managed_mcp_db_query  # local import
                    if summary_mode:
                        rd = run_managed_mcp_db_query(host_name, sql_digest)
                        ri = run_managed_mcp_db_query(host_name, sql_instance)
                        # derive tables from digest result
                        tables_agg: dict[str, dict] = {}
                        if isinstance(rd, list):
                            for r in rd:
                                if not isinstance(r, dict):
                                    continue
                                sample = r.get("sample_query") or ""
                                exec_count = float(r.get("exec_count") or 0)
                                avg_s = float(r.get("avg_s") or 0.0)
                                total_s = exec_count * avg_s
                                for t in _extract_tables_from_sql(str(sample)):
                                    agg = tables_agg.setdefault(t, {"table": t, "exec_count": 0, "total_s": 0.0})
                                    agg["exec_count"] += int(exec_count)
                                    agg["total_s"] += total_s
                        tables_rows = sorted(tables_agg.values(), key=lambda x: x["total_s"], reverse=True)[:10]
                        text = (
                            "Slow query summary (managed fallback):\n\n"
                            "Digests:\n" + json.dumps(rd[:10] if isinstance(rd, list) else rd, indent=2, ensure_ascii=False, default=str) +
                            "\n\nInstances:\n" + json.dumps(ri[:10] if isinstance(ri, list) else ri, indent=2, ensure_ascii=False, default=str) +
                            "\n\nTables:\n" + json.dumps(tables_rows, indent=2, ensure_ascii=False, default=str)
                        )
                        if len(text) > MAX_CHAT_RESULT_CHARS:
                            text = text[:MAX_CHAT_RESULT_CHARS] + "\n\n[truncated]"
                        return text
                    else:
                        result = run_managed_mcp_db_query(host_name, sql)
                        pretty = json.dumps(result, indent=2, ensure_ascii=False, default=str) if isinstance(result, (list, dict)) else str(result)
                        response_text = (
                            "Here are the top slow queries by rocksdb_key_skipped_count:\n\n"
                            f"{pretty}"
                        )
                        if len(response_text) > MAX_CHAT_RESULT_CHARS:
                            response_text = response_text[:MAX_CHAT_RESULT_CHARS] + "\n\n[truncated]"
                        return response_text
                except Exception as e2:
                    logger.exception("Managed MCP direct attempt failed: %s", e2)
            fallback_name = host_name
            if not fallback_name:
                try:
                    SiteSetting.update_db_cache()
                    managed = getattr(SiteSetting, "managed_mcp_agents", None) or []
                    if isinstance(managed, list) and len(managed) == 1:
                        maybe_name = str((managed[0] or {}).get("name", "")).strip()
                        if maybe_name:
                            fallback_name = maybe_name
                except Exception:
                    pass
            if fallback_name:
                try:
                    from app.mcp.managed import run_managed_mcp_db_query  # local import to avoid overhead
                    if summary_mode:
                        rd = run_managed_mcp_db_query(fallback_name, sql_digest)
                        ri = run_managed_mcp_db_query(fallback_name, sql_instance)
                        tables_agg: dict[str, dict] = {}
                        if isinstance(rd, list):
                            for r in rd:
                                if not isinstance(r, dict):
                                    continue
                                sample = r.get("sample_query") or ""
                                exec_count = float(r.get("exec_count") or 0)
                                avg_s = float(r.get("avg_s") or 0.0)
                                total_s = exec_count * avg_s
                                for t in _extract_tables_from_sql(str(sample)):
                                    agg = tables_agg.setdefault(t, {"table": t, "exec_count": 0, "total_s": 0.0})
                                    agg["exec_count"] += int(exec_count)
                                    agg["total_s"] += total_s
                        tables_rows = sorted(tables_agg.values(), key=lambda x: x["total_s"], reverse=True)[:10]
                        text = (
                            "Slow query summary (managed fallback):\n\n"
                            "Digests:\n" + json.dumps(rd[:10] if isinstance(rd, list) else rd, indent=2, ensure_ascii=False, default=str) +
                            "\n\nInstances:\n" + json.dumps(ri[:10] if isinstance(ri, list) else ri, indent=2, ensure_ascii=False, default=str) +
                            "\n\nTables:\n" + json.dumps(tables_rows, indent=2, ensure_ascii=False, default=str)
                        )
                        if len(text) > MAX_CHAT_RESULT_CHARS:
                            text = text[:MAX_CHAT_RESULT_CHARS] + "\n\n[truncated]"
                        return text
                    else:
                        result = run_managed_mcp_db_query(fallback_name, sql)
                        pretty = json.dumps(result, indent=2, ensure_ascii=False, default=str) if isinstance(result, (list, dict)) else str(result)
                        response_text = (
                            "Here are the top slow queries by rocksdb_key_skipped_count:\n\n"
                            f"{pretty}"
                        )
                        if len(response_text) > MAX_CHAT_RESULT_CHARS:
                            response_text = response_text[:MAX_CHAT_RESULT_CHARS] + "\n\n[truncated]"
                        return response_text
                except Exception as e2:
                    logger.exception("Managed MCP fallback failed: %s", e2)
                    return f"Failed to run slow query via MCP: {e2}"
            logger.exception("DB slow query via MCP failed: %s", e)
            return f"Failed to run slow query via MCP: {e}"

    def _chat_start(
        self,
    ) -> Generator[ChatEvent, None, Tuple[DBChatMessage, DBChatMessage]]:
        db_user_message = chat_repo.create_message(
            session=self.db_session,
            chat=self.db_chat_obj,
            chat_message=DBChatMessage(
                role=MessageRole.USER.value,
                trace_url=self._trace_manager.trace_url,
                content=self.user_question.strip(),
            ),
        )
        db_assistant_message = chat_repo.create_message(
            session=self.db_session,
            chat=self.db_chat_obj,
            chat_message=DBChatMessage(
                role=MessageRole.ASSISTANT.value,
                trace_url=self._trace_manager.trace_url,
                content="",
            ),
        )
        yield ChatEvent(
            event_type=ChatEventType.DATA_PART,
            payload=ChatStreamDataPayload(
                chat=self.db_chat_obj,
                user_message=db_user_message,
                assistant_message=db_assistant_message,
            ),
        )
        return db_user_message, db_assistant_message

    def _search_knowledge_graph(
        self,
        user_question: str,
        annotation_silent: bool = False,
    ) -> Generator[ChatEvent, None, Tuple[KnowledgeGraphRetrievalResult, str]]:
        kg_config = self.engine_config.knowledge_graph
        if kg_config is None or kg_config.enabled is False:
            return KnowledgeGraphRetrievalResult(), ""

        with self._trace_manager.span(
            name="search_knowledge_graph", input=user_question
        ) as span:
            if not annotation_silent:
                if kg_config.using_intent_search:
                    yield ChatEvent(
                        event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                        payload=ChatStreamMessagePayload(
                            state=ChatMessageSate.KG_RETRIEVAL,
                            display="Identifying The Question's Intents and Perform Knowledge Graph Search",
                        ),
                    )
                else:
                    yield ChatEvent(
                        event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                        payload=ChatStreamMessagePayload(
                            state=ChatMessageSate.KG_RETRIEVAL,
                            display="Searching the Knowledge Graph for Relevant Context",
                        ),
                    )

            knowledge_graph, knowledge_graph_context = (
                self.retrieve_flow.search_knowledge_graph(user_question)
            )

            span.end(
                output={
                    "knowledge_graph": knowledge_graph,
                    "knowledge_graph_context": knowledge_graph_context,
                }
            )

        return knowledge_graph, knowledge_graph_context

    def _refine_user_question(
        self,
        user_question: str,
        chat_history: Optional[List[ChatMessage]] = [],
        refined_question_prompt: Optional[str] = None,
        knowledge_graph_context: str = "",
        annotation_silent: bool = False,
    ) -> Generator[ChatEvent, None, str]:
        with self._trace_manager.span(
            name="refine_user_question",
            input={
                "user_question": user_question,
                "chat_history": chat_history,
                "knowledge_graph_context": knowledge_graph_context,
            },
        ) as span:
            if not annotation_silent:
                yield ChatEvent(
                    event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                    payload=ChatStreamMessagePayload(
                        state=ChatMessageSate.REFINE_QUESTION,
                        display="Query Rewriting for Enhanced Information Retrieval",
                    ),
                )

            prompt_template = RichPromptTemplate(refined_question_prompt)
            refined_question = self._fast_llm.predict(
                prompt_template,
                graph_knowledges=knowledge_graph_context,
                chat_history=chat_history,
                question=user_question,
                current_date=datetime.now().strftime("%Y-%m-%d"),
            )

            if not annotation_silent:
                yield ChatEvent(
                    event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                    payload=ChatStreamMessagePayload(
                        state=ChatMessageSate.REFINE_QUESTION,
                        message=refined_question,
                    ),
                )

            span.end(output=refined_question)

            return refined_question

    def _clarify_question(
        self,
        user_question: str,
        chat_history: Optional[List[ChatMessage]] = [],
        knowledge_graph_context: str = "",
    ) -> Generator[ChatEvent, None, Tuple[bool, str]]:
        """
        Check if the question clear and provided enough context information, otherwise, it is necessary to
        stop the conversation early and ask the user for the further clarification.

        Args:
            user_question: str
            knowledge_graph_context: str

        Returns:
            bool: Determine whether further clarification of the issue is needed from the user.
            str: The content of the questions that require clarification from the user.
        """
        with self._trace_manager.span(
            name="clarify_question",
            input={
                "user_question": user_question,
                "knowledge_graph_context": knowledge_graph_context,
            },
        ) as span:
            prompt_template = RichPromptTemplate(
                self.engine_config.llm.clarifying_question_prompt
            )

            prediction = self._fast_llm.predict(
                prompt_template,
                graph_knowledges=knowledge_graph_context,
                chat_history=chat_history,
                question=user_question,
            )
            # TODO: using structured output to get the clarity result.
            clarity_result = prediction.strip().strip(".\"'!")
            need_clarify = clarity_result.lower() != "false"
            need_clarify_response = clarity_result if need_clarify else ""

            if need_clarify:
                yield ChatEvent(
                    event_type=ChatEventType.TEXT_PART,
                    payload=need_clarify_response,
                )

            span.end(
                output={
                    "need_clarify": need_clarify,
                    "need_clarify_response": need_clarify_response,
                }
            )

            return need_clarify, need_clarify_response

    def _search_relevance_chunks(
        self, user_question: str
    ) -> Generator[ChatEvent, None, List[NodeWithScore]]:
        with self._trace_manager.span(
            name="search_relevance_chunks", input=user_question
        ) as span:
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.SEARCH_RELATED_DOCUMENTS,
                    display="Retrieving the Most Relevant Documents",
                ),
            )

            relevance_chunks = self.retrieve_flow.search_relevant_chunks(user_question)

            span.end(
                output={
                    "relevance_chunks": relevance_chunks,
                }
            )

            return relevance_chunks

    def _generate_answer(
        self,
        user_question: str,
        knowledge_graph_context: str,
        relevant_chunks: List[NodeWithScore],
    ) -> Generator[ChatEvent, None, Tuple[str, List[SourceDocument]]]:
        with self._trace_manager.span(
            name="generate_answer", input=user_question
        ) as span:
            # Initialize response synthesizer.
            text_qa_template = RichPromptTemplate(
                template_str=self.engine_config.llm.text_qa_prompt
            )
            text_qa_template = text_qa_template.partial_format(
                current_date=datetime.now().strftime("%Y-%m-%d"),
                graph_knowledges=knowledge_graph_context,
                original_question=self.user_question,
            )
            response_synthesizer = get_response_synthesizer(
                llm=self._llm, text_qa_template=text_qa_template, streaming=True
            )

            # Initialize response.
            response = response_synthesizer.synthesize(
                query=user_question,
                nodes=relevant_chunks,
            )
            source_documents = self.retrieve_flow.get_source_documents_from_nodes(
                response.source_nodes
            )
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.SOURCE_NODES,
                    context=source_documents,
                ),
            )

            # Generate response.
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.GENERATE_ANSWER,
                    display="Generating a Precise Answer with AI",
                ),
            )
            response_text = ""
            for word in response.response_gen:
                response_text += word
                yield ChatEvent(
                    event_type=ChatEventType.TEXT_PART,
                    payload=word,
                )

            if not response_text:
                raise Exception("Got empty response from LLM")

            span.end(
                output=response_text,
                metadata={
                    "source_documents": source_documents,
                },
            )

            return response_text, source_documents

    def _post_verification(
        self, user_question: str, response_text: str, chat_id: UUID, message_id: int
    ) -> Optional[str]:
        # post verification to external service, will return the post verification result url
        post_verification_url = self.engine_config.post_verification_url
        post_verification_token = self.engine_config.post_verification_token

        if not post_verification_url:
            return None

        external_request_id = f"{chat_id}_{message_id}"
        qa_content = f"User question: {user_question}\n\nAnswer:\n{response_text}"

        with self._trace_manager.span(
            name="post_verification",
            input={
                "external_request_id": external_request_id,
                "qa_content": qa_content,
            },
        ) as span:
            try:
                resp = requests.post(
                    post_verification_url,
                    json={
                        "external_request_id": external_request_id,
                        "qa_content": qa_content,
                    },
                    headers=(
                        {
                            "Authorization": f"Bearer {post_verification_token}",
                        }
                        if post_verification_token
                        else {}
                    ),
                    timeout=10,
                )
                resp.raise_for_status()
                job_id = resp.json()["job_id"]
                post_verification_link = urljoin(
                    f"{post_verification_url}/", str(job_id)
                )

                span.end(
                    output={
                        "post_verification_link": post_verification_link,
                    }
                )

                return post_verification_link
            except Exception as e:
                logger.exception("Failed to post verification: %s", e.message)
                return None

    def _chat_finish(
        self,
        db_assistant_message: ChatMessage,
        db_user_message: ChatMessage,
        response_text: str,
        knowledge_graph: KnowledgeGraphRetrievalResult = KnowledgeGraphRetrievalResult(),
        source_documents: Optional[List[SourceDocument]] = [],
        annotation_silent: bool = False,
        extra_meta: Optional[dict] = None,
    ):
        if not annotation_silent:
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.FINISHED,
                ),
            )

        post_verification_result_url = self._post_verification(
            self.user_question,
            response_text,
            self.db_chat_obj.id,
            db_assistant_message.id,
        )

        db_assistant_message.sources = [s.model_dump() for s in source_documents]
        db_assistant_message.graph_data = knowledge_graph.to_stored_graph_dict()
        db_assistant_message.content = response_text
        db_assistant_message.post_verification_result_url = post_verification_result_url
        # attach additional meta if provided
        if extra_meta:
            try:
                safe_meta = _json_safe(extra_meta)
                current_meta = getattr(db_assistant_message, "meta", None) or {}
                if isinstance(current_meta, dict):
                    current_meta.update(safe_meta)
                    db_assistant_message.meta = current_meta
                else:
                    db_assistant_message.meta = safe_meta
            except Exception:
                db_assistant_message.meta = _json_safe(extra_meta)
        db_assistant_message.updated_at = datetime.now(UTC)
        db_assistant_message.finished_at = datetime.now(UTC)
        self.db_session.add(db_assistant_message)

        db_user_message.graph_data = knowledge_graph.to_stored_graph_dict()
        db_user_message.updated_at = datetime.now(UTC)
        db_user_message.finished_at = datetime.now(UTC)
        self.db_session.add(db_user_message)
        self.db_session.commit()

        yield ChatEvent(
            event_type=ChatEventType.DATA_PART,
            payload=ChatStreamDataPayload(
                chat=self.db_chat_obj,
                user_message=db_user_message,
                assistant_message=db_assistant_message,
            ),
        )

    # TODO: Separate _external_chat() method into another ExternalChatFlow class, but at the same time, we need to
    #  share some common methods through ChatMixin or BaseChatFlow.
    def _external_chat(self) -> Generator[ChatEvent | str, None, None]:
        ctx = langfuse_instrumentor_context.get().copy()
        db_user_message, db_assistant_message = yield from self._chat_start()
        langfuse_instrumentor_context.get().update(ctx)

        cache_messages = None
        goal, response_format = self.user_question, {}
        if settings.ENABLE_QUESTION_CACHE and len(self.chat_history) == 0:
            try:
                logger.info(
                    f"start to find_best_answer_for_question with question: {self.user_question}"
                )
                cache_messages = chat_repo.find_best_answer_for_question(
                    self.db_session, self.user_question
                )
                if cache_messages and len(cache_messages) > 0:
                    logger.info(
                        f"find_best_answer_for_question result {len(cache_messages)} for question {self.user_question}"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to find best answer for question {self.user_question}: {e}"
                )

        if not cache_messages or len(cache_messages) == 0:
            try:
                # 1. Generate the goal with the user question, knowledge graph and chat history.
                goal, response_format = yield from self._generate_goal()

                # 2. Check if the goal provided enough context information or need to clarify.
                if self.engine_config.clarify_question:
                    (
                        need_clarify,
                        need_clarify_response,
                    ) = yield from self._clarify_question(
                        user_question=goal, chat_history=self.chat_history
                    )
                    if need_clarify:
                        yield from self._chat_finish(
                            db_assistant_message=db_assistant_message,
                            db_user_message=db_user_message,
                            response_text=need_clarify_response,
                            annotation_silent=True,
                        )
                        return
            except Exception as e:
                goal = self.user_question
                logger.warning(
                    f"Failed to generate refined goal, fallback to use user question as goal directly: {e}",
                    exc_info=True,
                    extra={},
                )

            cache_messages = None
            if settings.ENABLE_QUESTION_CACHE:
                try:
                    logger.info(
                        f"start to find_recent_assistant_messages_by_goal with goal: {goal}, response_format: {response_format}"
                    )
                    cache_messages = chat_repo.find_recent_assistant_messages_by_goal(
                        self.db_session,
                        {"goal": goal, "Lang": response_format.get("Lang", "English")},
                        90,
                    )
                    logger.info(
                        f"find_recent_assistant_messages_by_goal result {len(cache_messages)} for goal {goal}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to find recent assistant messages by goal: {e}"
                    )

        stream_chat_api_url = (
            self.engine_config.external_engine_config.stream_chat_api_url
        )
        if cache_messages and len(cache_messages) > 0:
            stackvm_response_text = cache_messages[0].content
            task_id = cache_messages[0].meta.get("task_id")
            for chunk in stackvm_response_text.split(". "):
                if chunk:
                    if not chunk.endswith("."):
                        chunk += ". "
                    yield ChatEvent(
                        event_type=ChatEventType.TEXT_PART,
                        payload=chunk,
                    )
        else:
            logger.debug(
                f"Chatting with external chat engine (api_url: {stream_chat_api_url}) to answer for user question: {self.user_question}"
            )
            chat_params = {
                "goal": goal,
                "response_format": response_format,
                "namespace_name": "Default",
            }
            res = requests.post(stream_chat_api_url, json=chat_params, stream=True)

            # Notice: External type chat engine doesn't support non-streaming mode for now.
            stackvm_response_text = ""
            task_id = None
            for line in res.iter_lines():
                if not line:
                    continue

                # Append to final response text.
                chunk = line.decode("utf-8")
                if chunk.startswith("0:"):
                    word = json.loads(chunk[2:])
                    stackvm_response_text += word
                    yield ChatEvent(
                        event_type=ChatEventType.TEXT_PART,
                        payload=word,
                    )
                else:
                    yield line + b"\n"

                try:
                    if chunk.startswith("8:") and task_id is None:
                        states = json.loads(chunk[2:])
                        if len(states) > 0:
                            # accesss task by http://endpoint/?task_id=$task_id
                            task_id = states[0].get("task_id")
                except Exception as e:
                    logger.error(f"Failed to get task_id from chunk: {e}")

        response_text = stackvm_response_text
        base_url = stream_chat_api_url.replace("/api/stream_execute_vm", "")
        try:
            post_verification_result_url = self._post_verification(
                goal,
                response_text,
                self.db_chat_obj.id,
                db_assistant_message.id,
            )
            db_assistant_message.post_verification_result_url = (
                post_verification_result_url
            )
        except Exception:
            logger.error(
                "Specific error occurred during post verification job.", exc_info=True
            )

        trace_url = f"{base_url}?task_id={task_id}" if task_id else ""
        message_meta = {
            "task_id": task_id,
            "goal": goal,
            **response_format,
        }

        db_assistant_message.content = response_text
        db_assistant_message.trace_url = trace_url
        db_assistant_message.meta = message_meta
        db_assistant_message.updated_at = datetime.now(UTC)
        db_assistant_message.finished_at = datetime.now(UTC)
        self.db_session.add(db_assistant_message)

        db_user_message.trace_url = trace_url
        db_user_message.meta = message_meta
        db_user_message.updated_at = datetime.now(UTC)
        db_user_message.finished_at = datetime.now(UTC)
        self.db_session.add(db_user_message)
        self.db_session.commit()

        yield ChatEvent(
            event_type=ChatEventType.DATA_PART,
            payload=ChatStreamDataPayload(
                chat=self.db_chat_obj,
                user_message=db_user_message,
                assistant_message=db_assistant_message,
            ),
        )

    def _generate_goal(self) -> Generator[ChatEvent, None, Tuple[str, dict]]:
        try:
            refined_question = yield from self._refine_user_question(
                user_question=self.user_question,
                chat_history=self.chat_history,
                refined_question_prompt=self.engine_config.llm.generate_goal_prompt,
                annotation_silent=True,
            )

            goal = refined_question.strip()
            if goal.startswith("Goal: "):
                goal = goal[len("Goal: ") :].strip()
        except Exception as e:
            logger.error(f"Failed to refine question with related knowledge graph: {e}")
            goal = self.user_question

        response_format = {}
        try:
            clean_goal, response_format = parse_goal_response_format(goal)
            logger.info(f"clean goal: {clean_goal}, response_format: {response_format}")
            if clean_goal:
                goal = clean_goal
        except Exception as e:
            logger.error(f"Failed to parse goal and response format: {e}")

        return goal, response_format
