"""
LLM wrapper that runs the Claude CLI on the host.
Uses the same authorization as the CLI (e.g. after `claude auth login`); no API keys in the app.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Any, AsyncGenerator, Generator, Optional, Sequence

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)

logger = logging.getLogger(__name__)


def _messages_to_prompt(messages: Sequence[ChatMessage]) -> str:
    """Convert chat messages to a single prompt string for the CLI."""
    parts = []
    for msg in messages:
        role = getattr(msg.role, "value", str(msg.role)) if hasattr(msg.role, "value") else str(msg.role)
        content = msg.content if isinstance(msg.content, str) else (msg.content or "")
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def _run_cli(prompt: str, cli_path: str, agent_id: Optional[str]) -> str:
    """Run Claude CLI with prompt on stdin; return stdout. Uses CLI's own auth."""
    cmd: list[str]
    if agent_id and agent_id.strip():
        cmd = [cli_path, "agent", "run", agent_id.strip()]
    else:
        cmd = [cli_path, "run"]
    try:
        result = subprocess.run(
            cmd,
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=300,
            env=None,  # inherit env so CLI can use its stored credentials
        )
        out = (result.stdout or b"").decode("utf-8", errors="replace").strip()
        if result.returncode != 0:
            err = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            logger.warning("Claude CLI exited with code %s: %s", result.returncode, err)
            if not out and err:
                out = f"[CLI error {result.returncode}]: {err}"
        return out
    except FileNotFoundError:
        logger.exception("Claude CLI not found: %s", cli_path)
        return f"[Error: Claude CLI not found at '{cli_path}'. Install the CLI and run `claude auth login`.]"
    except subprocess.TimeoutExpired:
        logger.exception("Claude CLI timed out")
        return "[Error: Claude CLI timed out.]"
    except Exception as e:
        logger.exception("Claude CLI failed: %s", e)
        return f"[Error: {e}]"


class ClaudeCLILLM(BaseLLM):
    """
    LlamaIndex LLM that delegates to the Claude CLI on the host.
    Credentials are handled by the CLI (e.g. `claude auth login`); no API key in the app.
    """

    model: str = ""
    cli_path: str = "claude"
    agent_id: str = ""

    @classmethod
    def class_name(cls) -> str:
        return "ClaudeCLILLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model or "claude-cli",
            context_window=200_000,
            num_output=8192,
            is_chat_model=True,
            is_function_calling_model=False,
        )

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        text = _run_cli(prompt, self.cli_path, self.agent_id or None)
        return CompletionResponse(text=text)

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = _messages_to_prompt(messages)
        text = _run_cli(prompt, self.cli_path, self.agent_id or None)
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=text)
        )

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        text = _run_cli(prompt, self.cli_path, self.agent_id or None)
        yield CompletionResponse(text=text, delta=text)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = _messages_to_prompt(messages)
        text = _run_cli(prompt, self.cli_path, self.agent_id or None)
        yield ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
            delta=text,
        )

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.complete(prompt, formatted=formatted, **kwargs),
        )

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.chat(messages, **kwargs),
        )

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: self.complete(prompt, formatted=formatted, **kwargs),
        )
        yield resp

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: self.chat(messages, **kwargs),
        )
        yield resp
