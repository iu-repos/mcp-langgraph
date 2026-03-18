"""LangChain adapter for calling the Pluralsight legacy proxy.

The proxy only understands ``{"prompt": "<string>"} → {"response": "<string>"}``,
so standard OpenAI function-calling cannot be used.  Instead this class
implements **ReAct-style prompting**:

1. Tool schemas + conversation history are embedded in the prompt string.
2. The model is instructed to output either:
   - A JSON block ``{"tool_call": {"name": "...", "args": {...}}}`` when a
     tool is needed, OR
   - Plain text for the final answer.
3. The response is parsed: detected tool calls are converted to a proper
   ``AIMessage(tool_calls=[...])`` that LangGraph's ``ToolNode`` can execute.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from .proxy_llm import call_llm

# Matches a JSON object that contains a top-level "tool_call" key,
# even if surrounded by markdown code fences or extra whitespace.
_TOOL_CALL_RE = re.compile(
    r'```(?:json)?\s*({\s*"tool_call"\s*:.*?})\s*```|({\s*"tool_call"\s*:.*?})\s*$',
    re.DOTALL,
)

_SYSTEM_PROMPT = """\
You are a helpful assistant with access to the following tools:

{tool_descriptions}

RULES:
- If the user request requires a tool, respond with ONLY valid JSON in this exact format (no extra text):
  {{"tool_call": {{"name": "<tool_name>", "args": {{<arg_name>: <value>, ...}}}}}}
- If you have a final answer, respond with plain text only.
- Never mix tool-call JSON with explanatory text.
"""


class ProxyChatModel(BaseChatModel):
    """ChatModel that uses ReAct-style prompting for a legacy string-only proxy."""

    model_name: str = "proxy"
    bound_tools: List[BaseTool] = []

    def __init__(self, model_name: Optional[str] = None, **kwargs: Any):
        super().__init__(**kwargs)
        if model_name is not None:
            self.model_name = model_name

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        tool_choice: Optional[Any] = None,
        **kwargs: Any,
    ) -> "ProxyChatModel":
        new_model = ProxyChatModel(model_name=self.model_name)
        new_model.bound_tools = list(tools)
        return new_model

    @property
    def _llm_type(self) -> str:
        return "proxy"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "bound_tools": len(self.bound_tools)}

    def _tool_descriptions(self) -> str:
        lines: list[str] = []
        for tool in self.bound_tools:
            params = ", ".join(
                f"{k}: {v.get('type', 'any')}" for k, v in (tool.args or {}).items()
            )
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines) if lines else "(none)"

    def _build_prompt(self, messages: List[BaseMessage]) -> str:
        """Combine system instructions, tool descriptions, and history into one string prompt."""
        parts: list[str] = []

        if self.bound_tools:
            parts.append(
                _SYSTEM_PROMPT.format(tool_descriptions=self._tool_descriptions())
            )

        for msg in messages:
            if isinstance(msg, HumanMessage):
                parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    tc = msg.tool_calls[0]
                    parts.append(
                        f'Assistant: {{"tool_call": {{"name": "{tc["name"]}", "args": {json.dumps(tc["args"])}}}}}'
                    )
                else:
                    parts.append(f"Assistant: {msg.content}")
            elif isinstance(msg, ToolMessage):
                parts.append(f"Tool result: {msg.content}")
            else:
                parts.append(f"User: {msg.content}")

        parts.append("Assistant:")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_response(text: str) -> AIMessage:
        """Return an AIMessage, with tool_calls populated if a tool-call JSON is detected."""
        stripped = text.strip()

        # Try regex match first (handles markdown fences)
        m = _TOOL_CALL_RE.search(stripped)
        raw_json = (m.group(1) or m.group(2)).strip() if m else None

        # Fallback: try direct JSON parse if text looks like an object
        if raw_json is None and stripped.startswith("{"):
            raw_json = stripped

        if raw_json:
            try:
                parsed = json.loads(raw_json)
                if tc := parsed.get("tool_call"):
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": str(uuid.uuid4()),
                                "name": tc["name"],
                                "args": tc.get("args", {}),
                                "type": "tool_call",
                            }
                        ],
                    )
            except (json.JSONDecodeError, KeyError):
                pass  # not a valid tool-call blob → treat as plain text

        return AIMessage(content=text)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._build_prompt(messages)
        text = call_llm(prompt)
        ai_message = self._parse_response(text)
        return ChatResult(
            generations=[ChatGeneration(message=ai_message)],
            llm_output={"model_name": self.model_name},
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages=messages, stop=stop, **kwargs)


class ProxyChatModel(BaseChatModel):
    """ChatModel implementation that calls an OpenAI-compatible proxy.

    Converts LangChain messages and bound tools to the standard OpenAI
    ``messages`` + ``tools`` payload so the model can decide when to call a
    tool.  Handles both the standard ``choices`` response format and the
    legacy ``{"response": "..."}`` proxy format (which does not support tool
    calling).
    """

    model_name: str = "proxy"
    bound_tools: List[BaseTool] = []

    def __init__(self, model_name: Optional[str] = None, **kwargs: Any):
        super().__init__(**kwargs)
        if model_name is not None:
            self.model_name = model_name

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        tool_choice: Optional[Any] = None,
        **kwargs: Any,
    ) -> "ProxyChatModel":
        """Bind tools to the model."""
        new_model = ProxyChatModel(model_name=self.model_name)
        new_model.bound_tools = list(tools)
        return new_model

    @property
    def _llm_type(self) -> str:
        return "proxy"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "bound_tools": len(self.bound_tools)}

    @staticmethod
    def _messages_to_openai(messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to OpenAI API format."""
        result = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                entry: dict = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                result.append(entry)
            elif isinstance(msg, ToolMessage):
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": str(msg.content),
                    }
                )
            else:
                result.append({"role": "user", "content": str(msg.content)})
        return result

    @staticmethod
    def _tools_to_openai(tools: List[BaseTool]) -> List[dict]:
        """Convert LangChain tools to OpenAI API function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.args,
                },
            }
            for tool in tools
        ]

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        openai_messages = self._messages_to_openai(messages)
        openai_tools = (
            self._tools_to_openai(self.bound_tools) if self.bound_tools else None
        )
        data = call_llm(openai_messages, openai_tools)

        # Standard OpenAI choices format — tool_calls will be present here
        if choices := data.get("choices"):
            msg = choices[0]["message"]
            content = msg.get("content") or ""
            tool_calls_raw = msg.get("tool_calls") or []
            if tool_calls_raw:
                tool_calls = [
                    {
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "args": json.loads(tc["function"]["arguments"]),
                        "type": "tool_call",
                    }
                    for tc in tool_calls_raw
                ]
                ai_message = AIMessage(content=content, tool_calls=tool_calls)
            else:
                ai_message = AIMessage(content=content)
            return ChatResult(
                generations=[ChatGeneration(message=ai_message)],
                llm_output={"model_name": self.model_name},
            )

        # Legacy proxy format — plain text, no tool calling support
        if "response" in data:
            return ChatResult(
                generations=[
                    ChatGeneration(message=AIMessage(content=data["response"]))
                ],
                llm_output={"model_name": self.model_name},
            )

        raise RuntimeError(f"unexpected response format from proxy: {data}")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages=messages, stop=stop, **kwargs)
