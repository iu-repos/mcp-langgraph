"""LangChain adapter for calling an OpenAI-compatible LLM proxy.

This allows the existing LangGraph tooling (which expects a LangChain
``ChatModel`` with ``bind_tools`` and ``invoke``) to remain unchanged while
routing LLM requests through a simple HTTP proxy.

The proxy wrapper is implemented in ``proxy_llm.py`` and reads its
configuration from environment variables (typically via a ``.env`` file).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable

from .proxy_llm import call_llm


class ProxyChatModel(BaseChatModel):
    """ChatModel implementation that calls an OpenAI-compatible proxy.

    This is intentionally minimal: it converts the incoming LangChain message
    history into a text prompt, sends it to the proxy via ``call_llm``, and
    returns the text as an ``AIMessage`` inside a ``ChatResult``.
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

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        # Simple prompt conversion: prefix each message with its role.
        lines: List[str] = []
        for msg in messages:
            role = getattr(msg, "type", None) or getattr(msg, "role", "user")
            content = getattr(msg, "content", "")
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._messages_to_prompt(messages)
        text = call_llm(prompt)
        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(
            generations=[generation], llm_output={"model_name": self.model_name}
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Provide a simple async wrapper around the sync call.
        return self._generate(messages=messages, stop=stop, **kwargs)
