"""Optional Asqav input guardrail for the OpenAI Agents SDK.

Signs an ``input:check`` event and can stop a run before the agent acts by
tripping the guardrail tripwire when a caller-supplied predicate matches.
Unlike the hooks (which are fail-open observability), this is the blocking
surface: a tripped tripwire raises ``InputGuardrailTripwireTriggered`` inside
the SDK and the agent never runs."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable
from typing import Any, Callable

from asqav.extras._base import AsqavAdapter

try:
    from agents import (
        GuardrailFunctionOutput,
        InputGuardrail,
        RunContextWrapper,
    )
except ImportError as err:
    raise ImportError(
        "asqav-openai-agents requires the openai-agents SDK. "
        "Install with: pip install asqav-openai-agents"
    ) from err

logger = logging.getLogger("asqav")

_MAX_LEN = 200


def _field(obj: Any, name: str) -> Any:
    """Read ``name`` from a dict item or fall back to an object attribute."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _content_text(content: Any) -> str:
    """Text of one message ``content``: a plain string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [t for t in (_field(p, "text") for p in content) if isinstance(t, str)]
        return "\n".join(parts)
    return ""


def _input_to_text(value: Any) -> str:
    """Flatten a guardrail input to the text the predicate should see.

    The SDK passes ``str | list[TResponseInputItem]``. For a message list, feed
    the real message text to the predicate, never the list's Python repr.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks = [_content_text(_field(item, "content")) for item in value]
        return "\n".join(c for c in chunks if c)
    return str(value)


class _AsqavGuardrail(AsqavAdapter):
    """Holds the Asqav agent, the caller's block predicate, and fail-closed policy."""

    def __init__(
        self,
        block_if: Callable[[str], bool | Awaitable[bool]],
        *,
        api_key: str | None = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
        fail_closed: bool = True,
    ) -> None:
        super().__init__(api_key=api_key, agent_name=agent_name, agent_id=agent_id)
        self._block_if = block_if
        self._fail_closed = fail_closed

    async def __call__(
        self,
        context: RunContextWrapper[Any],
        agent: Any,
        input: Any,
    ) -> GuardrailFunctionOutput:
        text = _input_to_text(input)
        predicate_error: str | None = None
        try:
            decision = self._block_if(text)
            if inspect.isawaitable(decision):
                decision = await decision
            blocked = bool(decision)
        except Exception as exc:
            # The decision function itself errored. Fail closed by default so an
            # adversarial input cannot slip past by making the predicate throw.
            predicate_error = repr(exc)
            blocked = self._fail_closed
            logger.warning(
                "asqav guardrail predicate failed (%s): %s",
                "fail-closed" if self._fail_closed else "fail-open",
                exc,
            )
        payload: dict[str, Any] = {
            "agent": getattr(agent, "name", type(agent).__name__),
            "input": text[:_MAX_LEN],
            "blocked": blocked,
        }
        if predicate_error is not None:
            payload["predicate_error"] = predicate_error
        try:
            await self._sign_action_async("input:check", payload)
        except Exception as exc:
            logger.warning("asqav input:check signing failed (fail-open): %s", exc)
        return GuardrailFunctionOutput(
            output_info={"blocked": blocked},
            tripwire_triggered=blocked,
        )


def asqav_input_guardrail(
    block_if: Callable[[str], bool | Awaitable[bool]],
    *,
    api_key: str | None = None,
    agent_name: str | None = None,
    agent_id: str | None = None,
    fail_closed: bool = True,
) -> InputGuardrail[Any]:
    """Build an ``InputGuardrail`` that signs each checked input and trips the
    tripwire (blocking the run) when ``block_if(input_text)`` returns True.

    Pass the result to ``Agent(..., input_guardrails=[asqav_input_guardrail(...)])``.

    Args:
        block_if: Predicate over the input text; returning True blocks the run.
            May be sync or async; an async predicate is awaited.
        api_key: Optional API key override (uses ``asqav.init()`` default).
        agent_name: Name for an Asqav agent (calls ``Agent.create``).
        agent_id: ID of an existing Asqav agent (calls ``Agent.get``).
        fail_closed: When ``block_if`` raises, block the run (trip the tripwire)
            instead of allowing it, and record the error on the signed receipt.
            Defaults to True. Set False to keep the old allow-on-error behaviour.
    """
    guard = _AsqavGuardrail(
        block_if,
        api_key=api_key,
        agent_name=agent_name,
        agent_id=agent_id,
        fail_closed=fail_closed,
    )
    return InputGuardrail(guardrail_function=guard, name="asqav_input_guardrail")
