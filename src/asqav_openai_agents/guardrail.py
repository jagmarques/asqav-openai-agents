"""Optional Asqav input guardrail for the OpenAI Agents SDK.

Signs an ``input:check`` event and can stop a run before the agent acts by
tripping the guardrail tripwire when a caller-supplied predicate matches.
Unlike the hooks (which are fail-open observability), this is the blocking
surface: a tripped tripwire raises ``InputGuardrailTripwireTriggered`` inside
the SDK and the agent never runs."""

from __future__ import annotations

import logging
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


class _AsqavGuardrail(AsqavAdapter):
    """Holds the Asqav agent and the caller's block predicate."""

    def __init__(
        self,
        block_if: Callable[[str], bool],
        *,
        api_key: str | None = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(api_key=api_key, agent_name=agent_name, agent_id=agent_id)
        self._block_if = block_if

    async def __call__(
        self,
        context: RunContextWrapper[Any],
        agent: Any,
        input: Any,
    ) -> GuardrailFunctionOutput:
        text = input if isinstance(input, str) else str(input)
        blocked = False
        try:
            blocked = bool(self._block_if(text))
        except Exception as exc:
            logger.warning("asqav guardrail predicate failed (fail-open): %s", exc)
        try:
            self._sign_action(
                "input:check",
                {
                    "agent": getattr(agent, "name", type(agent).__name__),
                    "input": text[:_MAX_LEN],
                    "blocked": blocked,
                },
            )
        except Exception as exc:
            logger.warning("asqav input:check signing failed (fail-open): %s", exc)
        return GuardrailFunctionOutput(
            output_info={"blocked": blocked},
            tripwire_triggered=blocked,
        )


def asqav_input_guardrail(
    block_if: Callable[[str], bool],
    *,
    api_key: str | None = None,
    agent_name: str | None = None,
    agent_id: str | None = None,
) -> InputGuardrail[Any]:
    """Build an ``InputGuardrail`` that signs each checked input and trips the
    tripwire (blocking the run) when ``block_if(input_text)`` returns True.

    Pass the result to ``Agent(..., input_guardrails=[asqav_input_guardrail(...)])``.

    Args:
        block_if: Predicate over the input text; returning True blocks the run.
        api_key: Optional API key override (uses ``asqav.init()`` default).
        agent_name: Name for an Asqav agent (calls ``Agent.create``).
        agent_id: ID of an existing Asqav agent (calls ``Agent.get``).
    """
    guard = _AsqavGuardrail(
        block_if,
        api_key=api_key,
        agent_name=agent_name,
        agent_id=agent_id,
    )
    return InputGuardrail(guardrail_function=guard, name="asqav_input_guardrail")
