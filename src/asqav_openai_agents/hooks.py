"""OpenAI Agents SDK hooks that sign tool:start and tool:end events via the
Asqav API. All signing is fail-open. See README for usage."""

from __future__ import annotations

import logging
from typing import Any

from asqav.extras._base import AsqavAdapter

try:
    from agents import AgentHooks, RunHooks
    from agents.run_context import RunContextWrapper
    from agents.tool import Tool
except ImportError as err:
    raise ImportError(
        "asqav-openai-agents requires the openai-agents SDK. "
        "Install with: pip install asqav-openai-agents"
    ) from err

logger = logging.getLogger("asqav")


class _AsqavSigningMixin(AsqavAdapter):
    """Shared signing logic for the run-level and agent-level hook classes.

    Both ``on_tool_start`` and ``on_tool_end`` are async to match the SDK's
    hook protocol, but ``_sign_action`` itself is a fast fail-open call.
    """

    async def _sign_tool_start(self, agent: Any, tool: Tool) -> None:
        try:
            self._sign_action(
                "tool:start",
                {
                    "tool": getattr(tool, "name", type(tool).__name__),
                    "agent": getattr(agent, "name", type(agent).__name__),
                },
            )
        except Exception as exc:
            logger.warning("asqav tool:start signing failed (fail-open): %s", exc)

    async def _sign_tool_end(self, agent: Any, tool: Tool, result: object) -> None:
        try:
            self._sign_action(
                "tool:end",
                {
                    "tool": getattr(tool, "name", type(tool).__name__),
                    "agent": getattr(agent, "name", type(agent).__name__),
                    "output_type": type(result).__name__,
                    "output_length": len(str(result)),
                },
            )
        except Exception as exc:
            logger.warning("asqav tool:end signing failed (fail-open): %s", exc)


class AsqavRunHooks(_AsqavSigningMixin, RunHooks):
    """Run-level ``RunHooks`` that sign every tool call across all agents in a
    run. Pass via ``Runner.run(..., hooks=AsqavRunHooks(agent_name="..."))``.

    Args:
        api_key: Optional API key override (uses ``asqav.init()`` default).
        agent_name: Name for an Asqav agent (calls ``Agent.create``).
        agent_id: ID of an existing Asqav agent (calls ``Agent.get``).
    """

    async def on_tool_start(
        self,
        context: RunContextWrapper[Any],
        agent: Any,
        tool: Tool,
    ) -> None:
        """Sign ``tool:start`` with tool and agent name."""
        await self._sign_tool_start(agent, tool)

    async def on_tool_end(
        self,
        context: RunContextWrapper[Any],
        agent: Any,
        tool: Tool,
        result: object,
    ) -> None:
        """Sign ``tool:end`` with output metadata."""
        await self._sign_tool_end(agent, tool, result)


class AsqavAgentHooks(_AsqavSigningMixin, AgentHooks):
    """Agent-level ``AgentHooks`` that sign tool calls for a single agent.
    Assign to ``Agent(..., hooks=AsqavAgentHooks(agent_name="..."))``.

    Args:
        api_key: Optional API key override (uses ``asqav.init()`` default).
        agent_name: Name for an Asqav agent (calls ``Agent.create``).
        agent_id: ID of an existing Asqav agent (calls ``Agent.get``).
    """

    async def on_tool_start(
        self,
        context: RunContextWrapper[Any],
        agent: Any,
        tool: Tool,
    ) -> None:
        """Sign ``tool:start`` with tool and agent name."""
        await self._sign_tool_start(agent, tool)

    async def on_tool_end(
        self,
        context: RunContextWrapper[Any],
        agent: Any,
        tool: Tool,
        result: object,
    ) -> None:
        """Sign ``tool:end`` with output metadata."""
        await self._sign_tool_end(agent, tool, result)
