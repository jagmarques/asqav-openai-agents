"""Tests for asqav-openai-agents hooks and guardrail. All Asqav calls are
mocked; no network access occurs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_asqav():
    """Mock asqav.init() and Agent so no real API calls are made."""
    mock_agent = MagicMock()
    mock_agent.sign.return_value = MagicMock(signature="mock-sig", timestamp=1.0)
    with (
        patch("asqav.client._api_key", "sk_test_key"),
        patch("asqav.client.Agent.create", return_value=mock_agent),
        patch("asqav.client.Agent.get", return_value=mock_agent),
    ):
        yield mock_agent


def _tool(name="search"):
    t = MagicMock()
    t.name = name
    return t


def _agent(name="assistant"):
    a = MagicMock()
    a.name = name
    return a


class TestAsqavRunHooks:
    def test_init_creates_named_agent(self, mock_asqav: MagicMock):
        from asqav.client import Agent

        from asqav_openai_agents import AsqavRunHooks

        AsqavRunHooks(agent_name="test-agent")
        Agent.create.assert_called_once_with("test-agent")

    def test_is_run_hooks(self, mock_asqav: MagicMock):
        from agents import RunHooks

        from asqav_openai_agents import AsqavRunHooks

        assert isinstance(AsqavRunHooks(agent_name="t"), RunHooks)

    @pytest.mark.asyncio
    async def test_on_tool_start_signs(self, mock_asqav: MagicMock):
        from asqav_openai_agents import AsqavRunHooks

        hooks = AsqavRunHooks(agent_name="test-agent")
        with patch.object(hooks, "_sign_action", wraps=hooks._sign_action) as spy:
            await hooks.on_tool_start(MagicMock(), _agent(), _tool("search"))
            spy.assert_called_once()
            assert spy.call_args.args[0] == "tool:start"
            assert spy.call_args.args[1]["tool"] == "search"

    @pytest.mark.asyncio
    async def test_on_tool_end_signs(self, mock_asqav: MagicMock):
        from asqav_openai_agents import AsqavRunHooks

        hooks = AsqavRunHooks(agent_name="test-agent")
        with patch.object(hooks, "_sign_action", wraps=hooks._sign_action) as spy:
            await hooks.on_tool_end(MagicMock(), _agent(), _tool(), "result text")
            spy.assert_called_once()
            assert spy.call_args.args[0] == "tool:end"
            assert spy.call_args.args[1]["output_type"] == "str"

    @pytest.mark.asyncio
    async def test_fail_open(self, mock_asqav: MagicMock):
        from asqav_openai_agents import AsqavRunHooks

        hooks = AsqavRunHooks(agent_name="test-agent")
        with patch.object(hooks, "_sign_action", side_effect=RuntimeError("boom")):
            # Must not raise.
            await hooks.on_tool_start(MagicMock(), _agent(), _tool())
            await hooks.on_tool_end(MagicMock(), _agent(), _tool(), "out")


class TestAsqavAgentHooks:
    def test_is_agent_hooks(self, mock_asqav: MagicMock):
        from agents import AgentHooks

        from asqav_openai_agents import AsqavAgentHooks

        assert isinstance(AsqavAgentHooks(agent_name="t"), AgentHooks)

    @pytest.mark.asyncio
    async def test_on_tool_start_signs(self, mock_asqav: MagicMock):
        from asqav_openai_agents import AsqavAgentHooks

        hooks = AsqavAgentHooks(agent_name="test-agent")
        with patch.object(hooks, "_sign_action", wraps=hooks._sign_action) as spy:
            await hooks.on_tool_start(MagicMock(), _agent(), _tool("calc"))
            assert spy.call_args.args[1]["tool"] == "calc"


class TestAsqavInputGuardrail:
    def test_builds_input_guardrail(self, mock_asqav: MagicMock):
        from agents import InputGuardrail

        from asqav_openai_agents import asqav_input_guardrail

        g = asqav_input_guardrail(lambda t: False, agent_name="test-agent")
        assert isinstance(g, InputGuardrail)

    @pytest.mark.asyncio
    async def test_blocks_and_signs_when_predicate_true(self, mock_asqav: MagicMock):
        from asqav_openai_agents import asqav_input_guardrail

        g = asqav_input_guardrail(lambda t: "bad" in t, agent_name="test-agent")
        guard_fn = g.guardrail_function
        with patch.object(guard_fn, "_sign_action", wraps=guard_fn._sign_action) as spy:
            out = await guard_fn(MagicMock(), _agent(), "this is bad")
            assert out.tripwire_triggered is True
            assert spy.call_args.args[0] == "input:check"
            assert spy.call_args.args[1]["blocked"] is True

    @pytest.mark.asyncio
    async def test_allows_and_signs_when_predicate_false(self, mock_asqav: MagicMock):
        from asqav_openai_agents import asqav_input_guardrail

        g = asqav_input_guardrail(lambda t: False, agent_name="test-agent")
        out = await g.guardrail_function(MagicMock(), _agent(), "all good")
        assert out.tripwire_triggered is False
