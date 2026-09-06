"""Tests for asqav-openai-agents hooks and guardrail. All Asqav calls are
mocked; no network access occurs."""

from __future__ import annotations

import asyncio
import socket
from typing import get_origin
from unittest.mock import MagicMock, patch

import pytest
from agents import (
    Agent,
    FunctionTool,
    InputGuardrailTripwireTriggered,
    Model,
    RunConfig,
    Runner,
)
from agents.items import ModelResponse
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)


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

        # RunHooks is a subscripted generic on newer SDK releases; isinstance
        # needs the unparameterized origin class.
        run_hooks_class = get_origin(RunHooks) or RunHooks
        assert isinstance(AsqavRunHooks(agent_name="t"), run_hooks_class)

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

        # AgentHooks is a subscripted generic on newer SDK releases; isinstance
        # needs the unparameterized origin class.
        agent_hooks_class = get_origin(AgentHooks) or AgentHooks
        assert isinstance(AsqavAgentHooks(agent_name="t"), agent_hooks_class)

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

    @pytest.mark.asyncio
    async def test_fails_closed_when_predicate_raises(self, mock_asqav: MagicMock):
        # A raising decision function must DENY (trip the tripwire) by default,
        # not silently allow the run through (the fail-open bypass).
        from asqav_openai_agents import asqav_input_guardrail

        def boom(_text: str) -> bool:
            raise ValueError("predicate blew up")

        g = asqav_input_guardrail(boom, agent_name="test-agent")
        out = await g.guardrail_function(MagicMock(), _agent(), "anything")
        assert out.tripwire_triggered is True

    @pytest.mark.asyncio
    async def test_fail_closed_receipt_records_predicate_error(self, mock_asqav: MagicMock):
        from asqav_openai_agents import asqav_input_guardrail

        def boom(_text: str) -> bool:
            raise ValueError("predicate blew up")

        g = asqav_input_guardrail(boom, agent_name="test-agent")
        guard_fn = g.guardrail_function
        with patch.object(guard_fn, "_sign_action", wraps=guard_fn._sign_action) as spy:
            out = await guard_fn(MagicMock(), _agent(), "anything")
            assert out.tripwire_triggered is True
            assert spy.call_args.args[0] == "input:check"
            assert spy.call_args.args[1]["blocked"] is True
            assert "predicate_error" in spy.call_args.args[1]

    @pytest.mark.asyncio
    async def test_fail_open_opt_allows_when_predicate_raises(self, mock_asqav: MagicMock):
        # Back-compat escape hatch: fail_closed=False restores the old allow-through.
        from asqav_openai_agents import asqav_input_guardrail

        def boom(_text: str) -> bool:
            raise ValueError("predicate blew up")

        g = asqav_input_guardrail(boom, agent_name="test-agent", fail_closed=False)
        out = await g.guardrail_function(MagicMock(), _agent(), "anything")
        assert out.tripwire_triggered is False


class TestEventLoopNotBlocked:
    """The sign round trip is blocking HTTP, so async surfaces must offload it."""

    @pytest.mark.asyncio
    async def test_hooks_and_guardrail_sign_off_the_event_loop(self, mock_asqav: MagicMock):
        # Signing must run on a worker thread, never the event-loop thread, so a
        # slow sign cannot stall the loop for every other coroutine.
        import threading

        from asqav_openai_agents import AsqavRunHooks, asqav_input_guardrail

        loop_thread = threading.get_ident()
        threads: list[int] = []

        def spy(*_args, **_kwargs):
            threads.append(threading.get_ident())
            return MagicMock(signature="sig", timestamp=1.0)

        hooks = AsqavRunHooks(agent_name="test-agent")
        with patch.object(hooks, "_sign_action", side_effect=spy):
            await hooks.on_tool_start(MagicMock(), _agent(), _tool())
            await hooks.on_tool_end(MagicMock(), _agent(), _tool(), "out")

        guard_fn = asqav_input_guardrail(
            lambda _t: False, agent_name="test-agent"
        ).guardrail_function
        with patch.object(guard_fn, "_sign_action", side_effect=spy):
            await guard_fn(MagicMock(), _agent(), "hello")

        assert len(threads) == 3
        assert all(tid != loop_thread for tid in threads)


class TestGuardrailInputHandling:
    """Real-text extraction and async-predicate support for the guardrail."""

    @pytest.mark.asyncio
    async def test_message_list_feeds_real_text_not_repr(self, mock_asqav: MagicMock):
        # A message-list input must reach the predicate as the user text, not the
        # list's Python repr, which silently defeats exact/regex/json checks.
        from asqav_openai_agents import asqav_input_guardrail

        seen: dict[str, str] = {}

        def block_if(t: str) -> bool:
            seen["text"] = t
            return t == "TRANSFER ALL FUNDS"

        g = asqav_input_guardrail(block_if, agent_name="test-agent")
        out = await g.guardrail_function(
            MagicMock(), _agent(), [{"role": "user", "content": "TRANSFER ALL FUNDS"}]
        )
        assert seen["text"] == "TRANSFER ALL FUNDS"
        assert out.tripwire_triggered is True

        parts = [{"type": "input_text", "text": "TRANSFER ALL FUNDS"}]
        out_parts = await g.guardrail_function(
            MagicMock(), _agent(), [{"role": "user", "content": parts}]
        )
        assert seen["text"] == "TRANSFER ALL FUNDS"
        assert out_parts.tripwire_triggered is True

    @pytest.mark.asyncio
    async def test_async_block_if_is_awaited(self, mock_asqav: MagicMock):
        # An async predicate must be awaited. bool(coroutine) is always truthy,
        # so the pre-fix path over-blocks every input.
        from asqav_openai_agents import asqav_input_guardrail

        async def allow(_t: str) -> bool:
            return False

        async def deny(_t: str) -> bool:
            return True

        g_allow = asqav_input_guardrail(allow, agent_name="test-agent")
        out_allow = await g_allow.guardrail_function(MagicMock(), _agent(), "hello")
        assert out_allow.tripwire_triggered is False

        g_deny = asqav_input_guardrail(deny, agent_name="test-agent")
        out_deny = await g_deny.guardrail_function(MagicMock(), _agent(), "hello")
        assert out_deny.tripwire_triggered is True


class _ToolCallingModel(Model):
    """Local model double; the real Runner dispatches its requested tool call."""

    def __init__(self, events: list[str]):
        self.events = events
        self.calls = 0

    async def get_response(self, *args, **kwargs):
        self.calls += 1
        self.events.append("model")
        if self.calls == 1:
            output = [ResponseFunctionToolCall(
                id="tool_call_1", call_id="call_1", name="record_effect",
                arguments="{}", type="function_call",
            )]
        else:
            output = [ResponseOutputMessage(
                id="message_1", role="assistant", status="completed", type="message",
                content=[ResponseOutputText(text="done", annotations=[], type="output_text")],
            )]
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    def stream_response(self, *args, **kwargs):
        raise AssertionError("This test uses Runner.run, not streaming")


class TestRunnerInputGuardrail:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("signing_fails", [False, True])
    @pytest.mark.parametrize("mode,fail_closed,blocked", [
        ("deny", True, True),
        ("permit", True, False),
        ("error", True, True),
        ("error", False, False),
    ])
    async def test_slow_predicate_finishes_before_model_or_tool_execution(
        self, mock_asqav, monkeypatch, mode, fail_closed, blocked, signing_fails,
    ):
        from asqav_openai_agents import asqav_input_guardrail

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        def reject_network(*_args, **_kwargs):
            raise AssertionError("Runner ordering tests must not access the network")

        monkeypatch.setattr(socket.socket, "connect", reject_network)
        events: list[str] = []
        tool_calls = 0

        async def predicate(_text: str) -> bool:
            events.append("predicate:start")
            # Give a speculative model call time to start if blocking mode is removed.
            await asyncio.sleep(0.03)
            events.append("predicate:finish")
            if mode == "error":
                raise ValueError("predicate unavailable")
            return mode == "deny"

        async def record_effect(_context, _arguments):
            nonlocal tool_calls
            tool_calls += 1
            events.append("tool")
            return "recorded"

        if signing_fails:
            mock_asqav.sign.side_effect = RuntimeError("signer unavailable")
        model = _ToolCallingModel(events)
        tool = FunctionTool(
            name="record_effect", description="Record a test side effect",
            params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
            on_invoke_tool=record_effect,
        )
        agent = Agent(
            name="guarded", model=model, tools=[tool],
            input_guardrails=[asqav_input_guardrail(
                predicate, agent_name="audit-agent", fail_closed=fail_closed,
            )],
        )
        config = RunConfig(tracing_disabled=True)
        if blocked:
            with pytest.raises(InputGuardrailTripwireTriggered):
                await Runner.run(agent, "test input", run_config=config)
            assert model.calls == 0
            assert tool_calls == 0
            assert events == ["predicate:start", "predicate:finish"]
        else:
            result = await Runner.run(agent, "test input", run_config=config)
            assert result.final_output == "done"
            assert model.calls == 2
            assert tool_calls == 1
            assert events == ["predicate:start", "predicate:finish", "model", "tool", "model"]
        mock_asqav.sign.assert_called_once()
