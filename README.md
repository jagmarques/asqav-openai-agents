<p align="center">
  <a href="https://asqav.com">
    <img src="https://asqav.com/logo-text-white.png" alt="Asqav" width="200">
  </a>
</p>
<p align="center">
  Record tool-call events and check input before an agent starts.
</p>
<p align="center">
  <a href="https://www.asqav.com/">Website</a> |
  <a href="https://www.asqav.com/docs">Docs</a> |
  <a href="https://github.com/jagmarques/asqav-sdk">SDK</a>
</p>

# Asqav for the OpenAI Agents SDK

Record tool-call events and check input before an agent starts.

`asqav-openai-agents` connects [Asqav](https://asqav.com) to the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). Hooks submit tool-call events for signing, and an input guardrail checks your predicate before the first agent's model call. A signed event records what the integration reported; it does not independently establish what an agent did.

Asqav governs the agents you wire through it. An agent that never routes through the governed path produces no receipt and is not detected.

This package gives you two surfaces:

- **Hooks**, namely `AsqavRunHooks` and `AsqavAgentHooks`, attempt to sign `tool:start` and `tool:end` on the SDK's documented `RunHooks` and `AgentHooks` lifecycle. They observe and record, and they are fail-open: they never block tool execution.
- **Guardrail**, `asqav_input_guardrail`, checks the initial input before the first agent's model call. It attempts to sign an `input:check` event and trips the SDK tripwire when your predicate matches.

## Install

Install the source version from GitHub:

```bash
pip install "asqav-openai-agents[agents] @ git+https://github.com/jagmarques/asqav-openai-agents.git"
```

Or install this revision from a local checkout:

```bash
pip install ".[agents]"
```

These examples describe the source version. It requires Asqav 0.10.10 or later within the 0.10 series. The `[agents]` extra installs OpenAI Agents SDK 0.22.0 or later. You can omit the extra if you already have a compatible `openai-agents` installation.

## Usage: record tool-call events

```python
import asqav
from agents import Agent, Runner

from asqav_openai_agents import AsqavRunHooks

asqav.init(api_key="sk_...")

agent = Agent(name="assistant", instructions="Help the user.", tools=[...])

# Run-level hooks observe tool calls across all agents in the run.
result = await Runner.run(
    agent,
    "Search for the latest AI news",
    hooks=AsqavRunHooks(agent_name="my-agent"),
)
```

To scope signing to a single agent, attach `AsqavAgentHooks` directly:

```python
from asqav_openai_agents import AsqavAgentHooks

agent = Agent(
    name="assistant",
    instructions="Help the user.",
    tools=[...],
    hooks=AsqavAgentHooks(agent_name="my-agent"),
)
```

The hooks attempt to sign `tool:start` and `tool:end` events through the Asqav API. Signing failures produce warnings and allow tool execution to continue, so a run can have gaps in its signed record.

## Usage: check input before the agent starts

The guardrail uses `run_in_parallel=False`: the Runner waits for your predicate before the first agent's model call. A matching predicate blocks that run's model and tool execution. Input guardrails apply to the initial input only; they do not check each tool call or run again on handoffs. See the [OpenAI guardrail guide](https://developers.openai.com/api/docs/guides/agents/guardrails-approvals).

```python
from agents import Agent
from asqav_openai_agents import asqav_input_guardrail

def looks_like_exfiltration(text: str) -> bool:
    return "wire all funds" in text.lower()

agent = Agent(
    name="assistant",
    instructions="Help the user.",
    input_guardrails=[
        asqav_input_guardrail(looks_like_exfiltration, agent_name="my-agent"),
    ],
)
```

When the predicate returns True, the SDK raises `InputGuardrailTripwireTriggered`. Predicate errors also block by default; `fail_closed=False` allows input through on those errors. The guardrail attempts to sign the result before returning it. If signing fails, it logs a warning and keeps the predicate's decision, so a refusal does not guarantee a signed receipt.

## How it works

`AsqavRunHooks` and `AsqavAgentHooks` extend the Asqav adapter base class alongside the SDK's `RunHooks` / `AgentHooks`, overriding:

- `on_tool_start` attempts to sign `tool:start` with tool and agent name
- `on_tool_end` attempts to sign `tool:end` with output metadata

All hook signing is fail-open. If the Asqav API is unreachable, a warning is logged but the tool call proceeds normally.

## Data handling

In hash-only mode, this integration sends a context hash and SDK metadata; other agent/model calls have their own data handling. In full-payload mode, the SDK sends the event context to the configured signing service.

You can choose a mode at initialization:

```python
import asqav

asqav.init(api_key="sk_...", base_url="https://api.asqav.com", mode="hash-only")
```

## Configuration

```python
# Use an existing Asqav agent by ID
hooks = AsqavRunHooks(agent_id="ag_abc123")

# Override the API key
hooks = AsqavRunHooks(api_key="sk_other", agent_name="audit-agent")
```

## License

Elastic License 2.0. See [LICENSE](LICENSE).
