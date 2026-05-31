<p align="center">
  <a href="https://asqav.com">
    <img src="https://asqav.com/logo-text-white.png" alt="Asqav" width="200">
  </a>
</p>
<p align="center">
  Stop a rogue agent before it acts, and prove what it tried.
</p>
<p align="center">
  <a href="https://www.asqav.com/">Website</a> |
  <a href="https://www.asqav.com/docs">Docs</a> |
  <a href="https://github.com/jagmarques/asqav-sdk">SDK</a>
</p>

# Asqav for the OpenAI Agents SDK

Stop a rogue agent before it acts, and prove what it tried.

`asqav-openai-agents` plugs [Asqav](https://asqav.com) into the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). Every tool your agent invokes produces a tamper-evident signed record of what it attempted, giving you cryptographic proof of agent behaviour for EU AI Act, DORA, and SOC 2 audits.

This package gives you two surfaces:

- **Hooks** (`AsqavRunHooks`, `AsqavAgentHooks`) sign `tool:start` and `tool:end` on the SDK's documented `RunHooks` / `AgentHooks` lifecycle. They observe and record, and they are fail-open: they never block tool execution.
- **Guardrail** (`asqav_input_guardrail`) is the blocking surface. It signs each checked input and can stop a run before the agent acts by tripping the SDK tripwire when your predicate matches.

## Install

Not yet on PyPI. Install from GitHub:

```bash
pip install "git+https://github.com/jagmarques/asqav-openai-agents.git#egg=asqav-openai-agents[agents]"
```

The OpenAI Agents SDK is a peer dependency. If you already have `openai-agents` installed you can drop the `[agents]` extra. If it is missing, the package raises a clear `ImportError` telling you to install it.

## Usage: sign every tool call

```python
import asqav
from agents import Agent, Runner

from asqav_openai_agents import AsqavRunHooks

asqav.init(api_key="sk_...")

agent = Agent(name="assistant", instructions="Help the user.", tools=[...])

# Run-level hooks sign tool calls across all agents in the run.
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

Every tool call produces signed `tool:start` and `tool:end` events through the Asqav API. Signatures use NIST FIPS 204 ML-DSA cryptography server-side, producing tamper-evident audit trails.

## Usage: stop a rogue agent before it acts

The guardrail runs before the agent and can block the run. It signs an `input:check` event each time, and trips the tripwire when your predicate returns True.

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

When the predicate matches, the SDK raises `InputGuardrailTripwireTriggered` and the agent never runs. The attempt is still signed, so you keep proof of what was blocked.

## How it works

`AsqavRunHooks` and `AsqavAgentHooks` extend the Asqav adapter base class alongside the SDK's `RunHooks` / `AgentHooks`, overriding:

- `on_tool_start` signs `tool:start` with tool and agent name
- `on_tool_end` signs `tool:end` with output metadata

All hook signing is fail-open. If the Asqav API is unreachable, a warning is logged but the tool call proceeds normally.

## Data handling

`asqav-openai-agents` is a thin wrapper around the `asqav` Python SDK and inherits its mode behaviour:

- **Asqav cloud (`*.asqav.com`):** the SDK hashes your action context locally and sends only the hash plus a small metadata bag. Raw prompts and tool arguments never leave your infrastructure.
- **Self-hosted:** the SDK sends the full context so the server can run policy checks, PII redaction, and richer audit views.

You can override per call:

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

MIT
