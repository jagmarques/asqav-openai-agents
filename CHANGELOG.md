# Changelog

Notable changes to asqav-openai-agents. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

First release.

### Added
- `AsqavRunHooks` and `AsqavAgentHooks`: OpenAI Agents SDK hooks that sign
  `tool:start` and `tool:end` events through the Asqav SDK (fail-open).
- `asqav_input_guardrail`: an optional input guardrail that signs an
  `input:check` event and can stop a run before the agent acts by tripping the
  SDK guardrail tripwire.
- Tag-gated PyPI publish workflow using OIDC trusted publishing.
- Pull-request CI that runs the test suite and a `python -m build` plus
  `twine check` dry run.

### Changed
- Pinned the `asqav` SDK dependency to the 0.8 line (`asqav>=0.8.0,<0.9.0`).
