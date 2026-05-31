# Changelog

All notable changes to `asqav-openai-agents` are listed here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [SemVer](https://semver.org/) and track the PyPI release.

## [Unreleased]

## [0.1.0]

Initial release. `AsqavRunHooks` and `AsqavAgentHooks` that sign `tool:start` and `tool:end` on the OpenAI Agents SDK lifecycle, plus an optional `asqav_input_guardrail` that signs `input:check` and can stop a run before the agent acts. Hook signing is fail-open through the Asqav API.
