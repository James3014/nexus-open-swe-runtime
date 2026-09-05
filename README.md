# Nexus Open SWE Runtime

External Open SWE / Deep Agents execution runtime for Nexus.

## Scope

This repository owns execution runtime implementation only:
- Semantic execution
- Diagnosis & bounded repair
- Model transport
- Tool execution inside authorized scope

It does not own Completion truth, Evidence Trust truth, or routing/governance authorities.

## Development

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run nexus-open-swe-runtime --help
```
