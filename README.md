# Nexus Open SWE Runtime

External Open SWE / Deep Agents execution runtime for Nexus.

## Scope

This repository owns execution runtime implementation only:
- Semantic execution
- Diagnosis & bounded repair
- Model transport
- Tool execution inside authorized scope

It does not own Completion truth, Evidence Trust truth, or routing/governance authorities.

## Projected tool exposure

When a worker request supplies the Runtime-owned `effect_authorization` and
`tool_projection_manifest` contracts, this runtime consumes them as external authority.
It does not select or widen those privileges. The projection must target
`backend_id=nexus-open-swe-runtime`, match the request operation/attempt/provider identity,
and pass its Wave 1 content hashes and effect-subset checks.

For projected requests, diagnosis and repair graphs are physically built with only the
selected tool names that this backend supports. The runtime then reads the compiled graph
surface before invocation and fails closed if actual exposure is wider than the projection.
Recovery identity binds the authorization hash, projection hash, backend, and projected tool
set so restart/reconciliation cannot substitute a wider projection.

Terminal and bounded-failure results include a derived
`nexus.open_swe_runtime.execution_exposure_receipt.v1` when a projection was consumed. The
receipt records the actual per-phase surfaces and their union; it is evidence only, not a new
permission authority. Requests without these optional fields retain the existing legacy
behavior until their host integration is migrated.

This source contract does not prove Nexus host wiring, DevSpace enforcement, deployment, or
production activation.

## Development

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run nexus-open-swe-runtime --help
```
