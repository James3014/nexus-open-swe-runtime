# Standalone Runtime Integration & Client Compatibility

## 1. Overview & Goal

`nexus-open-swe-runtime` is the standalone external execution runtime for Open SWE and Deep Agents in the Nexus ecosystem.
This document defines the runtime identity contract, client configuration guidance, and state root lifecycle to guarantee clean cross-repo compatibility between `Nexus-new` (or other orchestrators) and this standalone executable.

## 2. Runtime Identity Surface

The runtime exposes a deterministic identity probe to prove which installation, distribution version, and artifact hash is executing.

### CLI Probe
```bash
nexus-open-swe-runtime --identity
```

### JSON Protocol Probe
```json
{
  "schema": "nexus.open_swe_runtime.request.v1",
  "operation": "identity"
}
```

### Response Schema (`nexus.open_swe_runtime.result.v1`)
```json
{
  "schema": "nexus.open_swe_runtime.result.v1",
  "kind": "identity",
  "status": "IDENTIFIED",
  "distribution_name": "nexus-open-swe-runtime",
  "distribution_version": "0.1.0",
  "runtime_protocol_version": "nexus.open_swe_runtime.request.v1",
  "artifact_identity": {
    "module_file": "/path/to/site-packages/nexus_open_swe_runtime/cli.py",
    "module_sha256": "<sha256>",
    "deepagents_version": "0.7.6"
  },
  "authority_boundary": "execution_runtime_only",
  "process_started": false,
  "outcome_unknown": false,
  "retry_safe": true,
  "started_at": "<iso-timestamp>",
  "finished_at": "<iso-timestamp>"
}
```

## 3. Client Configuration Guidance for Orchestrators (`Nexus-new`)

To prevent accidental PATH resolution and version skew between older embedded environments and the standalone distribution:
1. **Explicit Executable Path**: Orchestrators should configure an explicit executable path (e.g. via `NEXUS_OPEN_SWE_RUNTIME_BIN` or deployment configuration pointing to the isolated virtualenv `/path/to/venv/bin/nexus-open-swe-runtime`), rather than relying on ambient `PATH`.
2. **Identity Handshake**: At worker pool initialization or healthcheck time, the orchestrator invokes `--identity` and asserts `distribution_name == "nexus-open-swe-runtime"` and `runtime_protocol_version == "nexus.open_swe_runtime.request.v1"`.
3. **Fail-Closed on Protocol Mismatch**: Any request with an unsupported schema version immediately fails closed with `OPEN_SWE_RUNTIME_PROTOCOL_FAILED` without spawning tasks or mutating state.

## 4. Persisted State Root & Restart / Upgrade Lifecycle

1. **Explicit State Root**: Every operation request MUST provide an explicit `runtime_state_root`.
2. **Durable Operation Identity**: Completed operations are indexed by `operation_id` under `<runtime_state_root>/operations/<op_id>.json`.
3. **No Duplicate Semantic Execution**:
   - Resending a completed operation returns the cached terminal result without re-invoking models or tools.
   - Ambiguous timeouts produce `OPEN_SWE_OUTCOME_UNKNOWN` (`outcome_unknown: true`). Such operations are reconcile-only; caller clients MUST NOT blindly redispatch them.
4. **Preserved Generation State**: Upgrading `nexus-open-swe-runtime` reuses the same state directory. Package extraction does not create a fragmented second state universe.

### Target operation reconciliation

Reconciliation is bound to the request's exact `operation_id`. The corresponding
`operations/<operation_id>.json` record is the only completion authority; a
workspace index is a projection and cannot satisfy a different target operation.
Workers also persist their resolved workspace and execution-material fingerprint
(prompt plus artifact path/content) in the operation record. A terminal result
is returned only when its operation and worker material (when supplied by an
execution replay request) match; missing or mismatched material, including a
legacy record without the required workspace binding, yields
`OPEN_SWE_OUTCOME_UNKNOWN` and never re-dispatches execution.

## 5. Security & Authority Boundaries

- **Execution Runtime Only**: The runtime possesses zero Workforce admission, Capability routing, Candidate acceptance, or Git commit/push authority.
- **Environment Containment**: Process execution does not leak controller secrets (`GITHUB_TOKEN`, `GEMINI_API_KEY`) into state files. State files are written with restricted permissions (`0o600`).
