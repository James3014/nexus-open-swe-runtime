"""Replay-safe execution resume projection over existing durable journals.

Issue #69 deliberately does not create another execution state owner. It reads
the existing DurableOperationJournal / DurableEffectJournal evidence and
produces a fail-closed doctor-style resume disposition.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

RESUME_SCHEMA = "nexus.open_swe_runtime.execution_resume.v1"
CLAIM_CEILING = "EXECUTION_RESUME_EVIDENCE_ONLY"
_AMBIGUOUS_OPERATION_STATES = {"ASK_DISPATCHING", "PROTOCOL_REPAIR_DISPATCHING", "OUTCOME_UNKNOWN"}
_AMBIGUOUS_EFFECT_STATES = {"INTENT", "DISPATCHING", "OUTCOME_UNKNOWN"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value.strip()


def project_execution_resume(
    operation_state: Mapping[str, Any],
    *,
    expected_operation_id: str,
    expected_attempt_id: str,
    expected_source_revision: str,
    expected_backend_id: str,
    expected_provider_id: str,
    expected_effect_authorization_hash: str,
    expected_tool_projection_hash: str,
    effect_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    state = dict(operation_state)
    identity = state.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("durable operation identity missing")
    expected = {
        "operation_id": _text(expected_operation_id, "expected_operation_id"),
        "attempt_id": _text(expected_attempt_id, "expected_attempt_id"),
        "source_revision": _text(expected_source_revision, "expected_source_revision"),
        "backend_id": _text(expected_backend_id, "expected_backend_id"),
        "provider_id": _text(expected_provider_id, "expected_provider_id"),
        "effect_authorization_hash": _text(expected_effect_authorization_hash, "expected_effect_authorization_hash"),
        "tool_projection_hash": _text(expected_tool_projection_hash, "expected_tool_projection_hash"),
    }
    observed = {
        "operation_id": identity.get("operation_id"),
        "attempt_id": identity.get("core_attempt_id"),
        "source_revision": identity.get("core_source_revision"),
        "backend_id": identity.get("tool_projection_backend_id"),
        "provider_id": identity.get("provider_id"),
        "effect_authorization_hash": identity.get("effect_authorization_hash"),
        "tool_projection_hash": identity.get("tool_projection_hash"),
    }
    drift = sorted(key for key, value in expected.items() if observed.get(key) != value)
    completed_effects: list[dict[str, Any]] = []
    ambiguous_effects: list[str] = []
    for raw in effect_records:
        record = dict(raw)
        effect_id = str(record.get("effect_id") or "")
        status = str(record.get("status") or "").upper()
        if status == "RESULT" and effect_id:
            completed_effects.append({
                "effect_id": effect_id,
                "turn_id": record.get("turn_id"),
                "tool_call_id": record.get("tool_call_id"),
                "tool_name": record.get("tool_name"),
                "replay_allowed": False,
            })
        elif status in _AMBIGUOUS_EFFECT_STATES:
            ambiguous_effects.append(effect_id or "<unknown>")
    status = str(state.get("status") or "").upper()
    status_for_unknown = (
        "PROTOCOL_REPAIR_DISPATCHING"
        if state.get("protocol_repair_status") == "DISPATCHING"
        else status
    )
    outcome_unknown = status_for_unknown in _AMBIGUOUS_OPERATION_STATES or bool(ambiguous_effects)
    terminal = state.get("terminal_result")
    terminal_result = dict(terminal) if isinstance(terminal, Mapping) else {}
    failure_reason = str(
        state.get("failure_reason")
        or state.get("error")
        or terminal_result.get("failure_reason")
        or terminal_result.get("error")
        or ""
    )
    retryable_transport_failure = bool(state.get("retry_safe")) and not outcome_unknown
    if drift:
        disposition, reason = "RECONCILE", "IDENTITY_DRIFT"
    elif outcome_unknown:
        disposition, reason = "RECONCILE", "OUTCOME_UNKNOWN"
    elif status == "COMPLETED":
        disposition, reason = "BLOCKED", "ALREADY_COMPLETED"
    elif retryable_transport_failure:
        disposition, reason = "SAFE", "RETRYABLE_TRANSPORT_FAILURE"
    else:
        disposition, reason = "SAFE", "EXACT_IDENTITY_NO_UNKNOWN_EFFECT"
    result: dict[str, Any] = {
        "schema": RESUME_SCHEMA,
        "operation_id": expected["operation_id"],
        "attempt_id": expected["attempt_id"],
        "observed_identity": observed,
        "expected_identity": expected,
        "identity_drift": drift,
        "phase": status,
        "provider_session_identity": state.get("conversation_id") or "",
        "completed_effects": sorted(completed_effects, key=lambda row: row["effect_id"]),
        "ambiguous_effects": sorted(ambiguous_effects),
        "outcome_unknown": outcome_unknown,
        "retryable_transport_failure": retryable_transport_failure,
        "failure_reason": failure_reason,
        "terminal_result": terminal_result,
        "disposition": disposition,
        "reason": reason,
        "replay_completed_effects": False,
        "claim_ceiling": CLAIM_CEILING,
    }
    result["resume_hash"] = _sha(result)
    return result


def doctor_readback(journal: Any, **expected: Any) -> dict[str, Any]:
    state = journal.read()
    records: list[Mapping[str, Any]] = []
    effect_journal = getattr(journal, "effect_journal", None)
    if effect_journal is not None and hasattr(effect_journal, "records"):
        records = list(effect_journal.records())
    return project_execution_resume(state, effect_records=records, **expected)


__all__ = ["RESUME_SCHEMA", "CLAIM_CEILING", "project_execution_resume", "doctor_readback"]
