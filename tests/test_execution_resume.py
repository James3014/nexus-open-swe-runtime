from __future__ import annotations

from nexus_open_swe_runtime.execution_resume import project_execution_resume


def state(status="RESPONSE_RECOVERED", **over):
    identity = {
        "operation_id": "op-1",
        "core_attempt_id": "attempt-1",
        "core_source_revision": "git-commit:" + "a" * 40,
        "tool_projection_backend_id": "nexus-open-swe-runtime",
        "provider_id": "opencli",
        "effect_authorization_hash": "auth-1",
        "tool_projection_hash": "projection-1",
    }
    identity.update(over.pop("identity", {}))
    return {"identity": identity, "status": status, "conversation_id": "conv-1", **over}


EXPECTED = {
    "expected_operation_id": "op-1",
    "expected_attempt_id": "attempt-1",
    "expected_source_revision": "git-commit:" + "a" * 40,
    "expected_backend_id": "nexus-open-swe-runtime",
    "expected_provider_id": "opencli",
    "expected_effect_authorization_hash": "auth-1",
    "expected_tool_projection_hash": "projection-1",
}


def test_exact_identity_resumes_without_replaying_completed_effect():
    out = project_execution_resume(
        state(), effect_records=[{"effect_id": "effect-1", "status": "RESULT", "turn_id": "t1", "tool_call_id": "c1", "tool_name": "write_file"}], **EXPECTED,
    )
    assert out["disposition"] == "SAFE"
    assert out["completed_effects"][0]["replay_allowed"] is False
    assert out["replay_completed_effects"] is False
    assert out["provider_session_identity"] == "conv-1"


def test_authorization_source_backend_provider_or_projection_drift_reconciles():
    fields = {
        "core_source_revision": "other",
        "tool_projection_backend_id": "other",
        "provider_id": "other",
        "effect_authorization_hash": "other",
        "tool_projection_hash": "other",
    }
    for field, value in fields.items():
        out = project_execution_resume(state(identity={field: value}), **EXPECTED)
        assert out["disposition"] == "RECONCILE"
        assert out["reason"] == "IDENTITY_DRIFT"
        assert out["identity_drift"]


def test_outcome_unknown_never_authorizes_retry():
    out = project_execution_resume(state(status="ASK_DISPATCHING"), **EXPECTED)
    assert out["disposition"] == "RECONCILE"
    assert out["outcome_unknown"] is True
    assert out["replay_completed_effects"] is False


def test_ambiguous_effect_requires_reconciliation():
    out = project_execution_resume(state(), effect_records=[{"effect_id": "e2", "status": "INTENT"}], **EXPECTED)
    assert out["disposition"] == "RECONCILE"
    assert out["ambiguous_effects"] == ["e2"]


def test_retryable_transport_failure_is_distinct_from_unknown_effect():
    out = project_execution_resume(
        state(status="TRANSPORT_FAILED", retry_safe=True, failure_reason="provider_unavailable"),
        **EXPECTED,
    )
    assert out["disposition"] == "SAFE"
    assert out["reason"] == "RETRYABLE_TRANSPORT_FAILURE"
    assert out["retryable_transport_failure"] is True
    assert out["outcome_unknown"] is False
    assert out["failure_reason"] == "provider_unavailable"


def test_completed_operation_is_terminal_and_preserves_terminal_result():
    out = project_execution_resume(
        state(status="COMPLETED", terminal_result={"status": "FAILED", "error": "repair_failed"}),
        **EXPECTED,
    )
    assert out["disposition"] == "BLOCKED"
    assert out["reason"] == "ALREADY_COMPLETED"
    assert out["terminal_result"]["error"] == "repair_failed"
    assert out["failure_reason"] == "repair_failed"
