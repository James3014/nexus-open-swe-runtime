"""Issue #68: additive hints + search-recovery evidence."""

from __future__ import annotations

import importlib.util


def _mod():
    spec = importlib.util.spec_from_file_location(
        "hint_execution", "nexus_open_swe_runtime/hint_execution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _mod()


def _hints(**overrides):
    data = {"repository": "owner/repo", "revision": "rev-1",
            "query_evidence_hash": "qhash", "retriever_policy": "rrf_k60_weighted",
            "candidates": ["src/a.py", "src/b.py"]}
    data.update(overrides)
    return data


def test_valid_hints_keep_recovery_tools():
    bound = H.bind_execution_hints(expected_repository="owner/repo",
        expected_revision="rev-1", hint_input=_hints(),
        authorized_tools=["search", "grep", "read", "edit"])
    assert bound["bound"] is True
    assert "search" in bound["authorized_tools"]
    assert "read" in bound["authorized_tools"]


def test_worker_discovers_file_outside_topk():
    bound = H.bind_execution_hints(expected_repository="owner/repo",
        expected_revision="rev-1", hint_input=_hints(), authorized_tools=["search"])
    receipt = H.build_hint_execution_receipt(bound=bound["bound"],
        hint_identity=bound["hint_identity"], hinted_reads=["src/a.py"],
        discovered_outside=["src/hidden.py"],
        tool_calls=[{"tool": "search", "count": 3}], widened=True)
    assert receipt["discovered_outside_hints"] == ["src/hidden.py"]
    assert receipt["widened_beyond_hints"] is True
    assert H.validate_hint_execution_receipt(receipt) == []


def test_foreign_hint_falls_back():
    bound = H.bind_execution_hints(expected_repository="owner/repo",
        expected_revision="rev-1", hint_input=_hints(repository="evil/repo"),
        authorized_tools=["search", "read"])
    assert bound["bound"] is False
    assert "hint_foreign_repository" in bound["blockers"]
    assert bound["authorized_tools"] == ["read", "search"]


def test_stale_hint_does_not_contaminate():
    bound = H.bind_execution_hints(expected_repository="owner/repo",
        expected_revision="rev-1", hint_input=_hints(revision="rev-9"),
        authorized_tools=["read"])
    assert bound["bound"] is False
    assert bound["hints"] == []


def test_hinted_vs_discovered_distinguished():
    receipt = H.build_hint_execution_receipt(bound=True,
        hint_identity={"binding_hash": "x"}, hinted_reads=["src/a.py", "src/a.py"],
        discovered_outside=[], tool_calls=[{"tool": "read", "count": 2}])
    assert receipt["hinted_read"] == ["src/a.py"]
    assert receipt["search_tool_calls"] == 2
    assert receipt["claim_ceiling"] == H.HINT_EXECUTION_CLAIM_CEILING


def test_missing_telemetry_explicit_and_tamper_fails():
    receipt = H.build_hint_execution_receipt(bound=False)
    assert receipt["hinted_read"] == []
    assert H.validate_hint_execution_receipt(receipt) == []
    tampered = dict(receipt)
    tampered["widened_beyond_hints"] = True
    assert "receipt_hash_mismatch" in H.validate_hint_execution_receipt(tampered)
