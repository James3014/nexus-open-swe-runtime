"""Issue #68 execution-seam integration coverage."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from nexus_open_swe_runtime import cli
from nexus_open_swe_runtime.hint_execution import (
    HintExecutionObserver,
    add_recovery_observation,
    bind_execution_hints,
    build_hint_execution_receipt,
)

_SPEC = importlib.util.spec_from_file_location("runtime_tests", Path(__file__).parent / "test_runtime.py")
assert _SPEC is not None and _SPEC.loader is not None
R = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(R)


def _graph_run(tmp_path, monkeypatch, revision):
    request = R._v2_request(tmp_path)
    request["retrieval_hints"] = {
        "repository": "James3014/Nexus-new",
        "revision": revision,
        "query_evidence_hash": "q" * 64,
        "candidates": ["a.py:VALUE"],
    }
    monkeypatch.setattr(
        cli,
        "_git_output",
        lambda _workspace, *args: {
            ("rev-parse", "HEAD"): "b" * 40,
            ("rev-parse", "--verify", "b" * 40 + "^{tree}"): "e" * 40,
            ("status", "--porcelain"): "",
            ("remote", "get-url", "origin"): "git@github.com:James3014/Nexus-new.git",
        }[args],
    )
    monkeypatch.setattr(cli, "create_checkpoint", lambda _root, _op, namespace: (None, namespace))
    sends = []

    def model_factory(_runtime, provider, model_id, transport_config, state_root):
        model = R.OpenCLIWebChatModel(
            executable=transport_config["executable"],
            intelligence_level=model_id,
            opencli_profile=transport_config["profile"],
            site_session=transport_config["site_session"],
            timeout_seconds=transport_config["timeout_seconds"],
            runtime_state_root=state_root,
        )
        R._composite_worker_model(model, sends)
        return model

    return request, sends, cli._worker_run(
        request,
        runtime_loader=cli._load_runtime,
        model_factory=model_factory,
        repair_factory=cli.build_repair_graph,
    )


def test_real_graph_consumes_valid_advisory_hints(tmp_path, monkeypatch):
    request, sends, result = _graph_run(tmp_path, monkeypatch, "git-commit:" + "b" * 40)
    assert result["status"] == "COMPLETED"
    assert "Advisory retrieval hints" in sends[0]
    assert result["hint_execution_receipt"]["bound"] is True
    assert result["hint_execution_receipt"]["hinted_total"] == 1
    persisted = cli._read_operation_state(cli._operation_path(request))
    assert persisted["hint_execution_receipt"] == result["hint_execution_receipt"]


def test_real_graph_stale_hints_fail_closed_without_prompt_contamination(tmp_path, monkeypatch):
    _request, sends, result = _graph_run(tmp_path, monkeypatch, "c" * 40)
    assert result["status"] == "COMPLETED"
    assert "Advisory retrieval hints" not in sends[0]
    assert result["hint_execution_receipt"]["bound"] is False
    assert "hint_stale_revision" in result["hint_binding_blockers"]


def test_material_fingerprint_binds_retrieval_hints(tmp_path):
    request = R._worker_request(tmp_path)
    before = cli._worker_material_fingerprint(request)
    request["retrieval_hints"] = {"repository": "o/r", "revision": "a", "query_evidence_hash": "q", "candidates": []}
    after = cli._worker_material_fingerprint(request)
    assert before != after


def test_symbol_candidate_normalizes_to_file_and_success_status_wins(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("error: str = 'valid content'\n", encoding="utf-8")
    bound = bind_execution_hints(
        expected_repository="o/r",
        expected_revision="a" * 40,
        hint_input={"repository": "o/r", "revision": "git-commit:" + "a" * 40, "query_evidence_hash": "q", "candidates": ["a.py:Thing"]},
        authorized_tools=["read_file", "grep"],
    )
    observer = HintExecutionObserver(bound, root)
    observer.on_tool_start({"name": "read_file"}, {"file_path": "/a.py"}, run_id="r1")
    observer.on_tool_end(SimpleNamespace(status="success", content="error: str = 'valid content'"), run_id="r1", name="read_file")
    receipt = observer.receipt()
    assert receipt["hinted_read"] == ["a.py"]
    assert receipt["discovered_outside_hints"] == []


def test_virtual_list_only_counts_existing_files(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "test.py").write_text("x = 1\n", encoding="utf-8")
    bound = bind_execution_hints(
        expected_repository="o/r",
        expected_revision="a",
        hint_input={"repository": "o/r", "revision": "a", "query_evidence_hash": "q", "candidates": []},
        authorized_tools=["ls"],
    )
    observer = HintExecutionObserver(bound, root)
    observer.on_tool_start({"name": "ls"}, {"path": "/src"}, run_id="l1")
    observer.on_tool_end("['/src', '/src/test.py']", run_id="l1", name="ls")
    assert observer.receipt()["discovered_outside_hints"] == ["src/test.py"]


def test_recovery_observation_preserves_prior_receipt():
    prior = build_hint_execution_receipt(
        bound=True,
        initial_hints=["a.py"],
        hinted_reads=["a.py"],
        tool_calls=[{"tool": "read_file", "count": 1}],
    )
    replay = build_hint_execution_receipt(
        bound=True,
        initial_hints=["a.py"],
        discovered_outside=["b.py"],
        tool_calls=[{"tool": "grep", "count": 2}],
    )
    merged = add_recovery_observation(prior, replay)
    assert merged["hinted_read"] == ["a.py"]
    assert merged["discovered_outside_hints"] == ["b.py"]
    assert merged["search_tool_calls"] == 3
    assert "worker_reconcile_replay" in merged["recovery_events"]
