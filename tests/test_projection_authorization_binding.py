"""#112: effect_authorization repository/workspace_id binding and the write-path ceiling on reconcile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_runtime import (
    FakeGraph,
    _project_worker_request,
    _record,
    _runtime,
    _worker_request,
)

from nexus_open_swe_runtime import cli
from nexus_open_swe_runtime.recovery import DurableOperationJournal, RecoveryIdentity

TOOLS = ("read_file", "record_diagnosis", "record_worker_result", "write_file")
REPAIR_SURFACE = ("read_file", "record_worker_result", "write_file")


def _reauthorize(request: dict, **changes: object) -> dict:
    """Re-sign effect_authorization and its projection after changing authorization fields."""
    material = {k: v for k, v in request["effect_authorization"].items() if k != "authorization_hash"}
    material.update(changes)
    authorization = {
        **material,
        "authorization_hash": cli._sha256(cli._canonical_json(material)),
    }
    projection = {k: v for k, v in request["tool_projection_manifest"].items() if k != "projection_hash"}
    projection["authorization_hash"] = authorization["authorization_hash"]
    request["effect_authorization"] = authorization
    request["tool_projection_manifest"] = {
        **projection,
        "projection_hash": cli._sha256(cli._canonical_json(projection)),
    }
    return request


def _run_write(request: dict, target: str) -> tuple[dict, FakeGraph]:
    workspace = Path(request["workspace_path"])
    repair = FakeGraph(
        REPAIR_SURFACE,
        _record("record_worker_result", {"summary": "done"}),
        effect=lambda: (workspace / target).write_text("TAMPERED\n", encoding="utf-8"),
    )
    diagnosis = FakeGraph(
        ("read_file", "record_diagnosis"),
        _record(
            "record_diagnosis",
            {"status": "ROOT_CAUSE_SUPPORTED", "summary": "s", "evidence_paths": ["a.py"]},
        ),
    )
    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_a: object(),
        diagnosis_factory=lambda *_a, **_k: diagnosis,
        repair_factory=lambda *_a, **_k: repair,
    )
    return result, repair


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "James3014/other-repo"),
        ("workspace_id", "some-other-workspace"),
    ],
)
def test_worker_run_rejects_authorization_bound_to_other_repository_or_workspace(
    tmp_path: Path, field: str, value: str
) -> None:
    request = _project_worker_request(_worker_request(tmp_path), TOOLS)
    request[field] = value  # the request names a different repository/workspace than the authorization
    result, repair = _run_write(request, "a.py")
    assert result["status"] == "OPEN_SWE_TOOL_PROJECTION_BLOCKED"
    assert result["error_code"] == "OPEN_SWE_EFFECT_AUTHORIZATION_BINDING_MISMATCH"
    assert repair.calls == 0
    assert (Path(request["workspace_path"]) / "a.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_worker_run_accepts_matching_repository_and_workspace(tmp_path: Path) -> None:
    request = _project_worker_request(_worker_request(tmp_path), TOOLS)
    request["repository"] = "James3014/Nexus-Open-SWE-Runtime"  # case-insensitive like Core ids
    request["workspace_id"] = request["effect_authorization"]["workspace_id"]
    result, repair = _run_write(request, "a.py")
    assert result["status"] == "COMPLETED"
    assert repair.calls == 1


def test_core_binding_repository_must_match_authorization_repository() -> None:
    request = {"effect_authorization": {"repository": "James3014/other-repo"}}
    core = {"core_repository": "James3014/nexus-open-swe-runtime"}
    with pytest.raises(cli.RuntimeErrorBounded, match="OPEN_SWE_EFFECT_AUTHORIZATION_BINDING_MISMATCH"):
        cli._assert_authorization_binding(request, core)
    request = {"effect_authorization": {"repository": "James3014/Nexus-Open-SWE-Runtime"}}
    cli._assert_authorization_binding(request, core)
    cli._assert_authorization_binding({}, core)


@pytest.mark.parametrize(("allowed", "resumed"), [(("a.py",), True), (("b.py",), False), (("a.py", "b.py"), False)])
def test_worker_reconcile_enforces_write_path_ceiling_before_any_write(
    tmp_path: Path, allowed: tuple[str, ...], resumed: bool
) -> None:
    """A recovery identity whose allowed_paths exceed the authorized ceiling is never resumed."""
    request = _project_worker_request(_worker_request(tmp_path), TOOLS)
    request.update(operation="worker_reconcile", provider_id="google_genai", model_id="test-model")
    workspace = Path(request["workspace_path"])
    (workspace / "b.py").write_text("SECRET_GATE = True\n", encoding="utf-8")
    state_root = Path(request["runtime_state_root"])
    identity = RecoveryIdentity(
        operation_id=request["operation_id"],
        execution_material_sha256="b" * 64,
        workspace=str(workspace.resolve()),
        task_id="task-1",
        unit_id="u1",
        session_id="session-1",
        allowed_paths=allowed,
        provider_id="google_genai",
        model_id="test-model",
        worker_identity_sha256=request["worker_identity_sha256"],
        transport_config_sha256=cli._sha256("{}"),
        runtime_identity_sha256=cli._sha256(
            cli._canonical_json({
                **cli._runtime_code_identity(),
                "deepagents": cli._deepagents_version(),
                "checkpoint_namespace": "open-swe-repair-v1",
            })
        ),
        effect_authorization_hash=request["effect_authorization"]["authorization_hash"],
        tool_projection_hash=request["tool_projection_manifest"]["projection_hash"],
        tool_projection_backend_id="nexus-open-swe-runtime",
        projected_tools=TOOLS,
    )
    journal = DurableOperationJournal(state_root, identity)
    journal.prepare()
    journal.ask_dispatching(turn_id="turn-1", prompt="repair", ordinal=0)
    journal.conversation_bound("conversation-1")
    cli._atomic_json(
        cli._operation_path(request),
        {
            **cli._write_started(request, "worker"),
            "status": "OPEN_SWE_OUTCOME_UNKNOWN",
            "outcome_unknown": True,
            "retry_safe": False,
            "execution_material_sha256": identity.execution_material_sha256,
            "session_id": identity.session_id,
        },
    )
    checkpoint, _ = cli.create_checkpoint(
        state_root, request["operation_id"], identity.checkpoint_namespace
    )
    checkpoint.conn.close()

    built: list[object] = []

    def model_builder(*args: object) -> object:
        built.append(args)
        raise RuntimeError("stop after resume was attempted")

    result = cli._worker_reconcile(
        request,
        runtime_loader=cli._load_runtime,
        model_builder=model_builder,
        graph_builder=lambda *_a, **_k: object(),
    )
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert bool(built) is resumed
    assert (workspace / "b.py").read_text(encoding="utf-8") == "SECRET_GATE = True\n"
    assert not list((state_root / "recovery" / "effects").glob("*.json"))
    json.dumps(result)


def _git_workspace(request: dict, origin: str) -> None:
    import subprocess

    workspace = request["workspace_path"]
    for args in (
        ["init", "-q"],
        ["remote", "add", "origin", origin],
    ):
        subprocess.run(["git", "-C", workspace, *args], check=True)


def test_worker_run_rejects_authorization_repository_differing_from_git_origin(
    tmp_path: Path,
) -> None:
    request = _project_worker_request(_worker_request(tmp_path), TOOLS)
    _git_workspace(request, "https://github.com/James3014/other-repo.git")
    result, repair = _run_write(request, "a.py")
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN" or result["status"].startswith("OPEN_SWE")
    assert result["status"] != "COMPLETED"
    assert result["error_code"] == "OPEN_SWE_EFFECT_AUTHORIZATION_BINDING_MISMATCH"
    assert repair.calls == 0
    assert (Path(request["workspace_path"]) / "a.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_worker_run_accepts_authorization_repository_matching_git_origin(tmp_path: Path) -> None:
    request = _project_worker_request(_worker_request(tmp_path), TOOLS)
    _git_workspace(request, "git@github.com:James3014/nexus-open-swe-runtime.git")
    result, repair = _run_write(request, "a.py")
    assert result["status"] == "COMPLETED"
    assert repair.calls == 1


@pytest.mark.parametrize(("origin", "resumed"), [("https://github.com/James3014/other-repo.git", False), ("https://github.com/James3014/nexus-open-swe-runtime", True)])
def test_worker_reconcile_enforces_git_origin_binding(
    tmp_path: Path, origin: str, resumed: bool
) -> None:
    """A recovery identity whose allowed_paths exceed the authorized ceiling is never resumed."""
    request = _project_worker_request(_worker_request(tmp_path), TOOLS)
    request.update(operation="worker_reconcile", provider_id="google_genai", model_id="test-model")
    workspace = Path(request["workspace_path"])
    _git_workspace(request, origin)
    (workspace / "b.py").write_text("SECRET_GATE = True\n", encoding="utf-8")
    state_root = Path(request["runtime_state_root"])
    identity = RecoveryIdentity(
        operation_id=request["operation_id"],
        execution_material_sha256="b" * 64,
        workspace=str(workspace.resolve()),
        task_id="task-1",
        unit_id="u1",
        session_id="session-1",
        allowed_paths=("a.py",),
        provider_id="google_genai",
        model_id="test-model",
        worker_identity_sha256=request["worker_identity_sha256"],
        transport_config_sha256=cli._sha256("{}"),
        runtime_identity_sha256=cli._sha256(
            cli._canonical_json({
                **cli._runtime_code_identity(),
                "deepagents": cli._deepagents_version(),
                "checkpoint_namespace": "open-swe-repair-v1",
            })
        ),
        effect_authorization_hash=request["effect_authorization"]["authorization_hash"],
        tool_projection_hash=request["tool_projection_manifest"]["projection_hash"],
        tool_projection_backend_id="nexus-open-swe-runtime",
        projected_tools=TOOLS,
    )
    journal = DurableOperationJournal(state_root, identity)
    journal.prepare()
    journal.ask_dispatching(turn_id="turn-1", prompt="repair", ordinal=0)
    journal.conversation_bound("conversation-1")
    cli._atomic_json(
        cli._operation_path(request),
        {
            **cli._write_started(request, "worker"),
            "status": "OPEN_SWE_OUTCOME_UNKNOWN",
            "outcome_unknown": True,
            "retry_safe": False,
            "execution_material_sha256": identity.execution_material_sha256,
            "session_id": identity.session_id,
        },
    )
    checkpoint, _ = cli.create_checkpoint(
        state_root, request["operation_id"], identity.checkpoint_namespace
    )
    checkpoint.conn.close()

    built: list[object] = []

    def model_builder(*args: object) -> object:
        built.append(args)
        raise RuntimeError("stop after resume was attempted")

    result = cli._worker_reconcile(
        request,
        runtime_loader=cli._load_runtime,
        model_builder=model_builder,
        graph_builder=lambda *_a, **_k: object(),
    )
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert bool(built) is resumed
    assert (workspace / "b.py").read_text(encoding="utf-8") == "SECRET_GATE = True\n"
    assert not list((state_root / "recovery" / "effects").glob("*.json"))
    json.dumps(result)
