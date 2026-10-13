"""Core-bound repair continuation of a BOUND worker session.

A continuation never mints or re-presents a binding: it is admitted against the
Core identity persisted on the BOUND session and keeps that exact identity.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nexus_open_swe_runtime import cli

ORIGIN = "git@github.com:James3014/Nexus-new.git"
CARD_REF = "tasks/task-card.md"
TARGET = "a.py"
INITIAL_OPERATION = "1" * 64
CONTINUE_OPERATION = "2" * 64


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


class _Graph:
    def __init__(self, surface, output, effect=None):
        self.surface = tuple(surface)
        self.output = output
        self.effect = effect
        self.calls = 0

    def get_graph(self):
        tools = {name: object() for name in self.surface}
        return SimpleNamespace(
            nodes={"tools": SimpleNamespace(data=SimpleNamespace(tools_by_name=tools))}
        )

    def invoke(self, _payload, config=None):
        self.calls += 1
        if self.effect is not None:
            self.effect()
        return self.output


def _record(name: str, envelope: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [SimpleNamespace(tool_calls=[{"name": name, "args": {"envelope": envelope}}])]
    }


def _runtime() -> dict[str, Any]:
    return {"human_message": lambda content: content}


class _Calls:
    def __init__(self) -> None:
        self.model = 0
        self.diagnosis = 0
        self.repair = 0

    def as_tuple(self) -> tuple[int, int, int]:
        return self.model, self.diagnosis, self.repair


def _factories(calls: _Calls, content: str) -> dict[str, Any]:
    def model_factory(*_args):
        calls.model += 1
        return SimpleNamespace(_conversation_id=None)

    def diagnosis_factory(*_args, **_kwargs):
        calls.diagnosis += 1
        return _Graph(
            cli.DIAGNOSIS_TOOLS,
            _record(
                "record_diagnosis",
                {
                    "status": "ROOT_CAUSE_SUPPORTED",
                    "summary": "the verifier finding points at the target",
                    "evidence_paths": [TARGET],
                },
            ),
        )

    def repair_factory(
        _model, workspace, _runtime, allowed_paths, _key, checkpointer=None, effect_journal=None
    ):
        calls.repair += 1
        journal = effect_journal
        assert journal is not None
        target = Path(workspace) / allowed_paths[0]

        def effect() -> None:
            old = target.read_text(encoding="utf-8") if target.exists() else None
            written = journal.intent(
                turn_id=f"turn-{calls.repair}",
                tool_call_id=f"call-{calls.repair}",
                tool_name="write_file",
                arguments={"file_path": allowed_paths[0], "content": content},
                path=target,
                preimage=old,
                postimage=content,
            )
            journal.recover_write(written)

        return _Graph(
            cli.REPAIR_TOOLS, _record("record_worker_result", {"summary": "repaired"}), effect
        )

    return {
        "runtime_loader": _runtime,
        "model_factory": model_factory,
        "diagnosis_factory": diagnosis_factory,
        "repair_factory": repair_factory,
    }


def _future(seconds: int = 3600) -> str:
    moment = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _bound_initial_request(tmp_path: Path, valid_until: str | None = "") -> dict[str, Any]:
    if valid_until == "":
        valid_until = _future()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "test@example.com")
    _git(workspace, "config", "user.name", "Test")
    _git(workspace, "remote", "add", "origin", ORIGIN)
    card = workspace / CARD_REF
    card.parent.mkdir(parents=True)
    card.write_text(
        "\n".join(
            [
                "- task_id: `task-1`",
                "- status: `ACTIVE`",
                "- worker_may_approve: `false`",
                "- worker_may_integrate: `false`",
                "- worker_may_push: `false`",
                "- AUTO_CHAIN: `false`",
                "- allow_deletions: `false`",
                "\n## Allowed files\n",
                f"- `{TARGET}`",
            ]
        ),
        encoding="utf-8",
    )
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-q", "-m", "base")
    base = _git(workspace, "rev-parse", "HEAD")
    base_tree = _git(workspace, "rev-parse", "HEAD^{tree}")
    contract = {
        "contract_id": "open-swe-test-contract",
        "requirements_hash": "sha256:" + "1" * 64,
        "required_verifier_ids": ["pytest"],
        "allowed_paths": [TARGET],
        "deletion_policy": "FORBID",
    }
    binding: dict[str, Any] = {
        "schema": "nexus.repository_mutation_binding.v1",
        "binding_id": "binding-test-1",
        "operation_id": INITIAL_OPERATION,
        "attempt_id": "attempt-test-1",
        "repository": {
            "canonical_id": "James3014/Nexus-new",
            "origin": ORIGIN,
            "source_revision": f"git-commit:{base}",
            "source_tree": f"git-tree:{base_tree}",
            "workspace_identity": "sha256:" + cli._sha256(str(workspace.resolve())),
            "workspace_mode": "managed_worktree",
        },
        "integration_authority": {
            "execution_lane": "DIRECT_DELEGATED",
            "authority_ref": "owner-direct:test",
            "authority_hash": "sha256:" + "2" * 64,
        },
        "capability_discovery": {
            "required": True,
            "receipt_hash": "sha256:" + "3" * 64,
            "index_revision": f"git-commit:{base}",
        },
        "core": {
            "protocol_version": "0.1.0-experimental",
            "acceptance_contract": contract,
            "acceptance_contract_hash": cli._core_acceptance_contract_hash(contract),
        },
        "freshness": {
            "created_at": "2026-10-13T00:00:00Z",
            "valid_until": valid_until,
            "revalidate_before_first_effect": True,
        },
    }
    binding["binding_hash"] = cli._core_hash(binding)
    selected = {
        "admission_evidence_hash": "a" * 64,
        "admission_evidence_ref": "admission.json",
        "model": "google_genai/gemini-test",
        "provider": "google_genai",
        "role_ceiling": "bounded_candidate_generation",
        "selection_evidence_hash": "d" * 64,
        "selection_evidence_ref": "selection.json",
        "worker_id": "worker-1",
    }
    envelope = {
        "schema": "external_execution_envelope.v2",
        "binding": {
            "context_pack_sha256": "c" * 64,
            "item_id": "42",
            "item_type": "issue",
            "main_sha": base,
            "repository": "James3014/Nexus-new",
            "revision": "r1",
            "task_card_ref": CARD_REF,
            "task_card_hash": cli._sha256(card.read_bytes()),
        },
        "repository_mutation_binding": binding,
        "diagnosis": {
            "status": "PROVEN",
            "hypothesis": "the required target is absent",
            "next_probe": "inspect the absent target",
        },
        "definition_of_done": ["one bounded test"],
        "evidence_refs": [
            f"task_card:{CARD_REF}@" + "a" * 16,
            f"source_absence:{TARGET}@" + "b" * 16,
        ],
        "inspect_first": [f"task_card:{CARD_REF}@" + "a" * 16],
        "failure_guards": ["no other paths"],
        "implementation_direction": ["create one test"],
        "objective": "bounded repair",
        "required_semantics": ["one deterministic test"],
        "scope_signal": {
            "conditional_migration_paths": [],
            "forbidden_paths": [],
            "max_files": 1,
            "production_edit_paths": [],
            "read_only_authorities": [CARD_REF],
            "required_test_edit_paths": [TARGET],
            "scope_block_conditions": ["no extra files"],
            "scope_confidence": "HIGH",
            "verification_only_paths": [TARGET],
        },
        "selected_worker": selected,
        "stop_and_escalate": ["stop"],
        "verification_focus": ["pytest"],
    }
    raw = cli._canonical_json(envelope)
    artifact = tmp_path / "bound-envelope.json"
    artifact.write_text(raw, encoding="utf-8")
    return {
        "schema": cli.REQUEST_SCHEMA,
        "operation": "worker_run",
        "operation_id": INITIAL_OPERATION,
        "attempt_id": "attempt-test-1",
        "provider_id": "google_genai",
        "model_id": "gemini-test",
        "runtime_state_root": str(tmp_path / "state"),
        "workspace_path": str(workspace),
        "artifact_path": str(artifact),
        "prompt": "\n".join(
            [
                "task_id=task-1",
                "unit_id=u1",
                f"authorized_mutation_paths={json.dumps([TARGET])}",
                f"envelope_sha256={cli._sha256(raw)}",
                f"expected_base_sha={base}",
            ]
        ),
        "session_id": "",
        "worker_identity": dict(selected),
        "worker_identity_sha256": cli._sha256(cli._canonical_json(selected)),
        "core_binding_hash": binding["binding_hash"],
        "acceptance_contract_hash": binding["core"]["acceptance_contract_hash"],
    }


def _capture_candidate(workspace: Path) -> str:
    """What Nexus fanout does after a COMPLETED unit: commit the in-scope candidate."""
    _git(workspace, "add", "--", TARGET)
    _git(workspace, "commit", "-q", "-m", "candidate")
    return _git(workspace, "rev-parse", "HEAD")


def _write_delta(
    path: Path,
    initial: dict[str, Any],
    session_id: str,
    parent: str,
    overrides: dict[str, Any] | None = None,
):
    delta = {
        "schema": "unit_repair_delta.v2",
        "task_id": "task-1",
        "unit_id": "u1",
        "repair_index": 1,
        "parent_receipt_id": "e" * 64,
        "parent_candidate_commit": parent,
        "verification_id": "f" * 64,
        "session_id": session_id,
        "workspace_id": "ws-1",
        "workspace_path": initial["workspace_path"],
        "provider": "google_genai",
        "model": "google_genai/gemini-test",
        "allowed_mutation_paths": [TARGET],
        "findings": [{"verifier_id": "pytest", "returncode": 1}],
        "stop_conditions": ["scope_expands"],
        "claim_ceiling": "BOUNDED_REPAIR_ONLY",
        **(overrides or {}),
    }
    raw = json.dumps(delta, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    return cli._sha256(raw)


def _continue_request(
    tmp_path: Path,
    initial: dict[str, Any],
    session_id: str,
    parent: str,
    delta_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delta_path = tmp_path / "repair-delta.json"
    delta_sha = _write_delta(delta_path, initial, session_id, parent, delta_overrides)
    request = dict(initial)
    request.update(
        operation="worker_continue",
        operation_id=CONTINUE_OPERATION,
        session_id=session_id,
        artifact_path=str(delta_path),
        prompt="\n".join(
            [
                "Continue the exact same execution unit session for a bounded repair.",
                "task_id=task-1",
                "unit_id=u1",
                "repair_id=d001",
                f"repair_artifact_ref={delta_path}",
                f"repair_sha256={delta_sha}",
                f"parent_candidate_commit={parent}",
            ]
        ),
    )
    request.pop("attempt_id")
    return request


@pytest.fixture
def bound_session(tmp_path):
    initial = _bound_initial_request(tmp_path)
    calls = _Calls()
    result = cli._worker_run(initial, **_factories(calls, "VALUE = 1\n"))
    assert result["status"] == "COMPLETED", result
    assert result["core_binding_status"] == "BOUND"
    assert calls.as_tuple() == (1, 0, 1)
    parent = _capture_candidate(Path(initial["workspace_path"]))
    return SimpleNamespace(initial=initial, result=result, parent=parent)


def _run_continue(request: dict[str, Any], content: str = "VALUE = 2\n") -> tuple[dict, _Calls]:
    calls = _Calls()
    return cli._worker_run(request, **_factories(calls, content)), calls


def test_bound_continuation_keeps_same_core_identity_and_reconciles_with_hashes(
    tmp_path, bound_session
):
    initial, first = bound_session.initial, bound_session.result
    request = _continue_request(tmp_path, initial, first["session_id"], bound_session.parent)

    result, calls = _run_continue(request)

    assert result["status"] == "COMPLETED", result
    assert result["operation_id"] == CONTINUE_OPERATION
    assert result["session_id"] == first["session_id"]
    assert result["core_binding_status"] == "BOUND"
    assert result["core_binding_hash"] == first["core_binding_hash"]
    assert result["acceptance_contract_hash"] == first["acceptance_contract_hash"]
    assert calls.as_tuple() == (2, 1, 1)
    projection = result["core_change_set_projection"]
    assert projection["status"] == "PHYSICAL_CHANGESET_BOUND"
    assert projection["core_binding_hash"] == first["core_binding_hash"]
    assert projection["operation_id"] == CONTINUE_OPERATION
    assert projection["attempt_id"] == "attempt-test-1"
    assert projection["change_set"]["paths"] == [TARGET]
    assert (
        projection["change_set"]["source_revision"]
        == (first["core_change_set_projection"]["change_set"]["source_revision"])
    )
    workspace = Path(initial["workspace_path"])
    assert (workspace / TARGET).read_text(encoding="utf-8") == "VALUE = 2\n"
    assert _git(workspace, "rev-parse", "HEAD") == bound_session.parent

    reconcile = dict(request)
    reconcile.update(operation="worker_reconcile", prompt="", artifact_path="")
    assert cli.dispatch(reconcile) == result
    for hashes in (
        {"core_binding_hash": "", "acceptance_contract_hash": ""},
        {"core_binding_hash": "sha256:" + "9" * 64},
    ):
        mismatched = cli.dispatch({**reconcile, **hashes})
        assert mismatched["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
        assert mismatched["outcome_unknown"] is True

    # A second continuation stacks on the BOUND continuation, never on LEGACY_UNBOUND.
    second_parent = _capture_candidate(workspace)
    second = _continue_request(tmp_path, initial, first["session_id"], second_parent)
    second["operation_id"] = "3" * 64
    again, _calls = _run_continue(second, "VALUE = 3\n")
    assert again["status"] == "COMPLETED", again
    assert again["core_binding_status"] == "BOUND"
    assert again["core_binding_hash"] == first["core_binding_hash"]


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda r: r.pop("core_binding_hash"), "CORE_BINDING_ABSENT"),
        (
            lambda r: r.update(core_binding_hash="", acceptance_contract_hash=""),
            "CORE_BINDING_ABSENT",
        ),
        (lambda r: r.update(core_binding_hash="sha256:" + "9" * 64), "CORE_BINDING_HASH"),
        (lambda r: r.update(acceptance_contract_hash="sha256:" + "9" * 64), "CORE_BINDING_HASH"),
        (
            lambda r: Path(r["artifact_path"]).write_text(
                Path(r["artifact_path"]).read_text(encoding="utf-8").replace("d001", "d002") + " ",
                encoding="utf-8",
            ),
            "ARTIFACT_HASH",
        ),
        (
            lambda r: r.update(
                prompt="\n".join(
                    line for line in r["prompt"].splitlines() if not line.startswith("repair_sha")
                )
            ),
            "ARTIFACT_HASH",
        ),
    ],
)
def test_bound_continuation_rejects_absent_mismatched_or_tampered_binding(
    tmp_path, bound_session, mutate, reason
):
    request = _continue_request(
        tmp_path, bound_session.initial, bound_session.result["session_id"], bound_session.parent
    )
    mutate(request)
    result, calls = _run_continue(request)
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert result["failure_phase"] == "SEMANTIC_ADMISSION"
    assert result["error_code"] == f"OPEN_SWE_BOUND_CONTINUATION_REJECTED_{reason}"
    assert result["core_binding_status"] == "LEGACY_UNBOUND"
    assert calls.as_tuple() == (0, 0, 0)
    assert (Path(request["workspace_path"]) / TARGET).read_text(encoding="utf-8") == "VALUE = 1\n"


@pytest.mark.parametrize(
    "delta_overrides,reason",
    [
        ({"schema": "external_execution_envelope.v1"}, "ARTIFACT_SCHEMA"),
        ({"session_id": "ses_open_swe_00000000000000000000"}, "ARTIFACT_SESSION"),
        ({"unit_id": "u2"}, "ARTIFACT_SESSION"),
        ({"workspace_path": "/tmp/other-workspace"}, "ARTIFACT_SESSION"),
        ({"allowed_mutation_paths": [TARGET, "b.py"]}, "ARTIFACT_SCOPE"),
        ({"parent_candidate_commit": "0" * 40}, "ARTIFACT_PARENT"),
    ],
)
def test_bound_continuation_rejects_artifact_for_another_session_scope_or_parent(
    tmp_path, bound_session, delta_overrides, reason
):
    request = _continue_request(
        tmp_path,
        bound_session.initial,
        bound_session.result["session_id"],
        bound_session.parent,
        delta_overrides,
    )
    result, calls = _run_continue(request)
    assert result["error_code"] == f"OPEN_SWE_BOUND_CONTINUATION_REJECTED_{reason}"
    assert calls.as_tuple() == (0, 0, 0)


def test_bound_continuation_rejects_workspace_drift(tmp_path, bound_session):
    workspace = Path(bound_session.initial["workspace_path"])
    session_id = bound_session.result["session_id"]

    (workspace / TARGET).write_text("VALUE = 'dirty'\n", encoding="utf-8")
    result, calls = _run_continue(
        _continue_request(tmp_path, bound_session.initial, session_id, bound_session.parent)
    )
    assert result["error_code"] == "OPEN_SWE_BOUND_CONTINUATION_REJECTED_WORKSPACE_DIRTY"
    assert calls.as_tuple() == (0, 0, 0)


def test_bound_continuation_rejects_out_of_scope_committed_change(tmp_path, bound_session):
    workspace = Path(bound_session.initial["workspace_path"])
    (workspace / "b.py").write_text("OTHER = 1\n", encoding="utf-8")
    _git(workspace, "add", "b.py")
    _git(workspace, "commit", "-q", "-m", "widen")
    head = _git(workspace, "rev-parse", "HEAD")
    result, calls = _run_continue(
        _continue_request(tmp_path, bound_session.initial, bound_session.result["session_id"], head)
    )
    assert result["error_code"] == "OPEN_SWE_BOUND_CONTINUATION_REJECTED_WORKSPACE_SCOPE"
    assert calls.as_tuple() == (0, 0, 0)


def test_bound_continuation_requires_prior_completed_bound_terminal(tmp_path, bound_session):
    session_id = bound_session.result["session_id"]
    tampered = _continue_request(tmp_path, bound_session.initial, session_id, bound_session.parent)
    tampered["core_binding_hash"] = "sha256:" + "9" * 64
    rejected, _calls = _run_continue(tampered)
    assert rejected["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"

    # The workspace's last terminal is now an unknown outcome: fail closed.
    legit = _continue_request(tmp_path, bound_session.initial, session_id, bound_session.parent)
    legit["operation_id"] = "4" * 64
    result, calls = _run_continue(legit)
    assert result["error_code"] == "OPEN_SWE_BOUND_CONTINUATION_REJECTED_PRIOR_BOUND_TERMINAL"
    assert calls.as_tuple() == (0, 0, 0)


def test_bound_continuation_cannot_reuse_another_operation_id(tmp_path, bound_session):
    request = _continue_request(
        tmp_path, bound_session.initial, bound_session.result["session_id"], bound_session.parent
    )
    request["operation_id"] = INITIAL_OPERATION
    result, calls = _run_continue(request)
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert result["outcome_unknown"] is True
    assert calls.as_tuple() == (0, 0, 0)


def test_bound_continuation_rejects_partial_session_core_identity(tmp_path, bound_session):
    session_id = bound_session.result["session_id"]
    request = _continue_request(tmp_path, bound_session.initial, session_id, bound_session.parent)
    session_path = cli._session_path(request, session_id)
    context = json.loads(session_path.read_text(encoding="utf-8"))
    context.pop("core_source_tree")
    session_path.write_text(json.dumps(context), encoding="utf-8")
    result, calls = _run_continue(request)
    assert result["error_code"] == "OPEN_SWE_BOUND_CONTINUATION_REJECTED_SESSION_CORE_BINDING"
    assert calls.as_tuple() == (0, 0, 0)


def _legacy_session(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    workspace = tmp_path / "legacy"
    workspace.mkdir()
    (workspace / TARGET).write_text("VALUE = 1\n", encoding="utf-8")
    artifact = tmp_path / "legacy-evidence.json"
    artifact.write_text(
        '{"schema":"external_execution_envelope.v1","failure":"VALUE must be 2"}\n',
        encoding="utf-8",
    )
    initial = {
        "schema": cli.REQUEST_SCHEMA,
        "operation": "worker_run",
        "operation_id": "5" * 64,
        "provider_id": "google_genai",
        "model_id": "gemini-test",
        "runtime_state_root": str(tmp_path / "state"),
        "workspace_path": str(workspace),
        "artifact_path": str(artifact),
        "prompt": "\n".join(
            [
                "task_id=task-1",
                "unit_id=u1",
                f"authorized_mutation_paths={json.dumps([TARGET])}",
                "bounded repair",
            ]
        ),
        "session_id": "",
        "worker_identity_sha256": "c" * 64,
    }
    result, _calls = _run_continue(initial)
    assert result["status"] == "COMPLETED", result
    assert result["core_binding_status"] == "LEGACY_UNBOUND"
    return initial, result


def test_unbound_session_continuation_is_unchanged_and_cannot_claim_bound(tmp_path):
    initial, first = _legacy_session(tmp_path)
    continuation = dict(initial)
    continuation.update(
        operation="worker_continue", operation_id="6" * 64, session_id=first["session_id"]
    )
    result, calls = _run_continue(continuation, "VALUE = 3\n")
    assert result["status"] == "COMPLETED", result
    assert result["core_binding_status"] == "LEGACY_UNBOUND"
    assert result["core_change_set_projection"] == {"status": "LEGACY_UNBOUND"}
    assert calls.as_tuple() == (2, 1, 1)

    claimed = dict(continuation)
    claimed.update(
        operation_id="7" * 64,
        core_binding_hash="sha256:" + "1" * 64,
        acceptance_contract_hash="sha256:" + "2" * 64,
    )
    result, calls = _run_continue(claimed, "VALUE = 4\n")
    assert result["error_code"] == "OPEN_SWE_BOUND_CONTINUATION_REJECTED_SESSION_CORE_BINDING"
    assert result["core_binding_status"] == "LEGACY_UNBOUND"
    assert calls.as_tuple() == (0, 0, 0)


def test_recovery_identity_omits_empty_continuation_base_for_existing_identities():
    from nexus_open_swe_runtime.recovery import RecoveryIdentity

    fields = {
        "operation_id": "a" * 64,
        "execution_material_sha256": "b" * 64,
        "workspace": "/w",
        "task_id": "task-1",
        "unit_id": "u1",
        "session_id": "s",
        "allowed_paths": (TARGET,),
        "provider_id": "p",
        "model_id": "m",
        "worker_identity_sha256": "c" * 64,
        "transport_config_sha256": "d" * 64,
        "runtime_identity_sha256": "e" * 64,
    }
    initial = RecoveryIdentity(**fields)
    assert "core_continuation_base" not in initial.as_dict()
    continued = RecoveryIdentity(**fields, core_continuation_base="f" * 40)
    assert continued.as_dict()["core_continuation_base"] == "f" * 40
    assert continued.digest() != initial.digest()


def test_session_persists_binding_valid_until_and_expired_binding_rejects(tmp_path, bound_session):
    session_id = bound_session.result["session_id"]
    request = _continue_request(tmp_path, bound_session.initial, session_id, bound_session.parent)
    session_path = cli._session_path(request, session_id)
    context = json.loads(session_path.read_text(encoding="utf-8"))
    envelope = json.loads(Path(bound_session.initial["artifact_path"]).read_text("utf-8"))
    persisted = envelope["repository_mutation_binding"]["freshness"]["valid_until"]
    assert context["core_valid_until"] == persisted

    context["core_valid_until"] = "2026-01-01T00:00:00Z"
    session_path.write_text(json.dumps(context), encoding="utf-8")
    result, calls = _run_continue(request)
    assert result["error_code"] == "OPEN_SWE_BOUND_CONTINUATION_REJECTED_CORE_BINDING_EXPIRED"
    assert calls.as_tuple() == (0, 0, 0)
    assert (Path(request["workspace_path"]) / TARGET).read_text(encoding="utf-8") == "VALUE = 1\n"


def test_session_without_binding_deadline_cannot_be_continued(tmp_path):
    initial = _bound_initial_request(tmp_path, valid_until=None)
    first, _calls = _run_continue(initial, "VALUE = 1\n")
    assert first["status"] == "COMPLETED", first
    context = json.loads(
        cli._session_path(initial, first["session_id"]).read_text(encoding="utf-8")
    )
    assert "core_valid_until" not in context
    parent = _capture_candidate(Path(initial["workspace_path"]))
    result, calls = _run_continue(_continue_request(tmp_path, initial, first["session_id"], parent))
    assert result["error_code"] == "OPEN_SWE_BOUND_CONTINUATION_REJECTED_CORE_BINDING_FRESHNESS"
    assert calls.as_tuple() == (0, 0, 0)


@pytest.mark.parametrize("variant", ["extra_commit", "different_content"])
def test_parent_tree_must_equal_prior_projected_target_tree(tmp_path, variant):
    initial = _bound_initial_request(tmp_path)
    first, _calls = _run_continue(initial, "VALUE = 1\n")
    assert first["status"] == "COMPLETED", first
    workspace = Path(initial["workspace_path"])
    if variant == "different_content":
        # An in-scope change made outside the bound operation before capture.
        (workspace / TARGET).write_text("VALUE = 'unjournaled'\n", encoding="utf-8")
        parent = _capture_candidate(workspace)
    else:
        _capture_candidate(workspace)
        (workspace / TARGET).write_text("VALUE = 'unjournaled'\n", encoding="utf-8")
        parent = _capture_candidate(workspace)
    result, calls = _run_continue(_continue_request(tmp_path, initial, first["session_id"], parent))
    assert result["error_code"] == "OPEN_SWE_BOUND_CONTINUATION_REJECTED_PARENT_TREE"
    assert calls.as_tuple() == (0, 0, 0)
    assert (workspace / TARGET).read_text(encoding="utf-8") == "VALUE = 'unjournaled'\n"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda prior: prior.pop("core_change_set_projection"),
        lambda prior: prior["core_change_set_projection"].update(status="NO_PHYSICAL_CHANGESET"),
        lambda prior: prior["core_change_set_projection"].update(
            core_binding_hash="sha256:" + "9" * 64
        ),
        lambda prior: prior["core_change_set_projection"]["change_manifest"].update(
            target_tree="not-a-tree"
        ),
        lambda prior: prior["core_change_set_projection"].pop("change_manifest"),
        lambda prior: prior["core_change_set_projection"]["change_set"].update(
            source_revision="git-commit:" + "0" * 40
        ),
    ],
    ids=["missing", "status", "hash", "target_tree", "no_manifest", "source"],
)
def test_missing_or_malformed_prior_projection_rejects(tmp_path, bound_session, mutate):
    request = _continue_request(
        tmp_path, bound_session.initial, bound_session.result["session_id"], bound_session.parent
    )
    index_path = cli._workspace_index_path(request)
    prior = json.loads(index_path.read_text(encoding="utf-8"))
    mutate(prior)
    index_path.write_text(json.dumps(prior), encoding="utf-8")
    result, calls = _run_continue(request)
    assert result["error_code"] == "OPEN_SWE_BOUND_CONTINUATION_REJECTED_PRIOR_PROJECTION"
    assert calls.as_tuple() == (0, 0, 0)
