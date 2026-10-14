"""Core-bound edits to existing Task Card-listed files (#130).

Owner decision (James, 2026-10-14): the bound path may modify files the Task Card
explicitly lists, including production code.  Deletions, renames, conditional
migrations and any file not on the list stay forbidden.  These tests use a real
Git workspace so admission, the scoped backend and the Core change-set projection
are exercised against physical repository state.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from nexus_open_swe_runtime import cli

ORIGIN = "git@github.com:James3014/Nexus-new.git"
CARD_REF = "tasks/task-card.md"
OPERATION = "1" * 64


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


def _future(seconds: int = 3600) -> str:
    moment = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _card(paths: list[str]) -> str:
    return "\n".join(
        [
            "- task_id: `task-1`",
            "- status: `ACTIVE`",
            "- worker_may_approve: `false`",
            "- worker_may_integrate: `false`",
            "- worker_may_push: `false`",
            "- AUTO_CHAIN: `false`",
            "- allow_deletions: `false`",
            "\n## Allowed files\n",
            *(f"- `{path}`" for path in paths),
        ]
    )


def _bound_request(
    tmp_path: Path,
    *,
    base_files: dict[str, str],
    test_paths: list[str],
    production_paths: list[str],
    card_paths: list[str] | None = None,
    migration_paths: list[str] | None = None,
    refs: list[str] | None = None,
    prepare: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Build a real-git Core-bound v2 worker request.

    ``card_paths`` defaults to ``test_paths + production_paths`` and is also the
    authorized mutation path list and the Core contract's ``allowed_paths``.
    """
    card_paths = list(card_paths if card_paths is not None else [*test_paths, *production_paths])
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "test@example.com")
    _git(workspace, "config", "user.name", "Test")
    _git(workspace, "remote", "add", "origin", ORIGIN)
    card = workspace / CARD_REF
    card.parent.mkdir(parents=True)
    card.write_text(_card(card_paths), encoding="utf-8")
    for path, content in base_files.items():
        target = workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    if prepare is not None:
        prepare(workspace)
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-q", "-m", "base")
    base = _git(workspace, "rev-parse", "HEAD")
    base_tree = _git(workspace, "rev-parse", "HEAD^{tree}")
    contract = {
        "contract_id": "open-swe-test-contract",
        "requirements_hash": "sha256:" + "1" * 64,
        "required_verifier_ids": ["pytest"],
        "allowed_paths": card_paths,
        "deletion_policy": "FORBID",
    }
    binding: dict[str, Any] = {
        "schema": "nexus.repository_mutation_binding.v1",
        "binding_id": "binding-test-1",
        "operation_id": OPERATION,
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
            "created_at": "2026-10-14T00:00:00Z",
            "valid_until": _future(),
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
    if refs is None:
        refs = [f"task_card:{CARD_REF}@" + "a" * 16]
        for path in card_paths:
            kind = "source_file" if (workspace / path).exists() else "source_absence"
            refs.append(f"{kind}:{path}@" + "b" * 16)
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
            "hypothesis": "the listed production code returns the wrong value",
            "next_probe": "inspect the listed source",
        },
        "definition_of_done": ["bounded fix"],
        "evidence_refs": refs,
        "inspect_first": [f"task_card:{CARD_REF}@" + "a" * 16],
        "failure_guards": ["no other paths"],
        "implementation_direction": ["edit the listed files"],
        "objective": "bounded repair",
        "required_semantics": ["deterministic behaviour"],
        "scope_signal": {
            "conditional_migration_paths": list(migration_paths or []),
            "forbidden_paths": [],
            "max_files": len(card_paths),
            "production_edit_paths": list(production_paths),
            "read_only_authorities": [CARD_REF],
            "required_test_edit_paths": list(test_paths),
            "scope_block_conditions": ["no extra files"],
            "scope_confidence": "HIGH",
            "verification_only_paths": list(card_paths),
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
        "operation_id": OPERATION,
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
                f"authorized_mutation_paths={json.dumps(card_paths)}",
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


def _admission(request: dict[str, Any]) -> cli.SemanticAdmission:
    paths = tuple(
        json.loads(cli._prompt_field(request["prompt"], "authorized_mutation_paths"))
    )
    return cli._semantic_v2_admission(
        request,
        Path(request["workspace_path"]).resolve(),
        Path(request["artifact_path"]),
        request["prompt"],
        paths,
    )


def _run(
    request: dict[str, Any],
    writes: dict[str, str],
    extra_effect: Callable[[Path], None] | None = None,
) -> tuple[dict[str, Any], list[int]]:
    """Run the bound worker; the fake repair graph journals ``writes`` like the backend."""
    calls = [0, 0, 0]

    def model_factory(*_args):
        calls[0] += 1
        return SimpleNamespace(_conversation_id=None)

    def diagnosis_factory(*_args, **_kwargs):
        calls[1] += 1
        raise AssertionError("an admitted bound packet must skip diagnosis")

    def repair_factory(
        _model, workspace, _runtime, allowed_paths, _key, checkpointer=None, effect_journal=None
    ):
        calls[2] += 1
        assert effect_journal is not None

        def effect() -> None:
            for index, (path, content) in enumerate(writes.items()):
                target = Path(workspace) / path
                old = target.read_text(encoding="utf-8") if target.exists() else None
                written = effect_journal.intent(
                    turn_id="turn-1",
                    tool_call_id=f"call-{index}",
                    tool_name="edit_file" if old is not None else "write_file",
                    arguments={"file_path": path, "content": content},
                    path=target,
                    preimage=old,
                    postimage=content,
                )
                effect_journal.recover_write(written)
            if extra_effect is not None:
                extra_effect(Path(workspace))

        return _Graph(
            cli.REPAIR_TOOLS, _record("record_worker_result", {"summary": "repaired"}), effect
        )

    result = cli._worker_run(
        request,
        runtime_loader=lambda: {"human_message": lambda content: content},
        model_factory=model_factory,
        diagnosis_factory=diagnosis_factory,
        repair_factory=repair_factory,
    )
    return result, calls


# --- admitted -----------------------------------------------------------------


def test_edit_existing_production_file_is_admitted_and_bound(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=[],
        production_paths=["pkg/core.py"],
    )
    admission = _admission(request)
    assert admission.decision == cli.ADMIT, admission.reason_code

    result, calls = _run(request, {"pkg/core.py": "VALUE = 2\n"})

    assert result["status"] == "COMPLETED", result
    assert result["core_binding_status"] == "BOUND"
    assert result["diagnosis_status"] == "ROOT_CAUSE_SUPPORTED"
    assert calls == [1, 0, 1]
    projection = result["core_change_set_projection"]
    assert projection["status"] == "PHYSICAL_CHANGESET_BOUND"
    assert projection["change_set"]["paths"] == ["pkg/core.py"]
    assert projection["change_set"]["deleted_paths"] == []
    (entry,) = projection["change_manifest"]["entries"]
    workspace = Path(request["workspace_path"])
    base = cli._prompt_field(request["prompt"], "expected_base_sha")
    assert entry["change_type"] == "MODIFY"
    assert entry["before_oid"] == _git(workspace, "rev-parse", f"{base}:pkg/core.py")
    assert entry["after_oid"] == _git(workspace, "hash-object", "pkg/core.py")
    assert entry["before_mode"] == entry["after_mode"] == "100644"
    # The attested target tree is exactly base + the edited blob.
    _git(workspace, "add", "-A")
    assert projection["change_set"]["target_revision"] == (
        "git-tree:" + _git(workspace, "write-tree")
    )


def test_mixed_new_test_and_existing_production_is_admitted_and_bound(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=["tests/test_core.py"],
        production_paths=["pkg/core.py"],
    )
    admission = _admission(request)
    assert admission.decision == cli.ADMIT, admission.reason_code

    result, _calls = _run(
        request,
        {
            "tests/test_core.py": "def test_value():\n    assert True\n",
            "pkg/core.py": "VALUE = 2\n",
        },
    )

    assert result["status"] == "COMPLETED", result
    assert result["core_binding_status"] == "BOUND"
    projection = result["core_change_set_projection"]
    assert projection["status"] == "PHYSICAL_CHANGESET_BOUND"
    entries = {row["path"]: row for row in projection["change_manifest"]["entries"]}
    assert entries["pkg/core.py"]["change_type"] == "MODIFY"
    assert entries["pkg/core.py"]["before_oid"] is not None
    assert entries["tests/test_core.py"]["change_type"] == "ADD"
    assert entries["tests/test_core.py"]["before_oid"] is None


def test_union_order_need_not_match_card_order(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=["tests/test_core.py"],
        production_paths=["pkg/core.py"],
        card_paths=["pkg/core.py", "tests/test_core.py"],
    )
    assert _admission(request).decision == cli.ADMIT


# --- rejected at admission ------------------------------------------------------


def test_existing_file_not_on_the_list_is_rejected(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n", "pkg/other.py": "OTHER = 1\n"},
        test_paths=[],
        production_paths=["pkg/other.py"],
        card_paths=["pkg/core.py"],
    )
    admission = _admission(request)
    assert admission.decision == cli.REJECT
    assert admission.reason_code == "scope_paths"


def test_existing_file_beyond_the_list_is_rejected(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n", "pkg/other.py": "OTHER = 1\n"},
        test_paths=[],
        production_paths=["pkg/core.py", "pkg/other.py"],
        card_paths=["pkg/core.py"],
    )
    admission = _admission(request)
    assert admission.decision == cli.REJECT
    assert admission.reason_code == "scope_paths"


def test_path_listed_as_both_test_and_production_is_rejected(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=["pkg/core.py"],
        production_paths=["pkg/core.py"],
        card_paths=["pkg/core.py"],
    )
    admission = _admission(request)
    assert admission.decision == cli.REJECT
    assert admission.reason_code == "scope_paths"


def test_conditional_migration_is_still_rejected(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=[],
        production_paths=["pkg/core.py"],
        migration_paths=["pkg/core.py"],
    )
    admission = _admission(request)
    assert admission.decision == cli.REJECT
    assert admission.reason_code == "scope_migration"


def test_symlink_target_is_rejected(tmp_path):
    def prepare(workspace: Path) -> None:
        (workspace / "pkg").mkdir()
        (workspace / "pkg" / "real.py").write_text("VALUE = 1\n", encoding="utf-8")
        os.symlink("real.py", workspace / "pkg" / "core.py")

    request = _bound_request(
        tmp_path,
        base_files={},
        test_paths=[],
        production_paths=["pkg/core.py"],
        refs=[f"task_card:{CARD_REF}@" + "a" * 16, "source_file:pkg/core.py@" + "b" * 16],
        prepare=prepare,
    )
    admission = _admission(request)
    assert admission.decision == cli.REJECT
    assert admission.reason_code == "path_symlink"


def test_symlinked_parent_escaping_the_workspace_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "core.py").write_text("VALUE = 1\n", encoding="utf-8")

    def prepare(workspace: Path) -> None:
        os.symlink(str(outside), workspace / "pkg")

    request = _bound_request(
        tmp_path,
        base_files={},
        test_paths=[],
        production_paths=["pkg/core.py"],
        refs=[f"task_card:{CARD_REF}@" + "a" * 16, "source_file:pkg/core.py@" + "b" * 16],
        prepare=prepare,
    )
    admission = _admission(request)
    assert admission.decision == cli.REJECT
    assert admission.reason_code == "path_symlink"


def test_dot_dot_path_escaping_the_workspace_is_rejected(tmp_path):
    (tmp_path / "escape.py").write_text("VALUE = 1\n", encoding="utf-8")
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=[],
        production_paths=["../escape.py"],
        card_paths=["pkg/core.py"],
    )
    admission = _admission(request)
    assert admission.decision == cli.REJECT
    assert admission.reason_code == "predicate_exception"


def test_directory_target_is_rejected(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=[],
        production_paths=["pkg"],
        refs=[f"task_card:{CARD_REF}@" + "a" * 16, "source_file:pkg@" + "b" * 16],
    )
    admission = _admission(request)
    assert admission.decision == cli.REJECT
    assert admission.reason_code == "path_not_regular"


def test_existing_target_requires_source_file_evidence(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=[],
        production_paths=["pkg/core.py"],
        refs=[f"task_card:{CARD_REF}@" + "a" * 16, "source_absence:pkg/core.py@" + "b" * 16],
    )
    admission = _admission(request)
    assert admission.decision == cli.REJECT
    assert admission.reason_code == "source_evidence"


def test_existing_target_rejects_unanchored_source_file_evidence(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=[],
        production_paths=["pkg/core.py"],
        refs=[f"task_card:{CARD_REF}@" + "a" * 16, "source_file:pkg/core.py@not-an-anchor"],
    )
    admission = _admission(request)
    assert admission.decision == cli.REJECT
    assert admission.reason_code == "source_evidence"


def test_new_target_still_requires_source_absence_evidence(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={},
        test_paths=["tests/test_core.py"],
        production_paths=[],
        refs=[f"task_card:{CARD_REF}@" + "a" * 16, "source_file:tests/test_core.py@" + "b" * 16],
    )
    admission = _admission(request)
    assert admission.decision == cli.REJECT
    assert admission.reason_code == "source_evidence"


def test_existing_target_must_be_tracked_at_the_bound_base(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=[],
        production_paths=["pkg/core.py", "pkg/ignored.py"],
        refs=[
            f"task_card:{CARD_REF}@" + "a" * 16,
            "source_file:pkg/core.py@" + "b" * 16,
            "source_file:pkg/ignored.py@" + "b" * 16,
        ],
    )
    workspace = Path(request["workspace_path"])
    # An ignored, untracked file is invisible to `git status` but is not base content.
    (workspace / ".git" / "info" / "exclude").write_text("pkg/ignored.py\n", encoding="utf-8")
    (workspace / "pkg" / "ignored.py").write_text("SECRET = 1\n", encoding="utf-8")
    admission = _admission(request)
    assert admission.decision == cli.REJECT
    assert admission.reason_code == "source_untracked"


# --- deletion / rename / out-of-list at the Core change-set projection -----------


def _deleting(path: str) -> Callable[[Path], None]:
    return lambda workspace: (workspace / path).unlink()


def test_deleting_a_listed_existing_file_is_rejected(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=["tests/test_core.py"],
        production_paths=["pkg/core.py"],
    )
    result, _calls = _run(
        request,
        {"tests/test_core.py": "def test_value():\n    assert True\n"},
        extra_effect=_deleting("pkg/core.py"),
    )
    assert result["status"] != "COMPLETED"
    assert result["error_code"] == "OPEN_SWE_CORE_CHANGESET_DELETION_FORBIDDEN"


def test_renaming_between_listed_files_is_rejected(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=["pkg/renamed.py"],
        production_paths=["pkg/core.py"],
    )

    def rename(workspace: Path) -> None:
        (workspace / "pkg" / "core.py").rename(workspace / "pkg" / "renamed.py")

    result, _calls = _run(request, {}, extra_effect=rename)
    assert result["status"] != "COMPLETED"
    assert result["error_code"] == "OPEN_SWE_CORE_CHANGESET_DELETION_FORBIDDEN"


def test_renaming_a_listed_file_off_the_list_is_rejected(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=[],
        production_paths=["pkg/core.py"],
    )

    def rename(workspace: Path) -> None:
        (workspace / "pkg" / "core.py").rename(workspace / "pkg" / "elsewhere.py")

    result, _calls = _run(request, {}, extra_effect=rename)
    assert result["status"] != "COMPLETED"
    assert result["error_code"] == "OPEN_SWE_CORE_CHANGESET_SCOPE_MISMATCH"


def test_editing_an_unlisted_existing_file_is_rejected_at_projection(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n", "pkg/other.py": "OTHER = 1\n"},
        test_paths=[],
        production_paths=["pkg/core.py"],
    )

    def touch_other(workspace: Path) -> None:
        (workspace / "pkg" / "other.py").write_text("OTHER = 2\n", encoding="utf-8")

    result, _calls = _run(request, {"pkg/core.py": "VALUE = 2\n"}, extra_effect=touch_other)
    assert result["status"] != "COMPLETED"
    assert result["error_code"] == "OPEN_SWE_CORE_CHANGESET_SCOPE_MISMATCH"


# --- scoped worker backend --------------------------------------------------------


class _Delegate:
    def __getattr__(self, name: str) -> Any:  # pragma: no cover - must never be reached
        raise AssertionError(f"delegate.{name} must not be reachable")


def _backend(tmp_path: Path, allowed: tuple[str, ...]) -> tuple[cli.ScopedRepairBackend, Path]:
    root = tmp_path / "root"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "pkg" / "other.py").write_text("OTHER = 1\n", encoding="utf-8")
    return cli.ScopedRepairBackend(_Delegate(), root, allowed), root


def test_backend_exposes_no_delete_or_rename_surface(tmp_path):
    backend, _root = _backend(tmp_path, ("pkg/core.py",))
    for name in (
        "delete", "adelete", "remove", "aremove", "unlink", "rename", "arename",
        "move", "amove", "rmdir", "execute", "aexecute", "upload_files", "download_files",
    ):
        assert not hasattr(backend, name), name


def test_backend_forbids_unlisted_existing_file(tmp_path):
    backend, root = _backend(tmp_path, ("pkg/core.py",))
    with pytest.raises(PermissionError, match="OPEN_SWE_MUTATION_PATH_FORBIDDEN"):
        backend.edit("pkg/other.py", "OTHER = 1", "OTHER = 2")
    with pytest.raises(PermissionError, match="OPEN_SWE_MUTATION_PATH_FORBIDDEN"):
        backend.write("pkg/other.py", "OTHER = 2\n")
    assert (root / "pkg" / "other.py").read_text(encoding="utf-8") == "OTHER = 1\n"


def test_backend_forbids_symlink_and_escape_and_directory_targets(tmp_path):
    with pytest.raises(cli.RuntimeErrorBounded, match="OPEN_SWE_PATH_INVALID"):
        cli.ScopedRepairBackend(_Delegate(), tmp_path, ("../escape.py",))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    backend, root = _backend(tmp_path, ("pkg/link.py", "pkg/dir", "pkg/core.py", "out"))
    os.symlink("core.py", root / "pkg" / "link.py")
    (root / "pkg" / "dir").mkdir()
    os.symlink(str(outside), root / "out")
    with pytest.raises(PermissionError, match="OPEN_SWE_MUTATION_PATH_FORBIDDEN"):
        backend.write("pkg/link.py", "VALUE = 2\n")
    with pytest.raises(PermissionError, match="OPEN_SWE_MUTATION_PATH_FORBIDDEN"):
        backend.write("pkg/dir", "VALUE = 2\n")
    with pytest.raises(PermissionError, match="OPEN_SWE_MUTATION_PATH_FORBIDDEN"):
        backend.edit("out/core.py", "VALUE = 1", "VALUE = 2")
    with pytest.raises(cli.RuntimeErrorBounded, match="OPEN_SWE_PATH_INVALID"):
        backend.write("pkg/../../escape.py", "VALUE = 2\n")
    assert (root / "pkg" / "core.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert (outside / "core.py").read_text(encoding="utf-8") == "VALUE = 1\n"


# --- review hardening (#130 review of PR #131) -----------------------------------


def _run_real_backend(
    request: dict[str, Any],
    actions: Callable[[cli.ScopedRepairBackend, Path], None],
) -> dict[str, Any]:
    """Run the bound worker with a repair graph that drives the REAL scoped backend."""

    def repair_factory(
        _model, workspace, _runtime, allowed_paths, _key, checkpointer=None, effect_journal=None
    ):
        assert effect_journal is not None
        backend = cli.ScopedRepairBackend(
            _Delegate(), Path(workspace), tuple(allowed_paths), effect_journal
        )

        def effect() -> None:
            effect_journal.bind_turn("turn-1", "call-1")
            actions(backend, Path(workspace))

        return _Graph(
            cli.REPAIR_TOOLS, _record("record_worker_result", {"summary": "repaired"}), effect
        )

    return cli._worker_run(
        request,
        runtime_loader=lambda: {"human_message": lambda content: content},
        model_factory=lambda *_args: SimpleNamespace(_conversation_id=None),
        diagnosis_factory=lambda *_a, **_k: pytest.fail("bound packet must skip diagnosis"),
        repair_factory=repair_factory,
    )


def _executable(path: str) -> Callable[[Path], None]:
    return lambda workspace: os.chmod(workspace / path, 0o755)


def _crlf_executable(workspace: Path) -> None:
    (workspace / "pkg").mkdir(parents=True, exist_ok=True)
    (workspace / "pkg" / "tool.py").write_bytes(b"x = 1\r\ny = 2\r\nz = 3\r\n")
    os.chmod(workspace / "pkg" / "tool.py", 0o755)


def test_real_backend_edit_preserves_crlf_bytes_and_executable_mode(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={},
        test_paths=[],
        production_paths=["pkg/tool.py"],
        refs=[f"task_card:{CARD_REF}@" + "a" * 16, "source_file:pkg/tool.py@" + "b" * 16],
        prepare=_crlf_executable,
    )
    outcome: dict[str, Any] = {}

    def actions(backend: cli.ScopedRepairBackend, _workspace: Path) -> None:
        outcome["edit"] = backend.edit("pkg/tool.py", "y = 2", "y = 20")

    result = _run_real_backend(request, actions)

    assert getattr(outcome["edit"], "error", None) is None, outcome
    assert result["status"] == "COMPLETED", result
    assert result["core_binding_status"] == "BOUND"
    workspace = Path(request["workspace_path"])
    target = workspace / "pkg" / "tool.py"
    assert target.read_bytes() == b"x = 1\r\ny = 20\r\nz = 3\r\n"
    assert target.stat().st_mode & 0o777 == 0o755
    assert _git(workspace, "diff", "--numstat") == "1\t1\tpkg/tool.py"
    (entry,) = result["core_change_set_projection"]["change_manifest"]["entries"]
    assert entry["change_type"] == "MODIFY"
    assert entry["before_mode"] == entry["after_mode"] == "100755"


def test_real_backend_multiline_crlf_edit_preserves_untouched_bytes(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={},
        test_paths=[],
        production_paths=["pkg/tool.py"],
        refs=[f"task_card:{CARD_REF}@" + "a" * 16, "source_file:pkg/tool.py@" + "b" * 16],
        prepare=_crlf_executable,
    )

    def actions(backend: cli.ScopedRepairBackend, _workspace: Path) -> None:
        result = backend.edit("pkg/tool.py", "x = 1\ny = 2", "x = 10\ny = 20")
        assert getattr(result, "error", None) is None, result

    result = _run_real_backend(request, actions)
    assert result["status"] == "COMPLETED", result
    target = Path(request["workspace_path"]) / "pkg" / "tool.py"
    assert target.read_bytes() == b"x = 10\r\ny = 20\r\nz = 3\r\n"


def test_real_backend_write_preserves_executable_mode(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"bin/run.py": "print(1)\n"},
        test_paths=[],
        production_paths=["bin/run.py"],
        prepare=_executable("bin/run.py"),
    )

    def actions(backend: cli.ScopedRepairBackend, _workspace: Path) -> None:
        backend.write("bin/run.py", "print(2)\n")

    result = _run_real_backend(request, actions)
    assert result["status"] == "COMPLETED", result
    workspace = Path(request["workspace_path"])
    assert (workspace / "bin" / "run.py").stat().st_mode & 0o777 == 0o755
    assert "mode change" not in _git(workspace, "diff", "--summary")
    (entry,) = result["core_change_set_projection"]["change_manifest"]["entries"]
    assert entry["before_mode"] == entry["after_mode"] == "100755"


def test_journal_edit_of_executable_keeps_mode(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"bin/run.py": "print(1)\n"},
        test_paths=[],
        production_paths=["bin/run.py"],
        prepare=_executable("bin/run.py"),
    )
    result, _calls = _run(request, {"bin/run.py": "print(2)\n"})
    assert result["status"] == "COMPLETED", result
    workspace = Path(request["workspace_path"])
    assert (workspace / "bin" / "run.py").stat().st_mode & 0o777 == 0o755


def test_projection_rejects_mode_change(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=[],
        production_paths=["pkg/core.py"],
    )
    result, _calls = _run(
        request, {"pkg/core.py": "VALUE = 2\n"}, extra_effect=_executable("pkg/core.py")
    )
    assert result["status"] != "COMPLETED"
    assert result["error_code"] == "OPEN_SWE_CORE_CHANGESET_MODE_CHANGE_FORBIDDEN"


@pytest.mark.parametrize(
    "path", [".git/hooks/post-commit", "sub/.GIT/config", ".Git", "a/.git/b/c.py"]
)
def test_git_metadata_target_is_rejected_at_admission(tmp_path, path):
    request = _bound_request(
        tmp_path,
        base_files={},
        test_paths=[path],
        production_paths=[],
        refs=[f"task_card:{CARD_REF}@" + "a" * 16, f"source_absence:{path}@" + "b" * 16],
    )
    admission = _admission(request)
    assert admission.decision == cli.REJECT
    assert admission.reason_code == "path_git_metadata"


@pytest.mark.parametrize("path", [".git/hooks/post-commit", "sub/.GIT/config", ".git"])
def test_backend_refuses_git_metadata_paths(tmp_path, path):
    root = tmp_path / "root"
    (root / ".git" / "hooks").mkdir(parents=True)
    backend = cli.ScopedRepairBackend(_Delegate(), root, (path,))
    with pytest.raises(PermissionError, match="OPEN_SWE_MUTATION_PATH_FORBIDDEN"):
        backend.write(path, "#!/bin/sh\necho owned\n")
    assert not (root / ".git" / "hooks" / "post-commit").exists()


@pytest.mark.parametrize(
    "path", [".gitattributes", "pkg/.gitignore", ".GitModules", "a/b/.GITATTRIBUTES"]
)
def test_git_control_file_target_is_rejected_at_admission(tmp_path, path):
    request = _bound_request(
        tmp_path,
        base_files={},
        test_paths=[path],
        production_paths=[],
        refs=[f"task_card:{CARD_REF}@" + "a" * 16, f"source_absence:{path}@" + "b" * 16],
    )
    admission = _admission(request)
    assert admission.decision == cli.REJECT
    assert admission.reason_code == "path_git_control"


def test_task_card_cannot_authorize_editing_itself(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=[],
        production_paths=["pkg/core.py", CARD_REF],
        refs=[
            f"task_card:{CARD_REF}@" + "a" * 16,
            "source_file:pkg/core.py@" + "b" * 16,
            f"source_file:{CARD_REF}@" + "b" * 16,
        ],
    )
    admission = _admission(request)
    assert admission.decision == cli.REJECT
    assert admission.reason_code == "path_authority"


def test_allowed_path_overlapping_forbidden_paths_is_rejected(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=[],
        production_paths=["pkg/core.py"],
    )
    envelope = json.loads(Path(request["artifact_path"]).read_text(encoding="utf-8"))
    envelope["scope_signal"]["forbidden_paths"] = ["pkg"]
    raw = cli._canonical_json(envelope)
    Path(request["artifact_path"]).write_text(raw, encoding="utf-8")
    request["prompt"] = "\n".join(
        f"envelope_sha256={cli._sha256(raw)}" if line.startswith("envelope_sha256=") else line
        for line in request["prompt"].splitlines()
    )
    admission = _admission(request)
    assert admission.decision == cli.REJECT
    assert admission.reason_code == "path_authority"
