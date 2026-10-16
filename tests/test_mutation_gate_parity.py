"""Sibling-gate parity: every write entry point enforces the same path gates (#130).

Entry points: v2 admission, ScopedRepairBackend write / edit / async write, the
composite write (write_and_record_worker_result), bound repair continuation (#127),
and the unbound legacy/FALLBACK backend.  All of them call the single shared
predicate ``cli._mutation_target_violation``; this matrix proves each gate
rejects at each entry point on a real Git workspace.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pytest
from test_bound_edit_existing import CARD_REF, _admission, _bound_request, _git

from nexus_open_swe_runtime import cli
from nexus_open_swe_runtime.recovery import DurableEffectJournal, RecoveryIdentity


def _noop(_workspace: Path) -> None:
    return None


def _symlink_target(workspace: Path) -> None:
    (workspace / "pkg").mkdir(exist_ok=True)
    (workspace / "pkg" / "real.py").write_text("VALUE = 1\n", encoding="utf-8")
    os.symlink("real.py", workspace / "pkg" / "target.py")


def _directory_target(workspace: Path) -> None:
    (workspace / "pkg" / "target.py").mkdir(parents=True)
    (workspace / "pkg" / "target.py" / "inner.py").write_text("X = 1\n", encoding="utf-8")


def _escaping_parent(workspace: Path) -> None:
    outside = workspace.parent / "outside"
    outside.mkdir(exist_ok=True)
    (outside / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    os.symlink(str(outside), workspace / "pkg")


def _untracked_ignored(workspace: Path) -> None:
    (workspace / ".git" / "info" / "exclude").write_text("pkg/target.py\n", encoding="utf-8")
    (workspace / "pkg").mkdir(exist_ok=True)
    (workspace / "pkg" / "target.py").write_text("SECRET = 1\n", encoding="utf-8")


def _git_control(workspace: Path) -> None:
    (workspace / ".gitignore").write_text("*.log\n", encoding="utf-8")


# gate -> (before base commit, after base commit, target path)
GATES: dict[str, tuple[Callable[[Path], None], Callable[[Path], None], str]] = {
    "symlink": (_symlink_target, _noop, "pkg/target.py"),
    "regular_file": (_directory_target, _noop, "pkg/target.py"),
    "tracked_at_base": (_noop, _untracked_ignored, "pkg/target.py"),
    "git_segment": (_noop, _noop, ".git/hooks/post-commit"),
    "git_control_file": (_git_control, _noop, ".gitignore"),
    "authority_overlap": (_noop, _noop, CARD_REF),
    "containment": (_escaping_parent, _noop, "pkg/target.py"),
}


def _workspace_for(tmp_path: Path, gate: str) -> tuple[dict[str, Any], Path, str, str]:
    before, after, target = GATES[gate]
    request = _bound_request(
        tmp_path,
        base_files={"seed.py": "SEED = 1\n"},
        test_paths=[],
        production_paths=[target],
        refs=[
            f"task_card:{CARD_REF}@" + "a" * 16,
            f"source_file:{target}@" + "b" * 16,
            f"source_absence:{target}@" + "b" * 16,
        ],
        prepare=before,
    )
    workspace = Path(request["workspace_path"]).resolve()
    after(workspace)
    base = _git(workspace, "rev-parse", "HEAD")
    return request, workspace, target, base


def _identity(workspace: Path, target: str, base: str, *, bound: bool) -> RecoveryIdentity:
    core: dict[str, Any] = {}
    if bound:
        core = {
            "core_binding_hash": "sha256:" + "1" * 64,
            "acceptance_contract_hash": "sha256:" + "2" * 64,
            "core_repository": "James3014/Nexus-new",
            "core_source_revision": f"git-commit:{base}",
            "core_source_tree": "git-tree:" + _git(workspace, "rev-parse", "HEAD^{tree}"),
            "core_attempt_id": "attempt-1",
            "core_required_verifier_ids": ("pytest",),
            "core_deletion_policy": "FORBID",
        }
    return RecoveryIdentity(
        operation_id="a" * 64,
        execution_material_sha256="b" * 64,
        workspace=str(workspace),
        task_id="task-1",
        unit_id="u1",
        session_id="session-1",
        allowed_paths=(target,),
        provider_id="google_genai",
        model_id="gemini-test",
        worker_identity_sha256="c" * 64,
        transport_config_sha256="d" * 64,
        runtime_identity_sha256="e" * 64,
        composite_admitted=True,
        mutation_protected_paths=cli._protected_mutation_paths([CARD_REF]),
        **core,
    )


class _Delegate:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"delegate.{name} must not be reached")


def _backend(tmp_path: Path, workspace: Path, target: str, base: str, *, bound: bool):
    journal = DurableEffectJournal(
        tmp_path / "state", _identity(workspace, target, base, bound=bound)
    )
    journal.bind_turn("turn-1", "call-1")
    return cli.ScopedRepairBackend(_Delegate(), workspace, (target,), journal)


def _rejected_by(call: Callable[[], Any]) -> bool:
    try:
        result = call()
    except (PermissionError, cli.RuntimeErrorBounded):
        return True
    return getattr(result, "error", None) is not None and "not found" not in str(result.error)


_LAST_ADMISSION_REASON: list[str | None] = [None]
_LAST_CONTINUATION_CODE: list[str | None] = [None]


def _entry_admission(tmp_path, request, workspace, target, base):
    admission = _admission(request)
    _LAST_ADMISSION_REASON[0] = admission.reason_code
    return admission.decision == cli.REJECT


def _entry_backend_write(tmp_path, request, workspace, target, base):
    backend = _backend(tmp_path, workspace, target, base, bound=True)
    return _rejected_by(lambda: backend.write(target, "VALUE = 2\n"))


def _entry_backend_async_write(tmp_path, request, workspace, target, base):
    backend = _backend(tmp_path, workspace, target, base, bound=True)
    return _rejected_by(lambda: asyncio.run(backend.awrite(target, "VALUE = 2\n")))


def _entry_backend_edit(tmp_path, request, workspace, target, base):
    backend = _backend(tmp_path, workspace, target, base, bound=True)
    return _rejected_by(lambda: backend.edit(target, "VALUE = 1", "VALUE = 2"))


def _entry_composite(tmp_path, request, workspace, target, base):
    backend = _backend(tmp_path, workspace, target, base, bound=True)
    return _rejected_by(
        lambda: backend.write_and_record_worker_result(
            target, "VALUE = 2\n", {"summary": "done"}
        )
    )


def _entry_legacy_fallback(tmp_path, request, workspace, target, base):
    backend = _backend(tmp_path, workspace, target, base, bound=False)
    return _rejected_by(lambda: backend.write(target, "VALUE = 2\n"))


_GATE_CODES = (
    "PATH_ESCAPE", "PATH_GIT_METADATA", "PATH_GIT_CONTROL", "PATH_AUTHORITY", "PATH_SYMLINK",
    "PATH_PARENT", "PATH_NOT_REGULAR", "SOURCE_UNTRACKED", "WORKSPACE_SYMLINK",
)


def _continuation_code(workspace: Path, target: str, base: str) -> str:
    session_id = "session-1"
    delta = {
        "schema": "unit_repair_delta.v2",
        "task_id": "task-1",
        "unit_id": "u1",
        "session_id": session_id,
        "workspace_path": str(workspace),
        "allowed_mutation_paths": [target],
        "parent_candidate_commit": base,
    }
    raw = json.dumps(delta, sort_keys=True).encode("utf-8")
    deadline = datetime.now(timezone.utc) + timedelta(hours=1)
    core_identity = {
        "core_binding_hash": "sha256:" + "1" * 64,
        "acceptance_contract_hash": "sha256:" + "2" * 64,
        "core_repository": "James3014/Nexus-new",
        "core_source_revision": f"git-commit:{base}",
        "core_source_tree": "git-tree:" + _git(workspace, "rev-parse", "HEAD^{tree}"),
    }
    try:
        cli._bound_continuation_admission(
            {
                "core_binding_hash": core_identity["core_binding_hash"],
                "acceptance_contract_hash": core_identity["acceptance_contract_hash"],
            },
            workspace,
            f"repair_sha256={cli._sha256(raw)}\nparent_candidate_commit={base}",
            raw,
            session_id=session_id,
            task_id="task-1",
            unit_id="u1",
            allowed_paths=(target,),
            core_identity=core_identity,
            valid_until=deadline.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            protected_paths=[CARD_REF],
        )
    except cli.RuntimeErrorBounded as exc:
        return str(exc)
    return ""


def _entry_continuation(tmp_path, request, workspace, target, base):
    code = _continuation_code(workspace, target, base)
    _LAST_CONTINUATION_CODE[0] = code
    return code.removeprefix("OPEN_SWE_BOUND_CONTINUATION_REJECTED_") in _GATE_CODES


ENTRY_POINTS = {
    "v2_admission": _entry_admission,
    "backend_write": _entry_backend_write,
    "backend_async_write": _entry_backend_async_write,
    "backend_edit": _entry_backend_edit,
    "composite_write": _entry_composite,
    "bound_continuation": _entry_continuation,
    "legacy_fallback_backend": _entry_legacy_fallback,
}

# Tracked-at-base needs a bound base commit; an unbound legacy run has none
# (its workspace need not be a Git checkout), so that one cell is not applicable.
_NOT_APPLICABLE = {("tracked_at_base", "legacy_fallback_backend")}


# The reason code the SHARED predicate must return for each gate cell (#132 item 2).
# Rejection alone is not enough: a cell can be rejected by a different, earlier gate.
EXPECTED_REASON = {
    "symlink": "path_symlink",
    "regular_file": "path_not_regular",
    "tracked_at_base": "source_untracked",
    "git_segment": "path_git_metadata",
    "git_control_file": "path_git_control",
    "authority_overlap": "path_authority",
    "containment": "path_parent",
}
# Entry points whose own pre-checks fire before the shared predicate for a symlink.
_PRECHECK_REASON = {("symlink", "bound_continuation"): "workspace_symlink"}


def _simulate_symlink_swapped_after_the_symlink_gate(monkeypatch) -> None:
    """Containment case that the symlink gate does not catch.

    A symlink component planted AFTER the symlink walk (TOCTOU) is invisible to
    ``_workspace_path_has_symlink``; only the resolve-based containment check in
    ``_mutation_target_violation`` can still stop the write.  The gates themselves
    are unchanged; the symlink walk is told the tree is clean.
    """
    monkeypatch.setattr(cli, "_workspace_path_has_symlink", lambda _workspace, _relative: False)


@pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
@pytest.mark.parametrize("gate", sorted(GATES))
def test_each_gate_rejects_at_every_entry_point(tmp_path, monkeypatch, gate, entry):
    if (gate, entry) in _NOT_APPLICABLE:
        pytest.skip("tracked-at-base is a bound-only gate")
    request, workspace, target, base = _workspace_for(tmp_path, gate)
    if gate == "containment":
        _simulate_symlink_swapped_after_the_symlink_gate(monkeypatch)
    codes: list[str | None] = []
    real = cli._mutation_target_violation

    def spy(*args: Any, **kwargs: Any) -> str | None:
        code = real(*args, **kwargs)
        codes.append(code)
        return code

    monkeypatch.setattr(cli, "_mutation_target_violation", spy)
    assert ENTRY_POINTS[entry](tmp_path, request, workspace, target, base), (gate, entry)
    expected = _PRECHECK_REASON.get((gate, entry), EXPECTED_REASON[gate])
    if entry == "bound_continuation":
        assert _LAST_CONTINUATION_CODE[0] == (
            "OPEN_SWE_BOUND_CONTINUATION_REJECTED_" + expected.upper()
        ), (gate, entry, _LAST_CONTINUATION_CODE[0])
    elif entry == "v2_admission":
        assert _LAST_ADMISSION_REASON[0] == expected, (gate, entry, _LAST_ADMISSION_REASON[0])
    else:
        assert {code for code in codes if code} == {expected}, (gate, entry, codes)


@pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
def test_clean_existing_tracked_target_is_accepted_at_every_entry_point(tmp_path, entry):
    """Control: the gates do not reject a legitimate listed, tracked, regular file."""
    request = _bound_request(
        tmp_path,
        base_files={"pkg/target.py": "VALUE = 1\n"},
        test_paths=[],
        production_paths=["pkg/target.py"],
    )
    workspace = Path(request["workspace_path"]).resolve()
    base = _git(workspace, "rev-parse", "HEAD")
    if entry == "v2_admission":
        assert _admission(request).decision == cli.ADMIT
    elif entry == "bound_continuation":
        # Past the shared gate the continuation stops later (e.g. no prior BOUND
        # terminal for this workspace), never on a path gate.
        code = _continuation_code(workspace, "pkg/target.py", base)
        assert code.startswith("OPEN_SWE_BOUND_CONTINUATION_REJECTED_"), code
        assert code.removeprefix("OPEN_SWE_BOUND_CONTINUATION_REJECTED_") not in _GATE_CODES
    else:
        assert not ENTRY_POINTS[entry](tmp_path, request, workspace, "pkg/target.py", base)
        assert (workspace / "pkg" / "target.py").read_text(encoding="utf-8") == "VALUE = 2\n"
