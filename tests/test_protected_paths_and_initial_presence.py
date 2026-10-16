"""#132 items 1, 3 and 4: protected-path wiring, run-created files, fail-closed normalisation."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import pytest
from test_bound_continuation import (
    _bound_initial_request,
    _Calls,
    _capture_candidate,
    _continue_request,
    _factories,
    _run_continue,
)
from test_bound_edit_existing import (
    CARD_REF,
    _admission,
    _bound_request,
    _Delegate,
    _git,
    _run_real_backend,
)

from nexus_open_swe_runtime import cli
from nexus_open_swe_runtime.recovery import DurableEffectJournal

CAFE_NFC = unicodedata.normalize("NFC", "docs/caf\u00e9.md")
CAFE_NFD = unicodedata.normalize("NFD", "docs/caf\u00e9.md")

# --- item 1: protected paths reach the identity and the session -------------------


def test_worker_identity_carries_protected_paths_to_the_real_backend(tmp_path):
    """Admission never lets a listed path overlap the card, so widen the backend scope.

    The backend is built exactly as the worker builds it, from the journal identity
    that ``_worker_run`` assembled.  If ``mutation_protected_paths`` is not passed to
    that identity, the widened card path is a tracked regular file and the write goes
    through.
    """
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=[],
        production_paths=["pkg/core.py"],
    )
    workspace = Path(request["workspace_path"])
    card_before = (workspace / CARD_REF).read_text(encoding="utf-8")
    outcomes: list[str] = []

    def actions(_backend: cli.ScopedRepairBackend, root: Path) -> None:
        journal = _backend._effect_journal
        assert journal is not None
        widened = cli.ScopedRepairBackend(
            _Delegate(), root, ("pkg/core.py", CARD_REF), journal
        )
        try:
            widened.write(CARD_REF, "- status: `HIJACKED`\n")
            outcomes.append("WRITTEN")
        except PermissionError as exc:
            outcomes.append(str(exc))

    _run_real_backend(request, actions)
    assert outcomes == ["OPEN_SWE_MUTATION_PATH_FORBIDDEN"]
    assert (workspace / CARD_REF).read_text(encoding="utf-8") == card_before


def test_session_protected_paths_reach_the_bound_continuation_gate(tmp_path):
    """The continuation re-applies the protected paths persisted on the session."""
    initial = _bound_initial_request(tmp_path)
    calls = _Calls()
    result = cli._worker_run(initial, **_factories(calls, "VALUE = 1\n"))
    assert result["status"] == "COMPLETED", result
    parent = _capture_candidate(Path(initial["workspace_path"]))
    session_id = result["session_id"]
    session_path = cli._session_path(initial, session_id)
    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert set(session["core_protected_paths"]) == {CARD_REF}
    # Widen the persisted scope (a forged / drifted session) so that only the stored
    # protected paths can stop a continuation from editing the card itself.
    session["allowed_paths"] = [CARD_REF]
    session_path.write_text(json.dumps(session), encoding="utf-8")
    request = _continue_request(
        tmp_path, initial, session_id, parent, {"allowed_mutation_paths": [CARD_REF]}
    )
    outcome, calls = _run_continue(request)
    assert outcome["error_code"] == "OPEN_SWE_BOUND_CONTINUATION_REJECTED_PATH_AUTHORITY"
    assert calls.as_tuple() == (0, 0, 0)


# --- item 3: files created earlier in the run are not pre-existing ----------------

NEW = "pkg/new.py"


def _new_file_backend(tmp_path: Path):
    request = _bound_request(
        tmp_path,
        base_files={"seed.py": "SEED = 1\n"},
        test_paths=[],
        production_paths=[NEW],
        refs=[f"task_card:{CARD_REF}@" + "a" * 16, f"source_absence:{NEW}@" + "b" * 16],
    )
    workspace = Path(request["workspace_path"]).resolve()
    outcomes: dict[str, Any] = {}

    def actions(backend: cli.ScopedRepairBackend, root: Path) -> None:
        journal = backend._effect_journal
        assert journal is not None
        backend.write(NEW, "VALUE = 1\n")
        # The bound backend is rebuilt (resume / a second graph build) while the
        # untracked file the first backend created still exists.
        rebuilt = cli.ScopedRepairBackend(_Delegate(), root, (NEW,), journal)
        try:
            rebuilt.edit(NEW, "VALUE = 1", "VALUE = 2")
            outcomes["rebuilt"] = "EDITED"
        except PermissionError as exc:
            outcomes["rebuilt"] = str(exc)
        # A file that exists BEFORE the run and is untracked is still rejected.
        outcomes["file"] = (root / NEW).read_text(encoding="utf-8")

    return request, workspace, actions, outcomes


def test_rebuilt_bound_backend_keeps_run_created_untracked_file_editable(tmp_path):
    request, _workspace, actions, outcomes = _new_file_backend(tmp_path)
    _run_real_backend(request, actions)
    assert outcomes == {"rebuilt": "EDITED", "file": "VALUE = 2\n"}


def test_preexisting_untracked_file_is_still_source_untracked(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"seed.py": "SEED = 1\n"},
        test_paths=[],
        production_paths=[NEW],
        refs=[f"task_card:{CARD_REF}@" + "a" * 16, f"source_absence:{NEW}@" + "b" * 16],
    )
    workspace = Path(request["workspace_path"]).resolve()
    (workspace / "pkg").mkdir()
    (workspace / NEW).write_text("SECRET = 1\n", encoding="utf-8")
    base = _git(workspace, "rev-parse", "HEAD")
    journal = DurableEffectJournal(
        tmp_path / "state2",
        _identity_for(workspace, base),
    )
    journal.bind_turn("turn-1", "call-1")
    backend = cli.ScopedRepairBackend(_Delegate(), workspace, (NEW,), journal)
    with pytest.raises(PermissionError, match="OPEN_SWE_MUTATION_PATH_FORBIDDEN"):
        backend.edit(NEW, "SECRET = 1", "SECRET = 2")


def _identity_for(workspace: Path, base: str):
    from test_mutation_gate_parity import _identity

    return _identity(workspace, NEW, base, bound=True)


# --- item 4: protected path normalisation fails closed ----------------------------


@pytest.mark.parametrize(
    "bad", ["../x", "a/../../x", "a\\b", "", "   ", "a\x00b", "sec/**", "a?b", "x[1]", None, 7]
)
def test_invalid_protected_entry_fails_closed(bad):
    with pytest.raises(cli.RuntimeErrorBounded, match="OPEN_SWE_PATH_INVALID"):
        cli._protected_mutation_paths(["tasks/card.md", bad])


def test_protected_entries_are_normalised_to_nfc():
    out = cli._protected_mutation_paths([CAFE_NFD, CAFE_NFC])
    assert out == (CAFE_NFC.lower(),) * 2
    assert all(unicodedata.is_normalized("NFC", entry) for entry in out)


def test_valid_protected_entries_are_still_normalised():
    assert cli._protected_mutation_paths(["/Tasks/Card.md", "sec/"]) == (
        "tasks/card.md",
        "sec",
    )


def test_admission_rejects_an_invalid_forbidden_path_entry(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=[],
        production_paths=["pkg/core.py"],
    )
    envelope = json.loads(Path(request["artifact_path"]).read_text(encoding="utf-8"))
    envelope["scope_signal"]["forbidden_paths"] = ["../secret"]
    raw = cli._canonical_json(envelope)
    Path(request["artifact_path"]).write_text(raw, encoding="utf-8")
    request["prompt"] = "\n".join(
        f"envelope_sha256={cli._sha256(raw)}" if line.startswith("envelope_sha256=") else line
        for line in request["prompt"].splitlines()
    )
    admission = _admission(request)
    assert admission.decision == cli.REJECT
    assert admission.reason_code == "protected_path_invalid"


CAFE_NFC = unicodedata.normalize("NFC", "docs/café.md")
CAFE_NFD = unicodedata.normalize("NFD", "docs/café.md")


@pytest.mark.parametrize(
    ("protected", "written"), [(CAFE_NFC, CAFE_NFD), (CAFE_NFD, CAFE_NFC)], ids=["nfc>nfd", "nfd>nfc"]
)
def test_protected_path_blocks_the_other_unicode_normal_form(tmp_path, protected, written):
    assert protected != written
    violation = cli._mutation_target_violation(
        tmp_path,
        written,
        base_commit=None,
        protected=cli._protected_mutation_paths([protected]),
    )
    assert violation == "path_authority"


def test_admission_rejects_a_glob_shaped_forbidden_path(tmp_path):
    request = _bound_request(
        tmp_path,
        base_files={"pkg/core.py": "VALUE = 1\n"},
        test_paths=[],
        production_paths=["pkg/core.py"],
    )
    envelope = json.loads(Path(request["artifact_path"]).read_text(encoding="utf-8"))
    envelope["scope_signal"]["forbidden_paths"] = ["sec/**"]
    raw = cli._canonical_json(envelope)
    Path(request["artifact_path"]).write_text(raw, encoding="utf-8")
    request["prompt"] = "\n".join(
        f"envelope_sha256={cli._sha256(raw)}" if line.startswith("envelope_sha256=") else line
        for line in request["prompt"].splitlines()
    )
    admission = _admission(request)
    assert (admission.decision, admission.reason_code) == (cli.REJECT, "protected_path_invalid")


def test_non_string_scope_entry_is_not_silently_dropped():
    envelope = {"scope_signal": {"forbidden_paths": ["tasks/task-card.md", None]}}
    with pytest.raises(cli.RuntimeErrorBounded, match="OPEN_SWE_PATH_INVALID"):
        cli._protected_paths_from_envelope(envelope)


@pytest.mark.parametrize("stored", [["tasks/task-card.md", None], "tasks/task-card.md", ["sec/**"]])
def test_invalid_stored_session_protected_paths_reject_the_continuation(tmp_path, stored):
    initial = _bound_initial_request(tmp_path)
    result = cli._worker_run(initial, **_factories(_Calls(), "VALUE = 1\n"))
    assert result["status"] == "COMPLETED", result
    parent = _capture_candidate(Path(initial["workspace_path"]))
    session_id = result["session_id"]
    session_path = cli._session_path(initial, session_id)
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["core_protected_paths"] = stored
    session_path.write_text(json.dumps(session), encoding="utf-8")
    request = _continue_request(tmp_path, initial, session_id, parent)
    outcome, calls = _run_continue(request)
    assert outcome["error_code"] == (
        "OPEN_SWE_BOUND_CONTINUATION_REJECTED_PROTECTED_PATHS_INVALID"
    )
    assert calls.as_tuple() == (0, 0, 0)


# --- run-created exemption is bound to the file the run actually wrote ------------


def _journal_backend(tmp_path: Path):
    request = _bound_request(
        tmp_path,
        base_files={"seed.py": "SEED = 1\n"},
        test_paths=[],
        production_paths=[NEW],
        refs=[f"task_card:{CARD_REF}@" + "a" * 16, f"source_absence:{NEW}@" + "b" * 16],
    )
    workspace = Path(request["workspace_path"]).resolve()
    (workspace / "pkg").mkdir()
    base = _git(workspace, "rev-parse", "HEAD")
    journal = DurableEffectJournal(tmp_path / "state3", _identity_for(workspace, base))
    journal.bind_turn("turn-1", "call-1")
    return workspace, journal


def _rebuilt_edit_outcome(workspace: Path, journal: DurableEffectJournal, old: str) -> str:
    backend = cli.ScopedRepairBackend(_Delegate(), workspace, (NEW,), journal)
    try:
        backend.edit(NEW, old, "CHANGED = 1")
    except PermissionError as exc:
        return str(exc)
    return "EDITED"


def test_intent_only_record_gives_no_exemption_for_a_file_someone_else_created(tmp_path):
    workspace, journal = _journal_backend(tmp_path)
    journal.intent(
        turn_id="turn-1",
        tool_call_id="call-1",
        tool_name="write_file",
        arguments={"file_path": NEW, "content": "VALUE = 1\n"},
        path=workspace / NEW,
        preimage=None,
        postimage="VALUE = 1\n",
    )
    (workspace / NEW).write_text("VALUE = 1\n", encoding="utf-8")  # another process
    assert _rebuilt_edit_outcome(workspace, journal, "VALUE = 1") == (
        "OPEN_SWE_MUTATION_PATH_FORBIDDEN"
    )


def test_run_created_file_replaced_by_other_content_is_no_longer_exempt(tmp_path):
    workspace, journal = _journal_backend(tmp_path)
    backend = cli.ScopedRepairBackend(_Delegate(), workspace, (NEW,), journal)
    backend.write(NEW, "VALUE = 1\n")
    assert _rebuilt_edit_outcome(workspace, journal, "VALUE = 1") == "EDITED"
    (workspace / NEW).write_text("SWAPPED = 1\n", encoding="utf-8")
    assert _rebuilt_edit_outcome(workspace, journal, "SWAPPED = 1") == (
        "OPEN_SWE_MUTATION_PATH_FORBIDDEN"
    )


def test_forged_effect_record_with_missing_fields_gives_no_exemption(tmp_path):
    workspace, journal = _journal_backend(tmp_path)
    (workspace / NEW).write_text("VALUE = 1\n", encoding="utf-8")
    effect_id = "effect_" + "f" * 64
    forged = {
        "status": "RESULT",
        "effect_id": effect_id,
        "operation_id": journal.identity.operation_id,
        "path": str(workspace / NEW),
        "preimage_sha256": "absent",
    }
    (journal.root / f"{effect_id}.json").write_text(json.dumps(forged), encoding="utf-8")
    assert _rebuilt_edit_outcome(workspace, journal, "VALUE = 1") == (
        "OPEN_SWE_MUTATION_PATH_FORBIDDEN"
    )
