from __future__ import annotations

import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from nexus_open_swe_runtime import cli, recovery
from nexus_open_swe_runtime.recovery import (
    DurableEffectJournal,
    DurableOperationJournal,
    RecoveryIdentity,
    reconcile_bound_turn,
)


def _identity(tmp_path: Path) -> RecoveryIdentity:
    return RecoveryIdentity(
        operation_id="a" * 64,
        execution_material_sha256="b" * 64,
        workspace=str(tmp_path / "workspace"),
        task_id="task-1",
        unit_id="unit-1",
        session_id="session-1",
        allowed_paths=("a.py",),
        provider_id="opencli_chatgpt",
        model_id="balanced",
        worker_identity_sha256="c" * 64,
        transport_config_sha256="d" * 64,
        runtime_identity_sha256="e" * 64,
        checkpoint_namespace="open-swe-repair-v1",
    )


def _contending_fence(state_root: str, workspace: str, marker: str) -> None:
    identity = RecoveryIdentity(
        operation_id="a" * 64,
        execution_material_sha256="b" * 64,
        workspace=workspace,
        task_id="task-1",
        unit_id="unit-1",
        session_id="session-1",
        allowed_paths=("a.py",),
        provider_id="opencli_chatgpt",
        model_id="balanced",
        worker_identity_sha256="c" * 64,
        transport_config_sha256="d" * 64,
        runtime_identity_sha256="e" * 64,
    )
    journal = DurableOperationJournal(state_root, identity)
    with journal.fence():
        with Path(marker).open("a", encoding="utf-8") as stream:
            stream.write("entered\n")
            stream.flush()
        time.sleep(0.15)
        with Path(marker).open("a", encoding="utf-8") as stream:
            stream.write("leaving\n")
            stream.flush()


def test_operation_journal_is_atomic_owner_only_and_round_trips(tmp_path: Path):
    journal = DurableOperationJournal(tmp_path / "state", _identity(tmp_path))
    journal.prepare()
    journal.ask_dispatching(turn_id="turn_1", prompt="payload", ordinal=0)
    journal.conversation_bound("conversation-1")
    journal.response_recovered("turn_1", '{"type":"tool_call"}')

    restored = DurableOperationJournal.open(tmp_path / "state", _identity(tmp_path))
    state = restored.read()
    assert state["status"] == "RESPONSE_RECOVERED"
    assert state["conversation_id"] == "conversation-1"
    assert state["turn_id"] == "turn_1"
    assert state["prompt_sha256"]
    assert state["lease_epoch"] >= 1
    assert journal.path.stat().st_mode & 0o077 == 0


def test_operation_journal_persists_protocol_repair_origin_and_response(tmp_path: Path):
    journal = DurableOperationJournal(tmp_path / "state", _identity(tmp_path))
    journal.prepare()
    journal.conversation_bound("conversation-original")
    origin = '{"type":"tool_call","name":"write_file","arguments":{}}'
    journal.protocol_repair_started(
        origin=origin,
        origin_sha256=cli._sha256(origin),
        turn_id="turn_repair_1",
    )
    journal.ask_dispatching(turn_id="turn_repair_1", prompt="repair", ordinal=1)
    journal.conversation_bound("conversation-repair")
    journal.response_recovered("turn_repair_1", '{"type":"final","content":"ok"}')
    journal.protocol_repair_recovered('{"type":"final","content":"ok"}')

    state = journal.read()
    assert state["protocol_repair_origin"] == origin
    assert state["protocol_repair_origin_sha256"] == cli._sha256(origin)
    assert state["protocol_repair_turn_id"] == "turn_repair_1"
    assert state["protocol_repair_original_conversation_id"] == "conversation-original"
    assert state["protocol_repair_status"] == "RECOVERED"
    assert state["protocol_repair_response_sha256"] == cli._sha256(
        '{"type":"final","content":"ok"}'
    )


def test_effect_journal_accepts_exact_already_applied_write(tmp_path: Path):
    target = tmp_path / "a.py"
    journal = DurableEffectJournal(tmp_path / "state", _identity(tmp_path))
    effect = journal.intent(
        turn_id="turn_1",
        tool_call_id="call_1",
        tool_name="write_file",
        arguments={"file_path": "a.py", "content": "VALUE = 2\n"},
        path=target,
        preimage=None,
        postimage="VALUE = 2\n",
    )
    target.write_text("VALUE = 2\n", encoding="utf-8")
    assert journal.recover_write(effect) == "RESULT"
    assert journal.read(effect.effect_id)["status"] == "RESULT"


def test_effect_journal_persists_one_runtime_worker_receipt(tmp_path: Path):
    target = tmp_path / "a.py"
    journal = DurableEffectJournal(tmp_path / "state", _identity(tmp_path))
    effect = journal.intent(
        turn_id="turn_1",
        tool_call_id="call_1",
        tool_name="write_file",
        arguments={"file_path": "a.py", "content": "done\n"},
        path=target,
        preimage=None,
        postimage="done\n",
    )
    journal.recover_write(effect)
    receipt = journal.record_worker_result(effect, {"summary": "done"})
    assert receipt["schema"] == "nexus.open_swe_runtime.worker_result.v1"
    assert journal.read(effect.effect_id)["worker_result"]["effect_id"] == effect.effect_id


def _composite_effect(tmp_path: Path):
    target = tmp_path / "a.py"
    journal = DurableEffectJournal(tmp_path / "state", _identity(tmp_path))
    arguments = {
        "file_path": "a.py",
        "content": "done\n",
        "envelope": {"summary": "fixed a.py"},
    }
    effect = journal.intent(
        turn_id="turn_composite",
        tool_call_id="call_composite",
        tool_name="write_file_and_record_worker_result",
        arguments=arguments,
        path=target,
        preimage=None,
        postimage=arguments["content"],
    )
    return target, journal, effect, arguments


def test_composite_recovery_orders_intent_physical_postimage_result_then_receipt(
    tmp_path: Path,
):
    target, journal, effect, arguments = _composite_effect(tmp_path)

    intent = journal.read(effect.effect_id)
    assert intent["status"] == "INTENT"
    assert not target.exists()
    assert "worker_result" not in intent

    assert journal.recover_write(effect) == "RESULT"
    result = journal.read(effect.effect_id)
    assert target.read_text(encoding="utf-8") == arguments["content"]
    assert result["status"] == "RESULT"
    assert "worker_result" not in result

    receipt = journal.record_worker_result(effect, arguments["envelope"])
    persisted = journal.read(effect.effect_id)
    assert persisted["status"] == "RESULT"
    assert persisted["worker_result"] == receipt


def test_composite_recovery_persists_result_only_after_postimage_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target, journal, effect, _arguments = _composite_effect(tmp_path)
    real_replace = recovery.os.replace
    events: list[tuple[str, str, str | None]] = []

    def tracked_replace(source, destination):
        destination = Path(destination)
        if destination == target:
            events.append(("physical", journal.read(effect.effect_id)["status"], None))
        elif destination == journal._path(effect.effect_id):
            physical = target.read_text(encoding="utf-8") if target.exists() else None
            events.append(("journal", journal.read(effect.effect_id)["status"], physical))
        return real_replace(source, destination)

    monkeypatch.setattr("nexus_open_swe_runtime.recovery.os.replace", tracked_replace)
    journal.recover_write(effect)

    assert events[0] == ("physical", "INTENT", None)
    assert events[1] == ("journal", "INTENT", "done\n")
    assert journal.read(effect.effect_id)["status"] == "RESULT"


def test_composite_receipt_replay_is_exact_and_idempotent(tmp_path: Path, monkeypatch):
    target, journal, effect, arguments = _composite_effect(tmp_path)
    journal.recover_write(effect)
    first = journal.record_worker_result(effect, arguments["envelope"])
    restarted = DurableEffectJournal(tmp_path / "state", _identity(tmp_path))

    writes = 0
    real_replace = recovery.os.replace

    def count_replace(source, destination):
        nonlocal writes
        if Path(destination) == target:
            writes += 1
        return real_replace(source, destination)

    monkeypatch.setattr("nexus_open_swe_runtime.recovery.os.replace", count_replace)
    assert restarted.recover_write(effect) == "RESULT"
    second = restarted.record_worker_result(effect, arguments["envelope"])
    assert second == first
    assert writes == 0
    assert target.read_text(encoding="utf-8") == arguments["content"]


def test_composite_receipt_hash_is_recomputed_from_exact_receipt_material(
    tmp_path: Path,
):
    _target, journal, effect, arguments = _composite_effect(tmp_path)
    journal.recover_write(effect)
    receipt = journal.record_worker_result(effect, arguments["envelope"])
    material = dict(receipt)
    material.pop("receipt_sha256")
    expected = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    assert receipt["receipt_sha256"] == expected
    assert (
        journal.read(effect.effect_id)["worker_result_sha256"]
        == hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(lambda effect: replace(effect, postimage="other\n"), id="content"),
        pytest.param(lambda effect: replace(effect, tool_call_id="call_other"), id="call"),
        pytest.param(lambda effect: replace(effect, effect_id="effect_" + "f" * 64), id="effect"),
    ],
)
def test_composite_receipt_rejects_changed_effect_identity(
    tmp_path: Path,
    mutation,
):
    _target, journal, effect, arguments = _composite_effect(tmp_path)
    journal.recover_write(effect)
    journal.record_worker_result(effect, arguments["envelope"])
    with pytest.raises((RuntimeError, FileNotFoundError), match="RECOVERY|effect"):
        journal.record_worker_result(mutation(effect), arguments["envelope"])


def test_composite_receipt_rejects_changed_summary_on_exact_effect(tmp_path: Path):
    _target, journal, effect, arguments = _composite_effect(tmp_path)
    journal.recover_write(effect)
    journal.record_worker_result(effect, arguments["envelope"])
    with pytest.raises(RuntimeError, match="RECOVERY"):
        journal.record_worker_result(effect, {"summary": "different"})


def test_composite_recovery_rejects_divergent_physical_bytes_without_receipt(
    tmp_path: Path,
):
    target, journal, effect, arguments = _composite_effect(tmp_path)
    target.write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="PREIMAGE_MISMATCH"):
        journal.recover_write(effect)
    state = journal.read(effect.effect_id)
    assert state["status"] == "INTENT"
    assert "worker_result" not in state
    assert target.read_text(encoding="utf-8") != arguments["content"]


def test_composite_recovery_replays_physical_write_without_duplicate_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target, journal, effect, _arguments = _composite_effect(tmp_path)
    replaces = 0
    real_replace = recovery.os.replace

    def count_physical_replace(source, destination):
        nonlocal replaces
        if Path(destination) == target:
            replaces += 1
        return real_replace(source, destination)

    monkeypatch.setattr("nexus_open_swe_runtime.recovery.os.replace", count_physical_replace)
    assert journal.recover_write(effect) == "RESULT"
    assert journal.recover_write(effect) == "RESULT"
    assert replaces == 1


def test_effect_journal_recovers_edit_from_preimage(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("old\n", encoding="utf-8")
    journal = DurableEffectJournal(tmp_path / "state", _identity(tmp_path))
    effect = journal.intent(
        turn_id="turn_1",
        tool_call_id="call_1",
        tool_name="edit_file",
        arguments={"file_path": "a.py"},
        path=target,
        preimage="old\n",
        postimage="new\n",
    )
    assert journal.recover_write(effect) == "RESULT"
    assert target.read_text(encoding="utf-8") == "new\n"
    assert journal.read(effect.effect_id)["status"] == "RESULT"


def test_effect_journal_accepts_edit_postimage_and_rejects_divergent(tmp_path: Path):
    journal = DurableEffectJournal(tmp_path / "state", _identity(tmp_path))
    postimage = tmp_path / "post.py"
    postimage.write_text("new\n", encoding="utf-8")
    effect = journal.intent(
        turn_id="turn_1",
        tool_call_id="call_post",
        tool_name="edit_file",
        arguments={"file_path": "post.py"},
        path=postimage,
        preimage="old\n",
        postimage="new\n",
    )
    assert journal.recover_write(effect) == "RESULT"

    divergent = tmp_path / "divergent.py"
    divergent.write_text("other\n", encoding="utf-8")
    effect = journal.intent(
        turn_id="turn_1",
        tool_call_id="call_divergent",
        tool_name="edit_file",
        arguments={"file_path": "divergent.py"},
        path=divergent,
        preimage="old\n",
        postimage="new\n",
    )
    with pytest.raises(RuntimeError, match="PREIMAGE_MISMATCH"):
        journal.recover_write(effect)
    assert divergent.read_text(encoding="utf-8") == "other\n"


def test_scoped_backend_replays_edit_result_sync_and_async(tmp_path: Path):
    import asyncio

    target = tmp_path / "a.py"
    target.write_text("old\n", encoding="utf-8")
    journal = DurableEffectJournal(tmp_path / "state", _identity(tmp_path))
    journal.bind_turn("turn_9", "call_9")
    backend = cli.ScopedRepairBackend(object(), tmp_path, ("a.py",), journal)
    result = backend.edit("a.py", "old\n", "new\n")
    assert result.path == "a.py"
    assert result.occurrences == 1
    assert target.read_text(encoding="utf-8") == "new\n"

    target.write_text("new\n", encoding="utf-8")
    journal.bind_turn("turn_10", "call_10")
    result = asyncio.run(backend.aedit("a.py", "new\n", "final\n"))
    assert result.path == "a.py"
    assert result.occurrences == 1
    assert target.read_text(encoding="utf-8") == "final\n"


@pytest.mark.parametrize("newline", ["\r\n", "\r"])
def test_scoped_backend_edit_normalizes_hostile_newlines_once(tmp_path: Path, newline: str):
    target = tmp_path / "a.py"
    target.write_bytes(f"old{newline}".encode())
    journal = DurableEffectJournal(tmp_path / "state", _identity(tmp_path))
    journal.bind_turn("turn-9", "call-9")
    backend = cli.ScopedRepairBackend(object(), tmp_path, ("a.py",), journal)

    result = backend.edit("a.py", f"old{newline}", f"new{newline}")

    assert result.path == "a.py"
    assert result.occurrences == 1
    assert target.read_bytes() == b"new\n"
    records = list(journal.root.glob("*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text(encoding="utf-8"))["status"] == "RESULT"


def test_scoped_backend_records_write_effect_and_reads_operation_turn(tmp_path: Path):
    target = tmp_path / "a.py"
    journal = DurableEffectJournal(tmp_path / "state", _identity(tmp_path))
    journal.bind_turn("turn_9", "call_9")

    class Delegate:
        def write(self, file_path, content):
            target.write_text(content, encoding="utf-8")

    backend = cli.ScopedRepairBackend(Delegate(), tmp_path, ("a.py",), journal)
    backend.write("a.py", "done\n")
    records = list((tmp_path / "state" / "recovery" / "effects").glob("*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text(encoding="utf-8"))["status"] == "RESULT"


def test_scoped_backend_replays_result_without_second_delegate(tmp_path: Path):
    target = tmp_path / "a.py"
    journal = DurableEffectJournal(tmp_path / "state", _identity(tmp_path))
    journal.bind_turn("turn_9", "call_9")
    calls = 0

    class Delegate:
        def write(self, file_path, content):
            nonlocal calls
            calls += 1
            target.write_text(content, encoding="utf-8")

    backend = cli.ScopedRepairBackend(Delegate(), tmp_path, ("a.py",), journal)
    backend.write("a.py", "done\n")
    backend.write("a.py", "done\n")
    assert calls == 0


def test_operation_journal_lease_serializes_processes(tmp_path: Path):
    journal = DurableOperationJournal(tmp_path / "state", _identity(tmp_path))
    with journal.fence():
        assert journal.read()["lease_epoch"] == 1


def test_operation_fence_serializes_real_processes(tmp_path: Path):
    identity = _identity(tmp_path)
    journal = DurableOperationJournal(tmp_path / "state", identity)
    journal.prepare()
    marker = tmp_path / "fence-events.log"
    processes = [
        multiprocessing.get_context("fork").Process(
            target=_contending_fence,
            args=(str(tmp_path / "state"), identity.workspace, str(marker)),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=3)
    assert marker.read_text(encoding="utf-8").splitlines() in (
        ["entered", "leaving", "entered", "leaving"],
    )
    assert all(process.exitcode == 0 for process in processes)
    assert journal.read()["lease_epoch"] == 3


def test_reconcile_process_fence_gives_one_winner_and_zero_call_loser(tmp_path: Path):
    state_root = tmp_path / "state"
    request = {"runtime_state_root": str(state_root), "operation_id": "a" * 64}
    marker = tmp_path / "winner.marker"
    counts = tmp_path / "counts.log"

    def callback(req, **_kwargs):
        with marker.open("a+", encoding="utf-8") as stream:
            stream.seek(0)
            winner = bool(stream.read())
            if winner:
                return {"status": "COMPLETED", "calls": 0}
            stream.write("winner")
            stream.flush()
        with counts.open("a", encoding="utf-8") as stream:
            stream.write("detail update invoke effect\n")
        return {"status": "COMPLETED", "calls": 1}

    wrapped = cli._fenced_worker_reconcile(callback)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(lambda _: wrapped(request), range(2))
    assert first["status"] == second["status"] == "COMPLETED"
    assert first["calls"] + second["calls"] == 1
    assert counts.read_text(encoding="utf-8").splitlines() == ["detail update invoke effect"]


def test_reconcile_bound_turn_reads_exact_detail_without_ask_or_history(tmp_path: Path):
    journal = DurableOperationJournal(tmp_path / "state", _identity(tmp_path))
    journal.prepare()
    journal.ask_dispatching(turn_id="turn_1", prompt="payload", ordinal=0)
    journal.conversation_bound("conversation-1")

    class Model:
        _conversation_id = None

        def _detail_response(self, conversation_id, *, wait, turn_id):
            assert conversation_id == "conversation-1"
            assert wait is False
            assert turn_id == "turn_1"
            return '{"type":"tool_call"}'

    assert reconcile_bound_turn(journal, Model()) == '{"type":"tool_call"}'


def test_reconcile_bound_turn_reuses_persisted_response_without_detail(tmp_path: Path):
    journal = DurableOperationJournal(tmp_path / "state", _identity(tmp_path))
    journal.prepare()
    journal.ask_dispatching(turn_id="turn_1", prompt="payload", ordinal=0)
    journal.conversation_bound("conversation-1")
    response = '{"type":"tool_call"}'
    journal.response_recovered("turn_1", response)

    class Model:
        def _detail_response(self, *args, **kwargs):
            raise AssertionError("reconcile must not detail a recovered response")

    assert reconcile_bound_turn(journal, Model()) == response


def test_restart_trace_recovered_response_updates_graph_once_and_terminal_reconcile_is_idempotent(
    tmp_path: Path,
):
    journal = DurableOperationJournal(tmp_path / "state", _identity(tmp_path))
    journal.prepare()
    journal.ask_dispatching(turn_id="turn_1", prompt="payload", ordinal=0)
    journal.conversation_bound("conversation-1")
    calls: list[str] = []

    class Model:
        _conversation_id = None

        def _detail_response(self, conversation_id, *, wait, turn_id):
            calls.append(f"detail:{conversation_id}:{wait}:{turn_id}")
            return '{"type":"tool_call","name":"write_file","arguments":{"file_path":"a.py","content":"done"}}'

    recovered = reconcile_bound_turn(journal, Model())
    assert '"write_file"' in recovered
    assert calls == ["detail:conversation-1:False:turn_1"]
    journal.terminal({"status": "COMPLETED"})
    assert journal.read()["status"] == "COMPLETED"
    assert calls == ["detail:conversation-1:False:turn_1"]
