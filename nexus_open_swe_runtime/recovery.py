"""Crash durable recovery primitives for one bounded worker operation.

This module deliberately contains no model or Nexus policy.  It owns the small
amount of durable state needed to resume a DeepAgents repair after its host
process disappears.  Every record is owner-only, replaced atomically, and
fsynced before an external effect is allowed to proceed.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

_FLOCKS: dict[str, tuple[int, int, int]] = {}
_FLOCKS_GUARD = threading.RLock()


@contextlib.contextmanager
def operation_flock(lock_path: str | Path) -> Iterator[None]:
    """Re-entrant in-process wrapper around the cross-process operation flock."""
    path = str(Path(lock_path).expanduser().resolve())
    while True:
        with _FLOCKS_GUARD:
            held = _FLOCKS.get(path)
            if held is None or held[2] == threading.get_ident():
                break
        time.sleep(0.001)
    with _FLOCKS_GUARD:
        if held is not None and held[2] == threading.get_ident():
            _FLOCKS[path] = (held[0], held[1] + 1, held[2])
            try:
                yield
            finally:
                with _FLOCKS_GUARD:
                    fd, depth, owner = _FLOCKS[path]
                    if depth <= 1:
                        _FLOCKS.pop(path, None)
                    else:
                        _FLOCKS[path] = (fd, depth - 1, owner)
            return
        Path(path).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        _FLOCKS[path] = (fd, 1, threading.get_ident())
    try:
        yield
    finally:
        with _FLOCKS_GUARD:
            _FLOCKS.pop(path, None)
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(value: bytes | str) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def _fsync_replace(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink() or path.parent.is_symlink():
        raise RuntimeError("RECOVERY_STATE_SYMLINK")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(dict(value), stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


@dataclass(frozen=True)
class RecoveryIdentity:
    operation_id: str
    execution_material_sha256: str
    workspace: str
    task_id: str
    unit_id: str
    session_id: str
    allowed_paths: tuple[str, ...]
    provider_id: str
    model_id: str
    worker_identity_sha256: str
    transport_config_sha256: str
    runtime_identity_sha256: str
    checkpoint_namespace: str = "open-swe-repair-v1"

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["allowed_paths"] = list(self.allowed_paths)
        return value

    def digest(self) -> str:
        return _sha(_canonical(self.as_dict()))


class DurableOperationJournal:
    """One operation's state and its owner-only cross-process fence."""

    def __init__(self, state_root: str | Path, identity: RecoveryIdentity):
        self.root = Path(state_root).expanduser().resolve() / "recovery"
        self.identity = identity
        self.effect_journal: DurableEffectJournal | None = None
        self._current_turn_id = ""
        self._current_tool_call_id = ""
        self._fence_depth = 0
        self.path = self.root / "operations" / f"{identity.operation_id}.json"
        self.lock_path = self.root / "locks" / f"{identity.operation_id}.lock"

    @classmethod
    def open(cls, state_root: str | Path, identity: RecoveryIdentity) -> "DurableOperationJournal":
        journal = cls(state_root, identity)
        state = journal.read()
        if state and state.get("identity") != identity.as_dict():
            raise RuntimeError("RECOVERY_IDENTITY_MISMATCH")
        return journal

    @contextlib.contextmanager
    def fence(self) -> Iterator[dict[str, Any]]:
        if self._fence_depth:
            self._fence_depth += 1
            try:
                yield self.read()
            finally:
                self._fence_depth -= 1
            return
        with operation_flock(self.lock_path):
            self._fence_depth = 1
            state = self.read() or {}
            if state and state.get("identity") != self.identity.as_dict():
                raise RuntimeError("RECOVERY_IDENTITY_MISMATCH")
            epoch = int(state.get("lease_epoch", 0)) + 1
            state = dict(state)
            state.update({"identity": self.identity.as_dict(), "lease_epoch": epoch})
            self._write(state)
            yield state
            self._fence_depth = 0

    def _write(self, state: Mapping[str, Any]) -> None:
        _fsync_replace(self.path, state)

    def read(self) -> dict[str, Any]:
        if self.path.is_symlink():
            raise RuntimeError("RECOVERY_STATE_SYMLINK")
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return {}
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("RECOVERY_RECORD_CORRUPT") from exc
        if not isinstance(value, dict):
            raise RuntimeError("RECOVERY_RECORD_CORRUPT")
        if value.get("identity") != self.identity.as_dict():
            raise RuntimeError("RECOVERY_IDENTITY_MISMATCH")
        return value

    def _transition(self, **updates: Any) -> dict[str, Any]:
        with self.fence() as state:
            state.update(updates)
            self._write(state)
            return dict(state)

    def prepare(self) -> dict[str, Any]:
        return self._transition(status="PREPARED", checkpoint_namespace=self.identity.checkpoint_namespace)

    def ask_dispatching(self, *, turn_id: str, prompt: str, ordinal: int) -> dict[str, Any]:
        if not turn_id or ordinal < 0:
            raise RuntimeError("RECOVERY_TURN_INVALID")
        self._current_turn_id = turn_id
        if self.effect_journal is not None:
            self.effect_journal.bind_turn(turn_id)
        return self._transition(
            status="ASK_DISPATCHING",
            turn_id=turn_id,
            prompt_sha256=_sha(prompt),
            turn_ordinal=ordinal,
        )

    def conversation_bound(self, conversation_id: str) -> dict[str, Any]:
        if not conversation_id or "\x00" in conversation_id:
            raise RuntimeError("RECOVERY_CONVERSATION_INVALID")
        return self._transition(status="CONVERSATION_BOUND", conversation_id=conversation_id)

    def response_recovered(self, turn_id: str, response: str) -> dict[str, Any]:
        state = self.read()
        if state.get("turn_id") != turn_id or not state.get("conversation_id"):
            raise RuntimeError("RECOVERY_TURN_IDENTITY_UNKNOWN")
        return self._transition(
            status="RESPONSE_RECOVERED",
            response_sha256=_sha(response),
            recovered_response=response,
        )

    def protocol_repair_started(
        self, *, origin: str, origin_sha256: str, turn_id: str
    ) -> dict[str, Any]:
        """Persist the repair origin before dispatching its external turn."""
        if not origin or not origin_sha256 or not turn_id:
            raise RuntimeError("RECOVERY_PROTOCOL_REPAIR_INVALID")
        state = self.read()
        return self._transition(
            protocol_repair_origin=origin,
            protocol_repair_origin_sha256=origin_sha256,
            protocol_repair_turn_id=turn_id,
            protocol_repair_original_conversation_id=state.get("conversation_id", ""),
            protocol_repair_status="DISPATCHING",
        )

    def protocol_repair_recovered(self, response: str) -> dict[str, Any]:
        state = self.read()
        origin = state.get("protocol_repair_origin")
        expected = state.get("protocol_repair_origin_sha256")
        if not isinstance(origin, str) or not isinstance(expected, str) or _sha(origin) != expected:
            raise RuntimeError("RECOVERY_PROTOCOL_REPAIR_ORIGIN_INVALID")
        return self._transition(
            protocol_repair_status="RECOVERED",
            protocol_repair_response_sha256=_sha(response),
            protocol_repair_response=response,
        )

    def checkpoint_bound(self, checkpoint_id: str) -> dict[str, Any]:
        return self._transition(checkpoint_id=checkpoint_id, checkpoint_status="BOUND")

    def terminal(self, result: Mapping[str, Any]) -> dict[str, Any]:
        return self._transition(status="COMPLETED", terminal_result=dict(result))


@dataclass(frozen=True)
class Effect:
    effect_id: str
    operation_id: str
    turn_id: str
    tool_call_id: str
    tool_name: str
    path: str
    preimage_sha256: str
    postimage_sha256: str
    postimage: str
    arguments: Mapping[str, Any]


class DurableEffectJournal:
    """Durable INTENT/RESULT records for scoped file mutation."""

    def __init__(self, state_root: str | Path, identity: RecoveryIdentity):
        self.root = Path(state_root).expanduser().resolve() / "recovery" / "effects"
        self.identity = identity
        self._current_turn_id = ""
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def bind_turn(self, turn_id: str, tool_call_id: str = "") -> None:
        self._current_turn_id = turn_id
        self._current_tool_call_id = tool_call_id

    def _path(self, effect_id: str) -> Path:
        return self.root / f"{effect_id}.json"

    def intent(
        self, *, turn_id: str, tool_call_id: str, tool_name: str, arguments: Mapping[str, Any],
        path: Path, preimage: str | None, postimage: str,
    ) -> Effect:
        if tool_name not in {"write_file", "edit_file"}:
            raise RuntimeError("RECOVERY_TOOL_FORBIDDEN")
        effect_id = "effect_" + _sha(_canonical({
            "operation_id": self.identity.operation_id, "turn_id": turn_id,
            "tool_call_id": tool_call_id, "tool_name": tool_name,
            "arguments": dict(arguments),
        }))
        existing_path = self._path(effect_id)
        if existing_path.is_file() and not existing_path.is_symlink():
            existing = self.read(effect_id)
            if existing.get("status") in {"INTENT", "RESULT"}:
                if (
                    existing.get("operation_id") != self.identity.operation_id
                    or existing.get("turn_id") != turn_id
                    or existing.get("tool_call_id") != tool_call_id
                    or existing.get("tool_name") != tool_name
                    or existing.get("path") != str(path)
                    or existing.get("arguments") != dict(arguments)
                ):
                    raise RuntimeError("RECOVERY_EFFECT_IDENTITY_MISMATCH")
                return Effect(
                    effect_id, self.identity.operation_id, turn_id, tool_call_id, tool_name,
                    str(path), str(existing["preimage_sha256"]), str(existing["postimage_sha256"]),
                    str(existing["postimage"]), dict(existing["arguments"]),
                )
        effect = Effect(
            effect_id, self.identity.operation_id, turn_id, tool_call_id, tool_name,
            str(path), _sha(preimage) if preimage is not None else "absent",
            _sha(postimage), postimage,
            dict(arguments),
        )
        _fsync_replace(self._path(effect_id), {"status": "INTENT", **asdict(effect)})
        return effect

    def read(self, effect_id: str) -> dict[str, Any]:
        path = self._path(effect_id)
        if path.is_symlink():
            raise RuntimeError("RECOVERY_STATE_SYMLINK")
        return json.loads(path.read_text(encoding="utf-8"))

    def recover_write(self, effect: Effect) -> str:
        state = self.read(effect.effect_id)
        path = Path(effect.path)
        if path.is_symlink():
            raise RuntimeError("RECOVERY_WRITE_SYMLINK")
        if state.get("status") == "RESULT":
            return "RESULT"
        actual = _sha(path.read_text(encoding="utf-8")) if path.exists() else "absent"
        if effect.tool_name == "edit_file":
            if actual == effect.postimage_sha256:
                _fsync_replace(self._path(effect.effect_id), {"status": "RESULT", **asdict(effect)})
                return "RESULT"
            raise RuntimeError("RECOVERY_EDIT_UNRESOLVED_INTENT")
        if actual == effect.postimage_sha256:
            _fsync_replace(self._path(effect.effect_id), {"status": "RESULT", **asdict(effect)})
            return "RESULT"
        if actual != effect.preimage_sha256:
            raise RuntimeError("RECOVERY_WRITE_PREIMAGE_MISMATCH")
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(effect.postimage)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
        if _sha(path.read_text(encoding="utf-8")) != effect.postimage_sha256:
            raise RuntimeError("RECOVERY_WRITE_POSTIMAGE_MISMATCH")
        _fsync_replace(self._path(effect.effect_id), {"status": "RESULT", **asdict(effect)})
        return "RESULT"


def checkpoint_path(state_root: str | Path, operation_id: str) -> Path:
    root = Path(state_root).expanduser().resolve() / "recovery" / "checkpoints"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root / f"{operation_id}.sqlite"


def create_checkpoint(state_root: str | Path, operation_id: str, namespace: str) -> Any:
    """Return the supported persistent LangGraph SQLite saver."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError("LANGGRAPH_SQLITE_CHECKPOINT_UNAVAILABLE") from exc
    path = checkpoint_path(state_root, operation_id)
    if path.is_symlink():
        raise RuntimeError("RECOVERY_CHECKPOINT_SYMLINK")
    connection = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    os.chmod(path, 0o600)
    return saver, namespace


def validate_checkpoint(path: str | Path) -> bool:
    """Verify an existing checkpoint is a real initialized LangGraph SQLite DB."""
    raw = Path(path).expanduser()
    if raw.is_symlink():
        return False
    candidate = raw.resolve()
    if not candidate.is_file():
        return False
    try:
        connection = sqlite3.connect(f"file:{candidate}?mode=ro", uri=True)
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        connection.close()
    except sqlite3.Error:
        return False
    return {str(row[0]) for row in rows}.issuperset({"checkpoints", "writes"})


def reconcile_bound_turn(journal: DurableOperationJournal, model: Any) -> str:
    """Read one already-bound OpenCLI turn after restart.

    The model is deliberately restricted to its exact conversation and turn;
    this helper never invokes ``ask`` or ``history`` and therefore cannot
    duplicate the external request.
    """
    with journal.fence():
        return _reconcile_bound_turn_locked(journal, model)


def _reconcile_bound_turn_locked(journal: DurableOperationJournal, model: Any) -> str:
    state = journal.read()
    if state.get("status") == "COMPLETED":
        terminal = state.get("terminal_result")
        if isinstance(terminal, Mapping) and isinstance(terminal.get("response_text"), str):
            return str(terminal["response_text"])
        raise RuntimeError("RECOVERY_TERMINAL_RESPONSE_MISSING")
    if state.get("status") == "RESPONSE_RECOVERED":
        recovered = state.get("recovered_response")
        if not isinstance(recovered, str) or _sha(recovered) != state.get("response_sha256"):
            raise RuntimeError("RECOVERY_RESPONSE_CORRUPT")
        return recovered
    conversation_id = state.get("conversation_id")
    turn_id = state.get("turn_id")
    if not isinstance(conversation_id, str) or not conversation_id:
        raise RuntimeError("RECOVERY_CONVERSATION_ID_MISSING")
    if not isinstance(turn_id, str) or not turn_id:
        raise RuntimeError("RECOVERY_TURN_ID_MISSING")
    model._conversation_id = conversation_id
    response = model._detail_response(conversation_id, wait=False, turn_id=turn_id)
    journal.response_recovered(turn_id, response)
    return response
