from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import stat
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

from langchain_core.language_models.chat_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None  # type: ignore[assignment]

OPENCLI_WEB_PROTOCOL = "nexus.opencli_web_chat.v1"
_ALLOWED_LEVELS = frozenset({"fast", "balanced", "advanced", "very-high", "pro"})
_ALLOWED_SITE_SESSIONS = frozenset({"ephemeral", "persistent"})
_HARD_BLOCK_MARKERS = (
    "login required",
    "not logged in",
    "challenge",
    "captcha",
    "quota",
)
_BUSY_MARKERS = ("busy", "rate control", "rate-control", "too many requests")
_MIN_WEB_SEND_INTERVAL_SECONDS = 15.0
_POST_RESPONSE_SETTLE_SECONDS = 3.0
_MAX_WEB_TURNS_PER_OPERATION = 12


@dataclass
class _PacingState:
    condition: threading.Condition = field(default_factory=threading.Condition)
    in_flight: bool = False
    borrowers: int = 0
    last_send_started: float | None = None
    last_response_finished: float | None = None
    clock: Callable[[], float] = time.monotonic


_MAX_PACING_SESSION_KEYS = 64
_PACING_STATES_LOCK = threading.Lock()
_PACING_STATES: OrderedDict[tuple[str, str, str], _PacingState] = OrderedDict()


def _evict_idle_pacing_states(*, keep: tuple[str, str, str] | None = None) -> None:
    for stale_key, stale_state in tuple(_PACING_STATES.items()):
        if stale_key == keep or stale_state.in_flight or stale_state.borrowers:
            continue
        now = stale_state.clock()
        if (
            stale_state.last_send_started is not None
            and now < stale_state.last_send_started + _MIN_WEB_SEND_INTERVAL_SECONDS
        ) or (
            stale_state.last_response_finished is not None
            and now < stale_state.last_response_finished + _POST_RESPONSE_SETTLE_SECONDS
        ):
            continue
        del _PACING_STATES[stale_key]
        if len(_PACING_STATES) <= _MAX_PACING_SESSION_KEYS:
            break


def _shared_pacing_state(
    key: tuple[str, str, str],
    *,
    borrow: bool = False,
    clock: Callable[[], float] = time.time,
) -> _PacingState:
    with _PACING_STATES_LOCK:
        state = _PACING_STATES.get(key)
        created = state is None
        if state is None:
            state = _PacingState()
            state.clock = clock
            _PACING_STATES[key] = state
        else:
            _PACING_STATES.move_to_end(key)
        if borrow:
            state.borrowers += 1
        if len(_PACING_STATES) > _MAX_PACING_SESSION_KEYS:
            _evict_idle_pacing_states(keep=key)
            if created and len(_PACING_STATES) > _MAX_PACING_SESSION_KEYS:
                if borrow:
                    state.borrowers -= 1
                del _PACING_STATES[key]
                raise OpenCLIWebModelError("OPENCLI_WEB_PACING_REGISTRY_EXHAUSTED")
        return state


def _release_shared_pacing_state(key: tuple[str, str, str], state: _PacingState) -> None:
    with _PACING_STATES_LOCK:
        if state.borrowers <= 0:
            raise RuntimeError("OPENCLI_WEB_PACING_BORROW_UNDERFLOW")
        state.borrowers -= 1
        if len(_PACING_STATES) > _MAX_PACING_SESSION_KEYS:
            _evict_idle_pacing_states()


# ---------------------------------------------------------------------------
# Durable cross-process / restart pacing gate
# ---------------------------------------------------------------------------

_DURABLE_MIN_SEND_INTERVAL = _MIN_WEB_SEND_INTERVAL_SECONDS
_DURABLE_MIN_RESPONSE_SETTLE = _POST_RESPONSE_SETTLE_SECONDS
_DURABLE_STATE_SCHEMA = "nexus.opencli_durable_pacing.v1"
_DURABLE_CLOCK_SKEW_FUTURE = 300.0
_DURABLE_CLOCK_SKEW_PAST = 86400.0
_PACING_DIR_MODE = 0o700
_STATE_FILE_MODE = 0o600
_LOCK_FILE_MODE = 0o600


def _durable_pacing_key(executable: str, profile: str, site_session: str) -> str:
    """Hashed transport identity for durable pacing files."""
    material = f"{executable}\0{profile}\0{site_session}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


@dataclass
class DurablePacingState:
    """On-disk pacing record: timestamps only, no plaintext identity."""

    key: str = ""
    last_send_started: float = 0.0
    last_response_finished: float = 0.0


@dataclass
class DurablePacingLock:
    """Manages an fcntl.flock on a lock file under pacing_dir."""

    lock_path: str
    _fd: int = -1

    def acquire(self) -> None:
        fcntl = _fcntl
        if fcntl is None:
            raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_FLOCK_UNAVAILABLE")
        _validate_lock_file(self.lock_path)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        self._fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | nofollow, _LOCK_FILE_MODE)
        try:
            os.fchmod(self._fd, _LOCK_FILE_MODE)
            st = os.fstat(self._fd)
            if st.st_uid != os.getuid() or not stat.S_ISREG(st.st_mode) or st.st_mode & 0o077:
                raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_LOCK_UNSAFE")
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        except Exception:
            os.close(self._fd)
            self._fd = -1
            raise

    def release(self) -> None:
        if self._fd >= 0:
            try:
                fcntl = _fcntl
                if fcntl is not None:
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = -1

    def __enter__(self) -> "DurablePacingLock":
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


def _ensure_pacing_dir(pacing_dir: str) -> None:
    if os.path.lexists(pacing_dir) and (
        os.path.islink(pacing_dir) or not os.path.isdir(pacing_dir)
    ):
        raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_STATE_UNSAFE")
    os.makedirs(pacing_dir, mode=_PACING_DIR_MODE, exist_ok=True)
    os.chmod(pacing_dir, _PACING_DIR_MODE)
    st = os.stat(pacing_dir)
    if st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_STATE_UNSAFE")


def _validate_lock_file(path: str) -> None:
    if not os.path.lexists(path):
        return
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_LOCK_UNSAFE") from exc
    if (
        stat.S_ISLNK(st.st_mode)
        or not stat.S_ISREG(st.st_mode)
        or st.st_uid != os.getuid()
        or st.st_mode & 0o077
    ):
        raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_LOCK_UNSAFE")


def _read_validated_state(
    path: str,
    *,
    expected_key: str | None = None,
    clock: Callable[[], float] | None = None,
) -> DurablePacingState | None:
    """Read and validate one inode through one descriptor-bound operation."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(fd)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_uid != os.getuid()
                or opened.st_mode & 0o077
            ):
                raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_STATE_UNSAFE")
            with os.fdopen(fd, encoding="utf-8") as fh:
                fd = -1
                raw = json.load(fh)
        finally:
            if fd >= 0:
                os.close(fd)
    except FileNotFoundError:
        return None
    except OpenCLIWebModelError:
        raise
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_STATE_UNSAFE") from exc
        raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_STATE_MALFORMED") from exc
    except ValueError as exc:
        raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_STATE_MALFORMED") from exc
    if not isinstance(raw, Mapping):
        raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_STATE_MALFORMED")
    if raw.get("schema") != _DURABLE_STATE_SCHEMA:
        raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_STATE_MALFORMED")
    key = raw.get("key")
    if not isinstance(key, str) or not key:
        raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_STATE_MALFORMED")
    if expected_key is not None and key != expected_key:
        raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_STATE_IDENTITY_UNKNOWN")
    for field_name in ("last_send_started", "last_response_finished"):
        value = raw.get(field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_STATE_MALFORMED")
        if not math.isfinite(float(value)) or value < 0:
            raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_STATE_MALFORMED")
    if clock is not None:
        now = clock()
        for field_name in ("last_send_started", "last_response_finished"):
            value = raw.get(field_name)
            if isinstance(value, (int, float)) and value > 0:
                if value > now + _DURABLE_CLOCK_SKEW_FUTURE:
                    raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_STATE_UNSAFE")
                if now - value > _DURABLE_CLOCK_SKEW_PAST:
                    raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_STATE_UNSAFE")
    return DurablePacingState(
        key=key,
        last_send_started=float(raw["last_send_started"]),
        last_response_finished=float(raw["last_response_finished"]),
    )


def _validate_state_file(path: str, *, clock: Callable[[], float] | None = None) -> None:
    """Fail-closed validation of a pacing state file before use."""
    _read_validated_state(path, clock=clock)


def _read_durable_state(path: str) -> DurablePacingState:
    try:
        state = _read_validated_state(path)
    except OpenCLIWebModelError:
        return DurablePacingState()
    return state or DurablePacingState()


def _write_durable_state(path: str, state: DurablePacingState) -> None:
    payload = {
        "schema": _DURABLE_STATE_SCHEMA,
        "key": state.key,
        "last_send_started": state.last_send_started,
        "last_response_finished": state.last_response_finished,
    }
    parent = os.path.dirname(path)
    _ensure_pacing_dir(parent)
    fd, tmp = tempfile.mkstemp(prefix=".pacing.", dir=parent, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True, separators=(",", ":"))
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, _STATE_FILE_MODE)
        parent_stat = os.stat(parent)
        if parent_stat.st_uid != os.getuid() or parent_stat.st_mode & 0o077:
            raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_STATE_UNSAFE")
        os.replace(tmp, path)
        os.chmod(path, _STATE_FILE_MODE)
        dir_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


class DurablePacingBackend:
    """Cross-process/restart pacing gate using fcntl.flock + atomic timestamp state.

    Stores only hashed transport identity and send/response timestamps.  No
    plaintext profile, session, executable, prompts, or conversation state.
    """

    def __init__(
        self,
        runtime_state_root: str,
        *,
        executable: str,
        profile: str,
        site_session: str,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._key = _durable_pacing_key(executable, profile, site_session)
        if os.path.lexists(runtime_state_root) and (
            os.path.islink(runtime_state_root) or not os.path.isdir(runtime_state_root)
        ):
            raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_STATE_UNSAFE")
        os.makedirs(runtime_state_root, mode=_PACING_DIR_MODE, exist_ok=True)
        os.chmod(runtime_state_root, _PACING_DIR_MODE)
        root_stat = os.stat(runtime_state_root)
        if root_stat.st_uid != os.getuid() or root_stat.st_mode & 0o077:
            raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_STATE_UNSAFE")
        pacing_dir = os.path.join(runtime_state_root, "pacing")
        _ensure_pacing_dir(pacing_dir)
        self._state_path = os.path.join(pacing_dir, f"{self._key}.json")
        self._lock_path = os.path.join(pacing_dir, f"{self._key}.lock")
        self._clock = clock

    @contextmanager
    def acquire_send_lock(self) -> Iterator[DurablePacingState]:
        """Acquire flock, validate state, yield current state."""
        lock = DurablePacingLock(self._lock_path)
        lock.acquire()
        try:
            state = (
                _read_validated_state(
                    self._state_path,
                    expected_key=self._key,
                    clock=self._clock,
                )
                or DurablePacingState()
            )
            yield state
        finally:
            lock.release()

    def release_send_lock(self, lock: DurablePacingLock) -> None:
        lock.release()

    def state_path(self) -> str:
        return self._state_path

    def lock_path(self) -> str:
        return self._lock_path

    def persisted_state(self) -> DurablePacingState:
        return (
            _read_validated_state(
                self._state_path,
                expected_key=self._key,
                clock=self._clock,
            )
            or DurablePacingState()
        )

    def write_state(self, state: DurablePacingState) -> None:
        if state.key != self._key:
            raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_STATE_IDENTITY_UNKNOWN")
        _write_durable_state(self._state_path, state)


class OpenCLIWebModelError(RuntimeError):
    """Bounded failure from the ChatGPT Web transport."""


def _message_role(message: BaseMessage) -> str:
    if message.type == "human":
        return "user"
    if message.type == "ai":
        return "assistant"
    return str(message.type)


def _jsonable_content(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_jsonable_content(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable_content(item) for key, item in value.items()}
    return str(value)


def _message_payload(message: BaseMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": _message_role(message),
        "content": _jsonable_content(message.content),
    }
    if isinstance(message, AIMessage) and message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": str(call.get("id") or ""),
                "name": str(call.get("name") or ""),
                "arguments": dict(call.get("args") or {}),
            }
            for call in message.tool_calls
        ]
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = str(message.tool_call_id)
    return payload


def _tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    if not isinstance(function, Mapping):
        return ""
    name = function.get("name")
    return str(name) if isinstance(name, str) else ""


def _tool_call_id(name: str, arguments: Mapping[str, Any], raw: str) -> str:
    canonical = json.dumps(
        {"name": name, "arguments": dict(arguments), "raw": raw},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "opencli_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


class OpenCLIWebChatModel(BaseChatModel):
    """LangChain chat-model bridge backed by ChatGPT Web through OpenCLI.

    The bridge deliberately owns no repository tools and no Nexus authority. It
    only translates LangChain messages/tool declarations into one ChatGPT Web
    model turn and converts an explicit web response back into an ``AIMessage``.
    Tool execution remains inside Deep Agents / the external Open SWE runtime.

    Retained-conversation recovery and stricter response-schema enforcement are
    intentionally separate follow-up gates; this W3 bridge uses a fresh ChatGPT
    conversation for each model turn.
    """

    executable: str = "opencli"
    intelligence_level: str = "very-high"
    profile: str = ""
    timeout_seconds: int = 120
    site_session: str = "ephemeral"
    disable_streaming: bool = True
    runtime_state_root: str | None = None
    _conversation_id: str | None = PrivateAttr(default=None)
    _sleep: Callable[[float], None] = PrivateAttr(default=time.sleep)
    # Durable pacing uses epoch wall-clock timestamps so state survives reboot.
    _clock: Callable[[], float] = PrivateAttr(default=time.time)
    _pacing_state: _PacingState = PrivateAttr(default_factory=_PacingState)
    _web_turn_count: int = PrivateAttr(default=0)
    _budget_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _durable_pacing_backend: DurablePacingBackend | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context) if hasattr(super(), "model_post_init") else None
        if self.runtime_state_root and self.profile:
            self._durable_pacing_backend = DurablePacingBackend(
                self.runtime_state_root,
                executable=self.executable,
                profile=self.profile,
                site_session=self.site_session,
                clock=self._clock,
            )

    @property
    def model_name(self) -> str:
        """Provider-native identifier used by the Deep Agents harness registry."""
        return self.intelligence_level

    @property
    def _llm_type(self) -> str:
        return "opencli-chatgpt-web"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        params = dict(super()._get_ls_params(stop=stop, **kwargs))
        params["ls_provider"] = "opencli_chatgpt"
        params["ls_model_name"] = self.intelligence_level
        return params

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "executable": self.executable,
            "intelligence_level": self.intelligence_level,
            "site_session": self.site_session,
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=formatted, **kwargs)

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.profile:
            env["OPENCLI_PROFILE"] = self.profile
        return env

    def _run(self, argv: Sequence[str]) -> str:
        try:
            result = subprocess.run(
                list(argv),
                check=False,
                capture_output=True,
                text=True,
                timeout=max(self.timeout_seconds + 5, 35),
                shell=False,
                env=self._environment(),
            )
        except FileNotFoundError as exc:
            raise OpenCLIWebModelError("OPENCLI_NOT_FOUND") from exc
        except subprocess.TimeoutExpired as exc:
            raise OpenCLIWebModelError("OPENCLI_WEB_TIMEOUT") from exc
        if result.returncode != 0:
            diagnostic = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
            if "timed out" in diagnostic and "may still complete" in diagnostic:
                raise OpenCLIWebModelError("OPENCLI_WEB_TIMEOUT")
            if any(marker in diagnostic for marker in _HARD_BLOCK_MARKERS):
                raise OpenCLIWebModelError("OPENCLI_WEB_HARD_BLOCK")
            if any(marker in diagnostic for marker in _BUSY_MARKERS):
                raise OpenCLIWebModelError("OPENCLI_WEB_BUSY")
            raise OpenCLIWebModelError("OPENCLI_WEB_PROCESS_FAILURE")
        return result.stdout or ""

    def _cooldown_and_probe_status(self) -> None:
        self._sleep(60.0)
        self._run(
            [
                self.executable,
                "chatgpt",
                "status",
                "--site-session",
                self.site_session,
                "-f",
                "json",
            ]
        )
        raise OpenCLIWebModelError("OPENCLI_WEB_BUSY")

    def _select_intelligence_level(self) -> None:
        if self.site_session not in _ALLOWED_SITE_SESSIONS:
            raise OpenCLIWebModelError("OPENCLI_WEB_SITE_SESSION_INVALID")
        if self.intelligence_level not in _ALLOWED_LEVELS:
            raise OpenCLIWebModelError("OPENCLI_WEB_MODEL_LEVEL_INVALID")
        argv = [
            self.executable,
            "chatgpt",
            "model",
            self.intelligence_level,
            "--site-session",
            self.site_session,
            "-f",
            "json",
        ]
        try:
            self._run(argv)
        except OpenCLIWebModelError as exc:
            if str(exc) == "OPENCLI_WEB_BUSY":
                self._cooldown_and_probe_status()
            if str(exc) != "OPENCLI_WEB_PROCESS_FAILURE":
                raise
            self._sleep(10.0)
            try:
                self._run(argv)
            except OpenCLIWebModelError as retry_exc:
                if str(retry_exc) == "OPENCLI_WEB_BUSY":
                    self._cooldown_and_probe_status()
                raise

    def _render_prompt(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Mapping[str, Any]],
        tool_choice: str | None,
    ) -> str:
        message_payloads = [_message_payload(message) for message in messages]
        tool_payloads = [dict(tool) for tool in tools]
        turn_material = json.dumps(
            {
                "messages": message_payloads,
                "tools": tool_payloads,
                "tool_choice": tool_choice or "auto",
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        envelope = {
            "protocol": OPENCLI_WEB_PROTOCOL,
            "turn_id": "turn_" + hashlib.sha256(turn_material.encode("utf-8")).hexdigest()[:24],
            "role": "model_transport_only",
            "rules": [
                "Repository tools are executed by the external Open SWE runtime, not by ChatGPT Web.",
                "If a tool is needed, return one JSON object with type=tool_call, name, and arguments.",
                "If no tool is needed, return one JSON object with type=final and content.",
                "Do not claim that a tool ran unless a later tool message reports its result.",
            ],
            "messages": message_payloads,
            "tools": tool_payloads,
            "tool_choice": tool_choice or "auto",
        }
        return json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _extract_ask_result(stdout: str) -> tuple[str, str]:
        try:
            rows = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise OpenCLIWebModelError("OPENCLI_WEB_RESPONSE_INVALID") from exc
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
            raise OpenCLIWebModelError("OPENCLI_WEB_RESPONSE_INVALID")
        conversation_id = rows[0].get("conversationId")
        response = rows[0].get("response")
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise OpenCLIWebModelError("OPENCLI_WEB_CONVERSATION_ID_INVALID")
        if not isinstance(response, str):
            raise OpenCLIWebModelError("OPENCLI_WEB_RESPONSE_INVALID")
        return conversation_id.strip(), response.strip()

    @staticmethod
    def _extract_detail_response(stdout: str, expected_turn_id: str = "") -> str:
        try:
            rows = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise OpenCLIWebModelError("OPENCLI_WEB_RECONCILE_INVALID") from exc
        if not isinstance(rows, list) or not rows:
            raise OpenCLIWebModelError("OPENCLI_WEB_RECONCILE_INVALID")
        start = 0
        if expected_turn_id:
            matching_user_indexes = [
                index
                for index, row in enumerate(rows)
                if isinstance(row, Mapping)
                and row.get("Role") == "User"
                and expected_turn_id in str(row.get("Text") or "")
            ]
            if len(matching_user_indexes) != 1:
                raise OpenCLIWebModelError("OPENCLI_WEB_TURN_IDENTITY_UNKNOWN")
            start = matching_user_indexes[0] + 1
        for row in rows[start:]:
            if not isinstance(row, Mapping) or row.get("Role") != "Assistant":
                continue
            text = row.get("Text")
            generating = row.get("Generating")
            if isinstance(text, str) and text.strip() and generating is False:
                return text.strip()
        raise OpenCLIWebModelError("OPENCLI_WEB_RECONCILE_INCOMPLETE")

    @staticmethod
    def _extract_history_ids(stdout: str) -> list[str]:
        try:
            rows = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise OpenCLIWebModelError("OPENCLI_WEB_HISTORY_INVALID") from exc
        if not isinstance(rows, list):
            raise OpenCLIWebModelError("OPENCLI_WEB_HISTORY_INVALID")
        result: list[str] = []
        for row in rows[:12]:
            if isinstance(row, Mapping) and isinstance(row.get("Id"), str):
                result.append(str(row["Id"]))
        return result

    def _detail_response(self, conversation_id: str, *, wait: bool, turn_id: str = "") -> str:
        readback_timeout = max(self.timeout_seconds, 30) if wait else self.timeout_seconds
        detail = self._run(
            [
                self.executable,
                "chatgpt",
                "detail",
                conversation_id,
                "--wait",
                "true" if wait else "false",
                "--timeout",
                str(readback_timeout),
                "--stable",
                str(int(_POST_RESPONSE_SETTLE_SECONDS)),
                "--site-session",
                self.site_session,
                "-f",
                "json",
            ]
        )
        return self._extract_detail_response(detail, turn_id)

    def _session_pacing_state(self) -> _PacingState:
        if not self.profile:
            return self._pacing_state
        return _shared_pacing_state(self._pacing_key(), clock=self._clock)

    def _pacing_key(self) -> tuple[str, str, str]:
        return (self.executable, self.profile, self.site_session)

    def _begin_web_turn(self) -> tuple[_PacingState, tuple[str, str, str] | None]:
        pacing_key = self._pacing_key() if self.profile else None
        state = (
            _shared_pacing_state(pacing_key, borrow=True, clock=self._clock)
            if pacing_key is not None
            else self._pacing_state
        )
        owns_inflight = False
        try:
            with state.condition:
                while state.in_flight:
                    state.condition.wait()
                state.in_flight = True
                owns_inflight = True
                now = self._clock()
                eligible_at = now
                if state.last_send_started is not None:
                    eligible_at = max(
                        eligible_at,
                        state.last_send_started + _MIN_WEB_SEND_INTERVAL_SECONDS,
                    )
                if state.last_response_finished is not None:
                    eligible_at = max(
                        eligible_at,
                        state.last_response_finished + _POST_RESPONSE_SETTLE_SECONDS,
                    )
            if eligible_at > now:
                self._sleep(eligible_at - now)
            with state.condition:
                state.last_send_started = self._clock()
            return state, pacing_key
        except BaseException:
            with state.condition:
                if owns_inflight:
                    state.in_flight = False
                    state.condition.notify_all()
            if pacing_key is not None:
                _release_shared_pacing_state(pacing_key, state)
            raise

    def _finish_web_turn(
        self,
        state: _PacingState,
        pacing_key: tuple[str, str, str] | None,
        *,
        response_finished: bool,
    ) -> None:
        with state.condition:
            if response_finished:
                state.last_response_finished = self._clock()
            state.in_flight = False
            state.condition.notify_all()
        if pacing_key is not None:
            _release_shared_pacing_state(pacing_key, state)

    def _reconcile_timeout(self, turn_id: str) -> str:
        if self._conversation_id:
            return self._detail_response(self._conversation_id, wait=True, turn_id=turn_id)
        history = self._run(
            [
                self.executable,
                "chatgpt",
                "history",
                "--site-session",
                self.site_session,
                "-f",
                "json",
            ]
        )
        matches: list[str] = []
        for conversation_id in self._extract_history_ids(history):
            detail = self._run(
                [
                    self.executable,
                    "chatgpt",
                    "detail",
                    conversation_id,
                    "--wait",
                    "false",
                    "--site-session",
                    self.site_session,
                    "-f",
                    "json",
                ]
            )
            if turn_id in detail:
                matches.append(conversation_id)
        if len(matches) != 1:
            raise OpenCLIWebModelError("OPENCLI_WEB_TIMEOUT_RECONCILE_UNKNOWN")
        self._conversation_id = matches[0]
        return self._detail_response(matches[0], wait=True, turn_id=turn_id)

    def _reserve_web_turn(self) -> None:
        with self._budget_lock:
            if self._web_turn_count >= _MAX_WEB_TURNS_PER_OPERATION:
                raise OpenCLIWebModelError("OPENCLI_WEB_TURN_BUDGET_EXHAUSTED")
            self._web_turn_count += 1

    def _send_and_reconcile(self, prompt: str, *, budget_reserved: bool = False) -> str:
        if not budget_reserved:
            self._reserve_web_turn()
        if self._durable_pacing_backend is not None:
            return self._durable_send_and_reconcile(prompt)
        return self._inprocess_send_and_reconcile(prompt)

    def _inprocess_send_and_reconcile(self, prompt: str) -> str:
        pacing_state, pacing_key = self._begin_web_turn()
        response_ref = [False]
        try:
            return self._execute_web_send(prompt, response_finished_ref=response_ref)
        finally:
            self._finish_web_turn(
                pacing_state,
                pacing_key,
                response_finished=response_ref[0],
            )

    def _execute_web_send(self, prompt: str, *, response_finished_ref: list[bool] | None) -> str:
        try:
            prompt_envelope = json.loads(prompt)
        except json.JSONDecodeError:
            prompt_envelope = {}
        turn_id = (
            str(prompt_envelope.get("turn_id") or "")
            if isinstance(prompt_envelope, Mapping)
            else ""
        )
        argv = [self.executable, "chatgpt", "ask", prompt]
        if self._conversation_id:
            argv.extend(["--conversation", self._conversation_id])
        else:
            argv.append("--new")
        argv.extend(
            [
                "--wait",
                "true",
                "--timeout",
                str(self.timeout_seconds),
                "--site-session",
                self.site_session,
                "-f",
                "json",
            ]
        )
        try:
            stdout = self._run(argv)
        except OpenCLIWebModelError as exc:
            if str(exc) == "OPENCLI_WEB_BUSY":
                self._cooldown_and_probe_status()
            if str(exc) != "OPENCLI_WEB_TIMEOUT" or not turn_id:
                raise
            if response_finished_ref is not None:
                response_finished_ref[0] = True
            return self._reconcile_timeout(turn_id)
        conversation_id, _immediate_response = self._extract_ask_result(stdout)
        if self._conversation_id and conversation_id != self._conversation_id:
            raise OpenCLIWebModelError("OPENCLI_WEB_CONVERSATION_ID_MISMATCH")
        self._conversation_id = conversation_id
        response = self._detail_response(conversation_id, wait=True, turn_id=turn_id)
        if response_finished_ref is not None:
            response_finished_ref[0] = True
        return response

    def _durable_send_and_reconcile(self, prompt: str) -> str:
        backend = self._durable_pacing_backend
        if backend is None:
            raise OpenCLIWebModelError("OPENCLI_WEB_DURABLE_BACKEND_UNAVAILABLE")
        response_ref = [False]
        with backend.acquire_send_lock() as durable_state:
            now = self._clock()
            eligible_at = now
            if durable_state.key and durable_state.last_send_started >= 0:
                eligible_at = max(
                    eligible_at,
                    durable_state.last_send_started + _DURABLE_MIN_SEND_INTERVAL,
                )
            if durable_state.key and durable_state.last_response_finished >= 0:
                eligible_at = max(
                    eligible_at,
                    durable_state.last_response_finished + _DURABLE_MIN_RESPONSE_SETTLE,
                )
            if eligible_at > now:
                self._sleep(eligible_at - now)
            send_time = self._clock()
            new_state = DurablePacingState(
                key=backend._key,
                last_send_started=send_time,
                last_response_finished=durable_state.last_response_finished,
            )
            _write_durable_state(backend.state_path(), new_state)
            response = self._execute_web_send(prompt, response_finished_ref=response_ref)
            if response_ref[0]:
                finish_state = DurablePacingState(
                    key=new_state.key,
                    last_send_started=new_state.last_send_started,
                    last_response_finished=self._clock(),
                )
                _write_durable_state(backend.state_path(), finish_state)
        return response

    @staticmethod
    def _is_complete_protocol_response(response: str) -> bool:
        try:
            envelope = json.loads(response)
        except json.JSONDecodeError:
            return False
        if not isinstance(envelope, Mapping):
            return False
        kind = envelope.get("type")
        if kind == "final":
            return isinstance(envelope.get("content"), str)
        if kind == "tool_call":
            return isinstance(envelope.get("name"), str) and isinstance(
                envelope.get("arguments"), Mapping
            )
        return False

    def _refresh_protocol_response(self, response: str, *, turn_id: str) -> str:
        if self._is_complete_protocol_response(response) or not self._conversation_id:
            return response
        latest = response
        for _ in range(2):
            latest = self._detail_response(
                self._conversation_id,
                wait=True,
                turn_id=turn_id,
            )
            if self._is_complete_protocol_response(latest):
                return latest
        return latest

    def _repair_protocol_response(self, invalid_response: str) -> str:
        if not self._conversation_id:
            raise OpenCLIWebModelError("OPENCLI_WEB_CONVERSATION_ID_INVALID")
        repair_turn_id = (
            "turn_repair_"
            + hashlib.sha256(
                f"{self._conversation_id}\0{invalid_response}".encode("utf-8")
            ).hexdigest()[:20]
        )
        repair_prompt = json.dumps(
            {
                "protocol": OPENCLI_WEB_PROTOCOL,
                "turn_id": repair_turn_id,
                "instruction": (
                    "Your immediately previous response was incomplete or invalid for the declared JSON "
                    "protocol. Repeat the same intended response as exactly one complete JSON object only. "
                    "Do not execute any tool, do not change the intended tool name or arguments, and do not "
                    "add prose or markdown."
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        response = self._send_and_reconcile(repair_prompt)
        if not self._is_complete_protocol_response(response):
            raise OpenCLIWebModelError("OPENCLI_WEB_PROTOCOL_RESPONSE_INVALID")
        return response

    @staticmethod
    def _response_message(response: str, tools: Sequence[Mapping[str, Any]]) -> AIMessage:
        allowed = {name for tool in tools if (name := _tool_name(tool))}
        try:
            envelope = json.loads(response)
        except json.JSONDecodeError:
            return AIMessage(content=response)
        if not isinstance(envelope, Mapping):
            return AIMessage(content=response)
        kind = envelope.get("type")
        if kind == "final" and isinstance(envelope.get("content"), str):
            return AIMessage(content=str(envelope["content"]))
        if kind != "tool_call":
            return AIMessage(content=response)
        name = envelope.get("name")
        arguments = envelope.get("arguments")
        if not isinstance(name, str) or name not in allowed or not isinstance(arguments, Mapping):
            raise OpenCLIWebModelError("OPENCLI_WEB_TOOL_CALL_INVALID")
        args = dict(arguments)
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": _tool_call_id(name, args, response),
                    "type": "tool_call",
                }
            ],
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
        tools = kwargs.get("tools") or []
        if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
            raise OpenCLIWebModelError("OPENCLI_WEB_TOOLS_INVALID")
        normalized_tools = [dict(tool) for tool in tools if isinstance(tool, Mapping)]
        if len(normalized_tools) != len(tools):
            raise OpenCLIWebModelError("OPENCLI_WEB_TOOLS_INVALID")
        tool_choice = kwargs.get("tool_choice")
        if tool_choice is not None and not isinstance(tool_choice, str):
            raise OpenCLIWebModelError("OPENCLI_WEB_TOOL_CHOICE_INVALID")

        self._reserve_web_turn()
        self._select_intelligence_level()
        prompt = self._render_prompt(messages, normalized_tools, tool_choice)
        turn_id = str(json.loads(prompt)["turn_id"])
        response = self._send_and_reconcile(prompt, budget_reserved=True)
        response = self._refresh_protocol_response(response, turn_id=turn_id)
        if not self._is_complete_protocol_response(response):
            response = self._repair_protocol_response(response)
        message = self._response_message(response, normalized_tools)
        return ChatResult(generations=[ChatGeneration(message=message)])
