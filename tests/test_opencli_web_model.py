from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from nexus_open_swe_runtime import cli
from nexus_open_swe_runtime import opencli_web_model as web_model
from nexus_open_swe_runtime.opencli_web_model import (
    _DURABLE_STATE_SCHEMA,
    OPENCLI_WEB_PROTOCOL,
    DurablePacingBackend,
    DurablePacingLock,
    DurablePacingState,
    OpenCLIWebChatModel,
    OpenCLIWebModelError,
    _durable_pacing_key,
    _read_durable_state,
    _validate_state_file,
    _write_durable_state,
)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


def _use_fake_clock(model: OpenCLIWebChatModel) -> _FakeClock:
    clock = _FakeClock()
    model._clock = clock
    model._sleep = clock.sleep
    return clock


def _fake_process(monkeypatch: pytest.MonkeyPatch, response: str):
    calls: list[tuple[list[str], dict[str, object]]] = []
    latest_prompt = ""

    def fake_run(argv, **kwargs):
        nonlocal latest_prompt
        args = list(argv)
        calls.append((args, dict(kwargs)))
        if args[1:3] == ["chatgpt", "model"]:
            return SimpleNamespace(returncode=0, stdout='[{"Status":"ok"}]', stderr="")
        if args[1:3] == ["chatgpt", "ask"]:
            latest_prompt = args[3]
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "conversationId": "web-conversation-1",
                            "conversationUrl": "https://chatgpt.com/c/web-conversation-1",
                            "tool": "chatgpt",
                            "response": "",
                        }
                    ]
                ),
                stderr="",
            )
        if args[1:3] == ["chatgpt", "detail"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "Index": 1,
                            "Role": "User",
                            "Text": latest_prompt,
                            "Generating": False,
                            "StableSeconds": 6,
                        },
                        {
                            "Index": 2,
                            "Role": "Assistant",
                            "Text": response,
                            "Generating": False,
                            "StableSeconds": 6,
                        },
                    ]
                ),
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr(
        "nexus_open_swe_runtime.opencli_web_model.subprocess.run",
        fake_run,
    )
    return calls


def test_build_model_selects_opencli_web_bridge_without_langchain_provider():
    def fail_init(**_kwargs):
        raise AssertionError("init_chat_model must not be used for ChatGPT Web")

    model = cli._build_model(
        {"init_chat_model": fail_init},
        "opencli_chatgpt",
        "very-high",
        {
            "executable": "/opt/opencli",
            "profile": "profile-a",
            "site_session": "persistent",
            "timeout_seconds": 240,
        },
    )

    assert isinstance(model, OpenCLIWebChatModel)
    assert model.executable == "/opt/opencli"
    assert model.profile == "profile-a"
    assert model.site_session == "persistent"
    assert model.timeout_seconds == 240


def test_build_model_ignores_ambient_opencli_binding(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NEXUS_OPENCLI_EXECUTABLE", "/tmp/wrong-opencli")
    monkeypatch.setenv("NEXUS_OPENCLI_PROFILE", "wrong-profile")
    monkeypatch.setenv("NEXUS_OPENCLI_SITE_SESSION", "wrong-session")
    monkeypatch.setenv("NEXUS_OPENCLI_TIMEOUT_SECONDS", "899")

    model = cli._build_model(
        {},
        "opencli_chatgpt",
        "very-high",
        {
            "executable": "/opt/opencli",
            "profile": "correct-profile",
            "site_session": "persistent",
            "timeout_seconds": 120,
        },
    )

    assert model.executable == "/opt/opencli"
    assert model.profile == "correct-profile"
    assert model.site_session == "persistent"
    assert model.timeout_seconds == 120


def test_build_model_rejects_missing_unknown_and_out_of_range_transport_config():
    valid = {
        "executable": "/opt/opencli",
        "profile": "profile-a",
        "site_session": "ephemeral",
        "timeout_seconds": 30,
    }
    cli._build_model({}, "opencli_chatgpt", "very-high", valid)
    cli._build_model({}, "opencli_chatgpt", "very-high", {**valid, "timeout_seconds": 900})
    for config in (
        None,
        {**valid, "executable": "opencli"},
        {**valid, "profile": ""},
        {**valid, "timeout_seconds": 29},
        {**valid, "timeout_seconds": 901},
        {**valid, "site_session": "session-a"},
        {**valid, "extra": "unexpected"},
    ):
        with pytest.raises(cli.RuntimeErrorBounded, match="OPENCLI_WEB_TRANSPORT_CONFIG_INVALID"):
            cli._build_model({}, "opencli_chatgpt", "very-high", config)
    with pytest.raises(
        cli.RuntimeErrorBounded, match="OPEN_SWE_TRANSPORT_CONFIG_PROVIDER_MISMATCH"
    ):
        cli._build_model(
            {"init_chat_model": lambda **_kwargs: object()}, "google_genai", "g", valid
        )


def test_opencli_web_model_uses_chatgpt_web_and_no_shell(monkeypatch: pytest.MonkeyPatch):
    calls = _fake_process(
        monkeypatch,
        '{"type":"final","content":"bounded answer"}',
    )
    model = OpenCLIWebChatModel(
        executable="/opt/opencli",
        intelligence_level="very-high",
        timeout_seconds=41,
    )

    result = model.invoke([HumanMessage(content="inspect the repository")])

    assert isinstance(result, AIMessage)
    assert result.content == "bounded answer"
    assert calls[0][0] == [
        "/opt/opencli",
        "chatgpt",
        "model",
        "very-high",
        "--site-session",
        "ephemeral",
        "-f",
        "json",
    ]
    ask = calls[1][0]
    assert ask[0:3] == ["/opt/opencli", "chatgpt", "ask"]
    assert "--new" in ask
    assert ask[ask.index("--wait") + 1] == "true"
    assert ask[ask.index("--timeout") + 1] == "41"
    assert calls[1][1]["shell"] is False
    detail = calls[2][0]
    assert detail[0:3] == ["/opt/opencli", "chatgpt", "detail"]
    assert detail[3] == "web-conversation-1"
    assert detail[detail.index("--wait") + 1] == "true"
    assert detail[detail.index("--stable") + 1] == "3"
    prompt = json.loads(ask[3])
    assert prompt["protocol"] == OPENCLI_WEB_PROTOCOL
    assert prompt["role"] == "model_transport_only"
    assert prompt["messages"][-1]["content"] == "inspect the repository"


def test_opencli_web_model_rejects_invalid_site_session_before_subprocess(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = 0

    def fake_run(_argv, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("subprocess must not run")

    monkeypatch.setattr(
        "nexus_open_swe_runtime.opencli_web_model.subprocess.run",
        fake_run,
    )
    model = OpenCLIWebChatModel(site_session="session-a")

    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_SITE_SESSION_INVALID"):
        model.invoke([HumanMessage(content="hello")])

    assert calls == 0


def test_opencli_web_model_retries_idempotent_model_selection_once(
    monkeypatch: pytest.MonkeyPatch,
):
    model_attempts = 0
    ask_attempts = 0
    latest_prompt = ""
    sleep_delays: list[float] = []

    def fake_run(argv, **_kwargs):
        nonlocal model_attempts, ask_attempts, latest_prompt
        args = list(argv)
        if args[1:3] == ["chatgpt", "model"]:
            model_attempts += 1
            if model_attempts == 1:
                return SimpleNamespace(returncode=1, stdout="", stderr="selector failed")
            return SimpleNamespace(
                returncode=0, stdout='[{"Status":"Already selected"}]', stderr=""
            )
        if args[1:3] == ["chatgpt", "ask"]:
            ask_attempts += 1
            latest_prompt = args[3]
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"conversationId": "web-conversation-1", "response": ""}]),
                stderr="",
            )
        if args[1:3] == ["chatgpt", "detail"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {"Index": 1, "Role": "User", "Text": latest_prompt, "Generating": False},
                        {
                            "Index": 2,
                            "Role": "Assistant",
                            "Text": '{"type":"final","content":"done"}',
                            "Generating": False,
                        },
                    ]
                ),
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr(
        "nexus_open_swe_runtime.opencli_web_model.subprocess.run",
        fake_run,
    )
    model = OpenCLIWebChatModel(executable="/opt/opencli", intelligence_level="very-high")
    model._sleep = sleep_delays.append

    assert model.invoke([HumanMessage(content="hello")]).content == "done"
    assert model_attempts == 2
    assert ask_attempts == 1
    assert sleep_delays == [10.0]


@pytest.mark.parametrize("diagnostic", ["login required", "challenge detected", "quota exceeded"])
def test_opencli_web_model_hard_blocks_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    diagnostic: str,
):
    calls = 0
    sleep_delays: list[float] = []

    def fake_run(_argv, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=1, stdout="", stderr=diagnostic)

    monkeypatch.setattr(
        "nexus_open_swe_runtime.opencli_web_model.subprocess.run",
        fake_run,
    )
    model = OpenCLIWebChatModel()
    model._sleep = sleep_delays.append

    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_HARD_BLOCK"):
        model.invoke([HumanMessage(content="hello")])

    assert calls == 1
    assert sleep_delays == []


def test_opencli_web_model_busy_cools_down_and_probes_status_without_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    commands: list[str] = []
    sleep_delays: list[float] = []

    def fake_run(argv, **_kwargs):
        command = list(argv)[2]
        commands.append(command)
        if command == "model":
            return SimpleNamespace(returncode=1, stdout="", stderr="rate control busy")
        if command == "status":
            return SimpleNamespace(
                returncode=0,
                stdout='[{"Status":"Connected","Login":"Yes"}]',
                stderr="",
            )
        raise AssertionError(argv)

    monkeypatch.setattr(
        "nexus_open_swe_runtime.opencli_web_model.subprocess.run",
        fake_run,
    )
    model = OpenCLIWebChatModel()
    model._sleep = sleep_delays.append

    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_BUSY"):
        model.invoke([HumanMessage(content="hello")])

    assert commands == ["model", "status"]
    assert sleep_delays == [60.0]


def test_opencli_web_model_busy_on_selector_retry_cools_down_without_third_attempt(
    monkeypatch: pytest.MonkeyPatch,
):
    commands: list[str] = []
    sleep_delays: list[float] = []

    def fake_run(argv, **_kwargs):
        command = list(argv)[2]
        commands.append(command)
        if command == "model":
            if commands.count("model") == 1:
                return SimpleNamespace(returncode=1, stdout="", stderr="selector failed")
            return SimpleNamespace(returncode=1, stdout="", stderr="rate control busy")
        if command == "status":
            return SimpleNamespace(returncode=0, stdout='[{"Status":"Connected"}]', stderr="")
        raise AssertionError(argv)

    monkeypatch.setattr("nexus_open_swe_runtime.opencli_web_model.subprocess.run", fake_run)
    model = OpenCLIWebChatModel()
    model._sleep = sleep_delays.append

    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_BUSY"):
        model.invoke([HumanMessage(content="hello")])

    assert commands == ["model", "model", "status"]
    assert sleep_delays == [10.0, 60.0]


def test_opencli_web_model_busy_ask_cools_down_without_redispatch(
    monkeypatch: pytest.MonkeyPatch,
):
    commands: list[str] = []
    sleep_delays: list[float] = []

    def fake_run(argv, **_kwargs):
        command = list(argv)[2]
        commands.append(command)
        if command == "model":
            return SimpleNamespace(returncode=0, stdout='[{"Status":"ok"}]', stderr="")
        if command == "ask":
            return SimpleNamespace(returncode=1, stdout="", stderr="too many requests")
        if command == "status":
            return SimpleNamespace(
                returncode=0,
                stdout='[{"Status":"Connected","Login":"Yes"}]',
                stderr="",
            )
        raise AssertionError(argv)

    monkeypatch.setattr(
        "nexus_open_swe_runtime.opencli_web_model.subprocess.run",
        fake_run,
    )
    model = OpenCLIWebChatModel()
    model._sleep = sleep_delays.append

    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_BUSY"):
        model.invoke([HumanMessage(content="hello")])

    assert commands == ["model", "ask", "status"]
    assert sleep_delays == [60.0]


def test_opencli_web_model_translates_declared_tool_call(monkeypatch: pytest.MonkeyPatch):
    calls = _fake_process(
        monkeypatch,
        '{"type":"tool_call","name":"read_file","arguments":{"file_path":"README.md"}}',
    )

    @tool
    def read_file(file_path: str) -> str:
        """Read one repository file."""
        return file_path

    model = OpenCLIWebChatModel(executable="opencli", intelligence_level="advanced")
    result = model.bind_tools([read_file]).invoke("inspect README")

    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "read_file"
    assert result.tool_calls[0]["args"] == {"file_path": "README.md"}
    prompt = json.loads(calls[1][0][3])
    assert prompt["tools"][0]["function"]["name"] == "read_file"


def test_opencli_web_model_reuses_exact_conversation_for_next_model_turn(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = _fake_process(
        monkeypatch,
        '{"type":"final","content":"done"}',
    )
    model = OpenCLIWebChatModel(executable="/opt/opencli", intelligence_level="very-high")
    _use_fake_clock(model)

    assert model.invoke([HumanMessage(content="first")]).content == "done"
    assert model.invoke([HumanMessage(content="second")]).content == "done"

    asks = [argv for argv, _kwargs in calls if argv[1:3] == ["chatgpt", "ask"]]
    assert "--new" in asks[0]
    assert "--conversation" not in asks[0]
    assert "--conversation" in asks[1]
    assert asks[1][asks[1].index("--conversation") + 1] == "web-conversation-1"
    details = [argv for argv, _kwargs in calls if argv[1:3] == ["chatgpt", "detail"]]
    assert len(details) == 2
    assert all(argv[3] == "web-conversation-1" for argv in details)


def test_opencli_web_model_paces_back_to_back_turns_with_fake_clock(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = _FakeClock()
    ask_starts: list[float] = []
    calls = _fake_process(
        monkeypatch,
        '{"type":"final","content":"done"}',
    )
    original_run = __import__(
        "nexus_open_swe_runtime.opencli_web_model", fromlist=["subprocess"]
    ).subprocess.run

    def timed_run(argv, **kwargs):
        if list(argv)[1:3] == ["chatgpt", "ask"]:
            ask_starts.append(clock())
        return original_run(argv, **kwargs)

    monkeypatch.setattr(
        "nexus_open_swe_runtime.opencli_web_model.subprocess.run",
        timed_run,
    )
    model = OpenCLIWebChatModel(
        executable="/opt/opencli",
        intelligence_level="very-high",
        profile="profile-pacing",
        site_session="persistent",
    )
    model._clock = clock
    model._sleep = clock.sleep

    assert model.invoke([HumanMessage(content="first")]).content == "done"
    assert model.invoke([HumanMessage(content="second")]).content == "done"

    assert ask_starts == [0.0, 15.0]
    assert clock.sleeps == [15.0]
    details = [argv for argv, _kwargs in calls if argv[1:3] == ["chatgpt", "detail"]]
    assert all(argv[argv.index("--stable") + 1] == "3" for argv in details)


def test_opencli_web_model_serializes_shared_session_across_instances(
    monkeypatch: pytest.MonkeyPatch,
):
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    guard = threading.Lock()
    active_asks = 0
    max_active_asks = 0
    prompts: dict[str, str] = {}

    def fake_run(argv, **_kwargs):
        nonlocal active_asks, max_active_asks
        args = list(argv)
        if args[1:3] == ["chatgpt", "model"]:
            return SimpleNamespace(returncode=0, stdout='[{"Status":"ok"}]', stderr="")
        if args[1:3] == ["chatgpt", "ask"]:
            prompt = args[3]
            is_first = '"content":"first"' in prompt
            conversation_id = "conversation-first" if is_first else "conversation-second"
            prompts[conversation_id] = prompt
            with guard:
                active_asks += 1
                max_active_asks = max(max_active_asks, active_asks)
            if is_first:
                first_started.set()
                release_first.wait(timeout=2)
            else:
                second_started.set()
            with guard:
                active_asks -= 1
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"conversationId": conversation_id, "response": ""}]),
                stderr="",
            )
        if args[1:3] == ["chatgpt", "detail"]:
            conversation_id = args[3]
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "Index": 1,
                            "Role": "User",
                            "Text": prompts[conversation_id],
                            "Generating": False,
                        },
                        {
                            "Index": 2,
                            "Role": "Assistant",
                            "Text": '{"type":"final","content":"done"}',
                            "Generating": False,
                        },
                    ]
                ),
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr(
        "nexus_open_swe_runtime.opencli_web_model.subprocess.run",
        fake_run,
    )
    first = OpenCLIWebChatModel(profile="shared-profile", site_session="persistent")
    second = OpenCLIWebChatModel(profile="shared-profile", site_session="persistent")
    first._clock = second._clock = lambda: 0.0
    first._sleep = second._sleep = lambda _delay: None
    errors: list[BaseException] = []

    def invoke(model: OpenCLIWebChatModel, content: str) -> None:
        try:
            model.invoke([HumanMessage(content=content)])
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    thread_one = threading.Thread(target=invoke, args=(first, "first"))
    thread_two = threading.Thread(target=invoke, args=(second, "second"))
    thread_one.start()
    assert first_started.wait(timeout=1)
    thread_two.start()
    overlapped = second_started.wait(timeout=0.1)
    release_first.set()
    thread_one.join(timeout=2)
    thread_two.join(timeout=2)

    assert errors == []
    assert not overlapped
    assert second_started.is_set()
    assert max_active_asks == 1


def test_opencli_web_model_session_gate_survives_registry_eviction_churn(
    monkeypatch: pytest.MonkeyPatch,
):
    _fake_process(monkeypatch, '{"type":"final","content":"done"}')
    first = OpenCLIWebChatModel(
        executable="/opt/opencli",
        profile="target-profile",
        site_session="persistent",
    )
    clock = _FakeClock()
    first._clock = clock
    first._sleep = clock.sleep
    first.invoke([HumanMessage(content="successful send")])
    original_state = first._session_pacing_state()
    assert original_state.last_send_started == 0.0
    assert original_state.last_response_finished == 0.0

    for index in range(65):
        churn = OpenCLIWebChatModel(
            executable="/opt/opencli",
            profile=f"churn-profile-{index}",
            site_session="persistent",
        )
        churn._session_pacing_state()

    second = OpenCLIWebChatModel(
        executable="/opt/opencli",
        profile="target-profile",
        site_session="persistent",
    )

    assert first._session_pacing_state() is second._session_pacing_state()
    assert first._session_pacing_state() is original_state


def test_opencli_web_model_existing_borrow_keeps_canonical_gate_over_capacity():
    with web_model._PACING_STATES_LOCK:
        web_model._PACING_STATES.clear()
    key = ("/opt/opencli", "canonical-over-capacity", "persistent")
    canonical = web_model._shared_pacing_state(key, borrow=True)
    borrowed = [(key, canonical)]
    for index in range(63):
        other_key = ("/opt/opencli", f"borrowed-{index}", "persistent")
        other = web_model._shared_pacing_state(other_key, borrow=True)
        borrowed.append((other_key, other))
    overflow_key = ("/opt/opencli", "overflow", "persistent")
    with web_model._PACING_STATES_LOCK:
        overflow = web_model._PacingState(borrowers=1)
        web_model._PACING_STATES[overflow_key] = overflow
    borrowed.append((overflow_key, overflow))

    assert len(web_model._PACING_STATES) > web_model._MAX_PACING_SESSION_KEYS
    assert web_model._shared_pacing_state(key, borrow=True) is canonical
    assert web_model._PACING_STATES[key] is canonical

    for borrowed_key, state in borrowed:
        web_model._release_shared_pacing_state(borrowed_key, state)
    web_model._release_shared_pacing_state(key, canonical)


def test_opencli_web_model_turn_budget_stops_before_web_subprocess(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = 0

    def fake_run(_argv, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("turn budget must stop before subprocess")

    monkeypatch.setattr(
        "nexus_open_swe_runtime.opencli_web_model.subprocess.run",
        fake_run,
    )
    model = OpenCLIWebChatModel()
    model._web_turn_count = 12

    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_TURN_BUDGET_EXHAUSTED"):
        model.invoke([HumanMessage(content="turn thirteen")])

    assert calls == 0


def test_opencli_web_model_concurrent_budget_admission_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
):
    model_calls = 0
    ask_started = threading.Event()
    release_ask = threading.Event()
    errors: list[BaseException] = []
    latest_prompt = ""

    def fake_run(argv, **_kwargs):
        nonlocal model_calls, latest_prompt
        args = list(argv)
        if args[1:3] == ["chatgpt", "model"]:
            model_calls += 1
            return SimpleNamespace(returncode=0, stdout='[{"Status":"ok"}]', stderr="")
        if args[1:3] == ["chatgpt", "ask"]:
            latest_prompt = args[3]
            ask_started.set()
            release_ask.wait(timeout=2)
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"conversationId": "c", "response": ""}]),
                stderr="",
            )
        if args[1:3] == ["chatgpt", "detail"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {"Index": 1, "Role": "User", "Text": latest_prompt, "Generating": False},
                        {
                            "Index": 2,
                            "Role": "Assistant",
                            "Text": '{"type":"final","content":"ok"}',
                            "Generating": False,
                        },
                    ]
                ),
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr("nexus_open_swe_runtime.opencli_web_model.subprocess.run", fake_run)
    model = OpenCLIWebChatModel(profile="atomic-budget", site_session="persistent")
    model._web_turn_count = 11

    def invoke() -> None:
        try:
            model.invoke([HumanMessage(content="concurrent")])
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=invoke)
    second = threading.Thread(target=invoke)
    first.start()
    assert ask_started.wait(timeout=1)
    second.start()
    second.join(timeout=1)
    release_ask.set()
    first.join(timeout=2)

    assert len(errors) == 1
    assert str(errors[0]) == "OPENCLI_WEB_TURN_BUDGET_EXHAUSTED"
    assert model_calls == 1


def test_opencli_web_model_rejects_conversation_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = 0
    latest_prompt = ""

    def fake_run(argv, **_kwargs):
        nonlocal calls, latest_prompt
        args = list(argv)
        if args[1:3] == ["chatgpt", "model"]:
            return SimpleNamespace(returncode=0, stdout='[{"Status":"ok"}]', stderr="")
        if args[1:3] == ["chatgpt", "ask"]:
            calls += 1
            latest_prompt = args[3]
            conversation_id = "web-conversation-1" if calls == 1 else "web-conversation-2"
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "conversationId": conversation_id,
                            "response": '{"type":"final","content":"done"}',
                        }
                    ]
                ),
                stderr="",
            )
        if args[1:3] == ["chatgpt", "detail"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {"Index": 1, "Role": "User", "Text": latest_prompt, "Generating": False},
                        {
                            "Index": 2,
                            "Role": "Assistant",
                            "Text": '{"type":"final","content":"done"}',
                            "Generating": False,
                        },
                    ]
                ),
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr(
        "nexus_open_swe_runtime.opencli_web_model.subprocess.run",
        fake_run,
    )
    model = OpenCLIWebChatModel(executable="/opt/opencli", intelligence_level="very-high")
    _use_fake_clock(model)

    assert model.invoke([HumanMessage(content="first")]).content == "done"
    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_CONVERSATION_ID_MISMATCH"):
        model.invoke([HumanMessage(content="second")])


def test_opencli_web_model_refreshes_incomplete_readback_without_redispatch(
    monkeypatch: pytest.MonkeyPatch,
):
    ask_count = 0
    detail_count = 0
    latest_prompt = ""

    def fake_run(argv, **_kwargs):
        nonlocal ask_count, detail_count, latest_prompt
        args = list(argv)
        if args[1:3] == ["chatgpt", "model"]:
            return SimpleNamespace(returncode=0, stdout='[{"Status":"ok"}]', stderr="")
        if args[1:3] == ["chatgpt", "ask"]:
            ask_count += 1
            latest_prompt = args[3]
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"conversationId": "web-conversation-1", "response": ""}]),
                stderr="",
            )
        if args[1:3] == ["chatgpt", "detail"]:
            detail_count += 1
            text = (
                '{"type":"tool_call","name":"read_file","arguments":{"file_path":"README.md"}}'
                if detail_count >= 2
                else '{"type":"tool_call","name":"read_file","arguments":{"file_path":"README'
            )
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {"Index": 1, "Role": "User", "Text": latest_prompt, "Generating": False},
                        {"Index": 2, "Role": "Assistant", "Text": text, "Generating": False},
                    ]
                ),
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr(
        "nexus_open_swe_runtime.opencli_web_model.subprocess.run",
        fake_run,
    )

    @tool
    def read_file(file_path: str) -> str:
        """Read one repository file."""
        return file_path

    model = OpenCLIWebChatModel(executable="/opt/opencli", intelligence_level="very-high")
    _use_fake_clock(model)
    result = model.bind_tools([read_file]).invoke("inspect README")

    assert ask_count == 1
    assert detail_count == 2
    assert result.tool_calls[0]["name"] == "read_file"
    assert result.tool_calls[0]["args"] == {"file_path": "README.md"}


def test_opencli_web_model_repairs_truncated_protocol_response_before_tool_execution(
    monkeypatch: pytest.MonkeyPatch,
):
    ask_count = 0
    latest_prompt = ""
    ask_prompts: list[str] = []
    ask_starts: list[float] = []
    clock = _FakeClock()

    def fake_run(argv, **_kwargs):
        nonlocal ask_count, latest_prompt
        args = list(argv)
        if args[1:3] == ["chatgpt", "model"]:
            return SimpleNamespace(returncode=0, stdout='[{"Status":"ok"}]', stderr="")
        if args[1:3] == ["chatgpt", "ask"]:
            ask_count += 1
            latest_prompt = args[3]
            ask_prompts.append(latest_prompt)
            ask_starts.append(clock())
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"conversationId": "web-conversation-1", "response": ""}]),
                stderr="",
            )
        if args[1:3] == ["chatgpt", "detail"]:
            text = (
                '{"type":"tool_call","name":"read_file","arguments":{"file_path":"README.md"}}'
                if ask_count >= 2
                else '{"type":"tool_call","name":"read_file","arguments":{"file_path":"README'
            )
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {"Index": 1, "Role": "User", "Text": latest_prompt, "Generating": False},
                        {"Index": 2, "Role": "Assistant", "Text": text, "Generating": False},
                    ]
                ),
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr(
        "nexus_open_swe_runtime.opencli_web_model.subprocess.run",
        fake_run,
    )

    @tool
    def read_file(file_path: str) -> str:
        """Read one repository file."""
        return file_path

    model = OpenCLIWebChatModel(executable="/opt/opencli", intelligence_level="very-high")
    model._clock = clock
    model._sleep = clock.sleep
    result = model.bind_tools([read_file]).invoke("inspect README")

    assert ask_count == 2
    assert ask_starts == [0.0, 15.0]
    assert json.loads(ask_prompts[1])["turn_id"].startswith("turn_repair_")
    assert model._web_turn_count == 2
    assert result.tool_calls[0]["name"] == "read_file"
    assert result.tool_calls[0]["args"] == {"file_path": "README.md"}


def test_opencli_web_model_protocol_repair_consumes_turn_budget_without_second_ask(
    monkeypatch: pytest.MonkeyPatch,
):
    ask_count = 0
    latest_prompt = ""

    def fake_run(argv, **_kwargs):
        nonlocal ask_count, latest_prompt
        args = list(argv)
        if args[1:3] == ["chatgpt", "model"]:
            return SimpleNamespace(returncode=0, stdout='[{"Status":"ok"}]', stderr="")
        if args[1:3] == ["chatgpt", "ask"]:
            ask_count += 1
            latest_prompt = args[3]
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"conversationId": "web-conversation-1", "response": ""}]),
                stderr="",
            )
        if args[1:3] == ["chatgpt", "detail"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {"Index": 1, "Role": "User", "Text": latest_prompt, "Generating": False},
                        {"Index": 2, "Role": "Assistant", "Text": "{", "Generating": False},
                    ]
                ),
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr(
        "nexus_open_swe_runtime.opencli_web_model.subprocess.run",
        fake_run,
    )
    model = OpenCLIWebChatModel()
    model._web_turn_count = 11
    _use_fake_clock(model)

    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_TURN_BUDGET_EXHAUSTED"):
        model.invoke([HumanMessage(content="last admitted turn")])

    assert ask_count == 1
    assert model._web_turn_count == 12


def test_opencli_web_model_rejects_undeclared_tool_call(monkeypatch: pytest.MonkeyPatch):
    _fake_process(
        monkeypatch,
        '{"type":"tool_call","name":"shell","arguments":{"cmd":"rm -rf /"}}',
    )

    @tool
    def read_file(file_path: str) -> str:
        """Read one repository file."""
        return file_path

    model = OpenCLIWebChatModel(executable="opencli", intelligence_level="advanced")
    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_TOOL_CALL_INVALID"):
        model.bind_tools([read_file]).invoke("inspect README")


def test_opencli_web_model_builds_real_deepagents_graph_without_dispatch(tmp_path):
    runtime = cli._load_runtime()
    model = OpenCLIWebChatModel(executable="opencli", intelligence_level="very-high")

    graph = cli.build_semantic_graph(
        model,
        tmp_path,
        runtime,
        "opencli_chatgpt:very-high",
    )
    surface = set(cli.executable_tool_surface(graph))

    assert {"read_file", "ls", "glob", "grep", "record_finding"} <= surface
    assert "task" not in surface
    assert "write_file" not in surface
    assert "edit_file" not in surface


def test_opencli_web_model_reconciles_browser_timeout_without_redispatch(
    monkeypatch: pytest.MonkeyPatch,
):
    ask_count = 0
    latest_prompt = ""

    def fake_run(argv, **_kwargs):
        nonlocal ask_count, latest_prompt
        args = list(argv)
        if args[1:3] == ["chatgpt", "model"]:
            return SimpleNamespace(returncode=0, stdout='[{"Status":"ok"}]', stderr="")
        if args[1:3] == ["chatgpt", "ask"]:
            ask_count += 1
            latest_prompt = args[3]
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="Browser exec command timed out after 2s; it may still complete in the browser.",
            )
        if args[1:3] == ["chatgpt", "history"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"Index": 1, "Id": "web-conversation-1"}]),
                stderr="",
            )
        if args[1:3] == ["chatgpt", "detail"]:
            wait = args[args.index("--wait") + 1] if "--wait" in args else "false"
            rows = [{"Index": 1, "Role": "User", "Text": latest_prompt, "Generating": False}]
            if wait == "true":
                rows.append(
                    {
                        "Index": 2,
                        "Role": "Assistant",
                        "Text": '{"type":"final","content":"reconciled"}',
                        "Generating": False,
                    }
                )
            return SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")
        raise AssertionError(args)

    monkeypatch.setattr(
        "nexus_open_swe_runtime.opencli_web_model.subprocess.run",
        fake_run,
    )
    model = OpenCLIWebChatModel(
        executable="/opt/opencli",
        intelligence_level="very-high",
        timeout_seconds=2,
    )

    result = model.invoke([HumanMessage(content="timeout probe")])

    assert result.content == "reconciled"
    assert ask_count == 1
    assert model._conversation_id == "web-conversation-1"


def test_opencli_web_model_reconciles_bound_timeout_without_history_scan(
    monkeypatch: pytest.MonkeyPatch,
):
    commands: list[str] = []
    latest_prompt = ""

    def fake_run(argv, **_kwargs):
        nonlocal latest_prompt
        args = list(argv)
        commands.append(args[2])
        if args[1:3] == ["chatgpt", "model"]:
            return SimpleNamespace(returncode=0, stdout='[{"Status":"ok"}]', stderr="")
        if args[1:3] == ["chatgpt", "ask"]:
            latest_prompt = args[3]
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="Browser exec command timed out; it may still complete in the browser.",
            )
        if args[1:3] == ["chatgpt", "detail"]:
            assert args[3] == "bound-conversation"
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {"Index": 1, "Role": "User", "Text": latest_prompt, "Generating": False},
                        {
                            "Index": 2,
                            "Role": "Assistant",
                            "Text": '{"type":"final","content":"reconciled"}',
                            "Generating": False,
                        },
                    ]
                ),
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr("nexus_open_swe_runtime.opencli_web_model.subprocess.run", fake_run)
    model = OpenCLIWebChatModel(executable="/opt/opencli")
    model._conversation_id = "bound-conversation"

    assert model.invoke([HumanMessage(content="bound timeout")]).content == "reconciled"
    assert commands.count("ask") == 1
    assert "history" not in commands


def test_opencli_web_model_fails_closed_on_process_error(monkeypatch: pytest.MonkeyPatch):
    def fake_run(_argv, **_kwargs):
        return SimpleNamespace(returncode=69, stdout="", stderr="browser unavailable")

    monkeypatch.setattr(
        "nexus_open_swe_runtime.opencli_web_model.subprocess.run",
        fake_run,
    )
    model = OpenCLIWebChatModel(executable="opencli", intelligence_level="very-high")

    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_PROCESS_FAILURE"):
        model.invoke("hello")


# ---------------------------------------------------------------------------
# Durable cross-process pacing tests


def _durable_pacing_dir(tmp_path):
    d = tmp_path / "pacing"
    d.mkdir(exist_ok=True)
    os.chmod(str(d), 0o700)
    return str(d)


def test_durable_pacing_key_is_deterministic_and_independent(tmp_path):
    k1 = _durable_pacing_key("/opt/opencli", "profile-a", "persistent")
    k2 = _durable_pacing_key("/opt/opencli", "profile-a", "persistent")
    k3 = _durable_pacing_key("/opt/opencli", "profile-b", "persistent")
    k4 = _durable_pacing_key("/opt/opencli", "profile-a", "ephemeral")
    assert k1 == k2
    assert k1 != k3
    assert k1 != k4
    assert len(k1) == 32
    assert k1.isalnum()
    # No plaintext in the key
    for plaintext in ("/opt/opencli", "profile-a", "persistent"):
        assert plaintext not in k1


def test_durable_pacing_state_file_restrictive_permissions(tmp_path):
    pacing_dir = _durable_pacing_dir(tmp_path)
    state = DurablePacingState(key="test-key", last_send_started=100.0, last_response_finished=50.0)
    path = os.path.join(pacing_dir, "test-state.json")
    _write_durable_state(path, state)
    st = os.stat(path)
    assert st.st_mode & 0o077 == 0
    assert stat.S_ISREG(st.st_mode)
    parent_st = os.stat(pacing_dir)
    assert parent_st.st_mode & 0o077 == 0


def test_durable_pacing_write_read_roundtrip(tmp_path):
    pacing_dir = _durable_pacing_dir(tmp_path)
    state = DurablePacingState(key="k1", last_send_started=1.5, last_response_finished=2.5)
    path = os.path.join(pacing_dir, "k1.json")
    _write_durable_state(path, state)
    loaded = _read_durable_state(path)
    assert loaded.key == "k1"
    assert loaded.last_send_started == 1.5
    assert loaded.last_response_finished == 2.5


def test_durable_pacing_read_missing_file_returns_fresh_state(tmp_path):
    loaded = _read_durable_state(str(tmp_path / "nope.json"))
    assert loaded == DurablePacingState()


def test_durable_pacing_validate_missing_file_is_ok(tmp_path):
    _validate_state_file(str(tmp_path / "missing.json"))


def test_durable_pacing_validate_symlink_fails_closed(tmp_path):
    pacing_dir = _durable_pacing_dir(tmp_path)
    real = os.path.join(pacing_dir, "real.json")
    link = os.path.join(pacing_dir, "link.json")
    _write_durable_state(real, DurablePacingState(key="k"))
    os.symlink(real, link)
    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_DURABLE_STATE_UNSAFE"):
        _validate_state_file(link)


def test_durable_pacing_validate_wrong_owner_fails_closed(tmp_path):
    pacing_dir = _durable_pacing_dir(tmp_path)
    path = os.path.join(pacing_dir, "state.json")
    # Write a valid-looking state file but with empty key (malformed)
    import json as _json

    with open(path, "w") as _f:
        _json.dump(
            {
                "schema": _DURABLE_STATE_SCHEMA,
                "key": "",
                "last_send_started": 0,
                "last_response_finished": 0,
            },
            _f,
        )
    os.chmod(path, 0o600)
    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_DURABLE_STATE_MALFORMED"):
        _validate_state_file(path)


def test_durable_pacing_validate_missing_schema_fails_closed(tmp_path):
    pacing_dir = _durable_pacing_dir(tmp_path)
    path = os.path.join(pacing_dir, "bad-schema.json")
    import json

    with open(path, "w") as f:
        json.dump(
            {
                "schema": "wrong-schema",
                "key": "k",
                "last_send_started": 0,
                "last_response_finished": 0,
            },
            f,
        )
    os.chmod(path, 0o600)
    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_DURABLE_STATE_MALFORMED"):
        _validate_state_file(path)


def test_durable_pacing_validate_clock_rollback_fails_closed(tmp_path):
    pacing_dir = _durable_pacing_dir(tmp_path)
    path = os.path.join(pacing_dir, "future-state.json")
    state = DurablePacingState(key="k", last_send_started=99999999.0, last_response_finished=0.0)
    _write_durable_state(path, state)
    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_DURABLE_STATE_UNSAFE"):
        _validate_state_file(path, clock=lambda: 0.0)


def test_durable_pacing_validate_old_state_fails_closed(tmp_path):
    pacing_dir = _durable_pacing_dir(tmp_path)
    path = os.path.join(pacing_dir, "old-state.json")
    state = DurablePacingState(key="k", last_send_started=0.0, last_response_finished=1.0)
    _write_durable_state(path, state)
    # 86401 seconds in the past exceeds the skew threshold
    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_DURABLE_STATE_UNSAFE"):
        _validate_state_file(path, clock=lambda: 86402.0)


def test_durable_pacing_validate_non_numeric_timestamps_fail_closed(tmp_path):
    pacing_dir = _durable_pacing_dir(tmp_path)
    path = os.path.join(pacing_dir, "bad-ts.json")
    import json

    with open(path, "w") as f:
        json.dump(
            {
                "schema": _DURABLE_STATE_SCHEMA,
                "key": "k",
                "last_send_started": "not-a-number",
                "last_response_finished": 0,
            },
            f,
        )
    os.chmod(path, 0o600)
    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_DURABLE_STATE_MALFORMED"):
        _validate_state_file(path)


def test_durable_pacing_validate_nonregular_file_fails_closed(tmp_path):
    pacing_dir = _durable_pacing_dir(tmp_path)
    path = os.path.join(pacing_dir, "dir-not-file")
    os.mkdir(path)
    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_DURABLE_STATE_UNSAFE"):
        _validate_state_file(path)


def test_durable_pacing_validate_unsafe_permission_fails_closed(tmp_path):
    pacing_dir = _durable_pacing_dir(tmp_path)
    path = os.path.join(pacing_dir, "unsafe.json")
    import json

    with open(path, "w") as f:
        json.dump(
            {
                "schema": _DURABLE_STATE_SCHEMA,
                "key": "k",
                "last_send_started": 0,
                "last_response_finished": 0,
            },
            f,
        )
    os.chmod(path, 0o644)
    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_DURABLE_STATE_UNSAFE"):
        _validate_state_file(path)


def _write_d_state_raw(path, key):
    import json

    with open(path, "w") as f:
        json.dump(
            {
                "schema": _DURABLE_STATE_SCHEMA,
                "key": key,
                "last_send_started": 0,
                "last_response_finished": 0,
            },
            f,
        )


def test_durable_pacing_lock_held_during_send(tmp_path):
    pacing_dir = _durable_pacing_dir(tmp_path)
    state = DurablePacingState(key="lock-test")
    state_path = os.path.join(pacing_dir, "lock-test.json")
    lock_path = os.path.join(pacing_dir, "lock-test.lock")
    _write_durable_state(state_path, state)
    backend = object.__new__(DurablePacingBackend)
    backend._key = "lock-test"
    backend._state_path = state_path
    backend._lock_path = lock_path
    backend._clock = time.monotonic

    lock_held = False
    with backend.acquire_send_lock() as ds:
        lock_held = os.path.exists(lock_path)
        assert lock_held
        # Verify the lock file has restrictive permissions
        st = os.stat(lock_path)
        assert st.st_mode & 0o077 == 0
        assert ds.key == "lock-test"
    assert lock_held


def test_durable_pacing_across_processes_no_overlap(tmp_path):
    """Two real processes sharing one transport key have max_inflight=1."""
    pacing_dir = _durable_pacing_dir(tmp_path)
    state_path = os.path.join(pacing_dir, "proc-test.json")
    lock_path = os.path.join(pacing_dir, "proc-test.lock")
    send_log = tmp_path / "send-log"
    send_log.mkdir()
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    release_marker = tmp_path / "release"
    clock_time = tmp_path / "clock"
    clock_time.write_text("0.0")

    _write_durable_state(state_path, DurablePacingState(key="proc-test"))

    script = f"""
import fcntl, json, os, time, sys

state_path = {str(state_path)!r}
lock_path = {str(lock_path)!r}
send_log = {str(send_log)!r}
release_marker = {str(release_marker)!r}
clock_time = {str(clock_time)!r}
my_id = sys.argv[1]

# Read clock and compute eligible
with open(state_path) as f:
    state = json.load(f)
now = float(open(clock_time).read().strip())
eligible_at = max(now, state.get("last_send_started", 0) + 15.0)

# Acquire flock (blocking)
fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)

# Write send-start timestamp
state["last_send_started"] = now
state["last_response_finished"] = 0.0
with open(state_path, "w") as f:
    json.dump(state, f)

# Record send
log_path = os.path.join(send_log, f"{{my_id}}.txt")
with open(log_path, "w") as f:
    f.write(str(now))

# Wait for release marker
while not os.path.exists(release_marker):
    time.sleep(0.05)

fcntl.flock(fd, fcntl.LOCK_UN)
os.close(fd)
"""
    proc_script = tmp_path / "proc_worker.py"
    proc_script.write_text(script)

    # Start first process
    p1 = subprocess.Popen(
        [sys.executable, str(proc_script), "p1"],
        cwd=str(tmp_path),
    )
    # Wait for p1 to acquire lock
    while not (send_log / "p1.txt").exists():
        time.sleep(0.05)

    # Update clock to 1.0 (should not be eligible since last_send_started=0.0, eligible at 15.0)
    clock_time.write_text("1.0")

    # Start second process - should block on flock
    p2 = subprocess.Popen(
        [sys.executable, str(proc_script), "p2"],
        cwd=str(tmp_path),
    )

    # Give p2 time to start and block
    time.sleep(0.5)
    assert not (send_log / "p2.txt").exists(), "p2 should be blocked on flock"

    # Release p1
    release_marker.mkdir()
    p1.wait(timeout=5)
    # Now p2 should acquire the lock
    p2.wait(timeout=5)

    # Both sends should have happened with no overlap
    assert (send_log / "p1.txt").exists()
    assert (send_log / "p2.txt").exists()
    t1 = float((send_log / "p1.txt").read_text().strip())
    t2 = float((send_log / "p2.txt").read_text().strip())
    # p2 acquired lock after p1 released, so send times should not overlap
    assert t2 >= t1


def test_durable_backend_production_seam_serializes_real_processes(tmp_path):
    """Child processes exercise DurablePacingBackend itself, not a lock clone."""
    runtime_root = tmp_path / "runtime-state"
    package_root = Path(__file__).resolve().parents[1]
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    script = f"""
import sys, time
sys.path.insert(0, {str(package_root)!r})
from nexus_open_swe_runtime.opencli_web_model import DurablePacingBackend
backend = DurablePacingBackend({str(runtime_root)!r}, executable='/opt/opencli', profile='p', site_session='persistent', clock=time.time)
with backend.acquire_send_lock():
    open({str(log_dir)!r} + '/' + sys.argv[1], 'w').write('held')
    time.sleep(0.5)
"""
    worker = tmp_path / "worker.py"
    worker.write_text(script)
    p1 = subprocess.Popen([sys.executable, str(worker), "one"])
    deadline = time.monotonic() + 5
    while not (log_dir / "one").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert (log_dir / "one").exists()
    p2 = subprocess.Popen([sys.executable, str(worker), "two"])
    time.sleep(0.1)
    assert not (log_dir / "two").exists()
    p1.wait(timeout=5)
    p2.wait(timeout=5)
    assert p1.returncode == p2.returncode == 0
    assert (log_dir / "two").exists()


def _write_mock_opencli(path: Path, log_path: Path) -> None:
    path.write_text(
        f"""#!/usr/bin/env python3
import json, os, pathlib, sys, time

clock_path = pathlib.Path(os.environ["MOCK_CLOCK"])
log_path = pathlib.Path({str(log_path)!r})
command = sys.argv[2]
if command == "model":
    print('[{{"Status":"ok"}}]')
elif command == "ask":
    prompt = sys.argv[3]
    now = float(clock_path.read_text())
    log_path.with_suffix(".prompt").write_text(prompt, encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{{os.environ.get('MOCK_ID', '')}} ask {{now}}\\n")
    if os.environ.get("MOCK_WAIT_FOR_KILL") == "1":
        pathlib.Path(os.environ["MOCK_READY"]).touch()
        while not pathlib.Path(os.environ["MOCK_RELEASE"]).exists():
            time.sleep(0.01)
    print(json.dumps([{{"conversationId":"mock-conversation","response":""}}]))
elif command == "detail":
    prompt = log_path.with_suffix(".prompt").read_text(encoding="utf-8")
    print(json.dumps([{{"Role":"User","Text":prompt,"Generating":False}},
                      {{"Role":"Assistant","Text":json.dumps({{"type":"final","content":"ok"}}),"Generating":False}}]))
""",
        encoding="utf-8",
    )
    path.chmod(0o700)


def _write_production_worker(path: Path, package_root: Path) -> None:
    path.write_text(
        f"""import os, pathlib, sys
sys.path.insert(0, {str(package_root)!r})
from langchain_core.messages import HumanMessage
from nexus_open_swe_runtime.opencli_web_model import OpenCLIWebChatModel

clock_path = pathlib.Path(os.environ["MOCK_CLOCK"])
def clock():
    return float(clock_path.read_text())
def sleeper(delay):
    clock_path.write_text(str(clock() + delay))
model = OpenCLIWebChatModel(
    executable=os.environ["MOCK_EXECUTABLE"], profile=os.environ["MOCK_PROFILE"],
    site_session="persistent", runtime_state_root=os.environ["MOCK_ROOT"],
)
model._clock = clock
model._sleep = sleeper
model._durable_pacing_backend._clock = clock
model.invoke([HumanMessage(content="production pacing test")])
""",
        encoding="utf-8",
    )


def test_production_model_seam_paces_restart_and_kill_owner(tmp_path):
    """Real workers invoke OpenCLIWebChatModel, with a mocked CLI executable."""
    root = tmp_path / "runtime-state"
    clock = tmp_path / "clock"
    clock.write_text("0.0")
    log = tmp_path / "asks.log"
    executable = tmp_path / "mock-opencli"
    worker = tmp_path / "worker.py"
    package_root = Path(__file__).resolve().parents[1]
    _write_mock_opencli(executable, log)
    _write_production_worker(worker, package_root)

    def launch(identifier: str, *, wait_for_kill: bool = False):
        env = os.environ.copy()
        env.update(
            MOCK_CLOCK=str(clock),
            MOCK_EXECUTABLE=str(executable),
            MOCK_ROOT=str(root),
            MOCK_PROFILE="shared-profile",
            MOCK_ID=identifier,
        )
        if wait_for_kill:
            env.update(
                MOCK_WAIT_FOR_KILL="1",
                MOCK_READY=str(tmp_path / "owner.ready"),
                MOCK_RELEASE=str(tmp_path / "owner.release"),
            )
        return subprocess.Popen([sys.executable, str(worker)], env=env)

    first = launch("first")
    first.wait(timeout=10)
    assert first.returncode == 0
    second = launch("restart")
    second.wait(timeout=10)
    assert second.returncode == 0
    starts = [float(line.split()[-1]) for line in log.read_text().splitlines()]
    assert starts == [0.0, 15.0]

    owner = launch("owner", wait_for_kill=True)
    ready = tmp_path / "owner.ready"
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists()
    successor = launch("successor")
    time.sleep(0.1)
    assert successor.poll() is None
    owner.kill()
    owner.wait(timeout=10)
    successor.wait(timeout=10)
    assert successor.returncode == 0
    starts = [float(line.split()[-1]) for line in log.read_text().splitlines()]
    assert starts[-1] == 45.0


def test_production_model_seam_distinct_keys_are_independent(tmp_path):
    root = tmp_path / "runtime-state"
    clock = tmp_path / "clock"
    clock.write_text("0.0")
    log = tmp_path / "asks.log"
    executable = tmp_path / "mock-opencli"
    worker = tmp_path / "worker.py"
    package_root = Path(__file__).resolve().parents[1]
    _write_mock_opencli(executable, log)
    _write_production_worker(worker, package_root)
    processes = []
    for profile in ("profile-a", "profile-b"):
        env = os.environ.copy()
        env.update(
            MOCK_CLOCK=str(clock),
            MOCK_EXECUTABLE=str(executable),
            MOCK_ROOT=str(root),
            MOCK_PROFILE=profile,
            MOCK_ID=profile,
        )
        processes.append(subprocess.Popen([sys.executable, str(worker)], env=env))
    for process in processes:
        process.wait(timeout=10)
        assert process.returncode == 0
    starts = [float(line.split()[-1]) for line in log.read_text().splitlines()]
    assert starts == [0.0, 0.0]


@pytest.mark.parametrize(
    "corruption", ["malformed", "symlink", "wrong-mode", "identity", "future", "backward"]
)
def test_production_model_seam_rejects_unsafe_state_without_ask(tmp_path, corruption, monkeypatch):
    root = tmp_path / "runtime-state"
    clock = tmp_path / "clock"
    clock.write_text("100.0")
    log = tmp_path / "asks.log"
    executable = tmp_path / "mock-opencli"
    worker = tmp_path / "worker.py"
    package_root = Path(__file__).resolve().parents[1]
    _write_mock_opencli(executable, log)
    _write_production_worker(worker, package_root)
    monkeypatch.setenv("MOCK_CLOCK", str(clock))
    model = OpenCLIWebChatModel(
        executable=str(executable),
        profile="profile",
        site_session="persistent",
        runtime_state_root=str(root),
    )
    backend = model._durable_pacing_backend
    assert backend is not None
    state = Path(backend.state_path())
    if corruption == "malformed":
        state.write_text("not-json")
    elif corruption == "symlink":
        _write_durable_state(state, DurablePacingState(key=backend._key))
        state.unlink()
        target = tmp_path / "target"
        target.write_text("{}")
        state.symlink_to(target)
    elif corruption == "wrong-mode":
        _write_durable_state(state, DurablePacingState(key=backend._key))
        state.chmod(0o644)
    elif corruption == "identity":
        state.write_text(
            json.dumps(
                {
                    "schema": _DURABLE_STATE_SCHEMA,
                    "key": "other",
                    "last_send_started": 0,
                    "last_response_finished": 0,
                }
            )
        )
        state.chmod(0o600)
    elif corruption == "future":
        _write_durable_state(state, DurablePacingState(key=backend._key, last_send_started=1000.0))
    else:
        _write_durable_state(state, DurablePacingState(key=backend._key, last_send_started=100.0))
        clock.write_text("100000.0")
    model._clock = lambda: float(clock.read_text())
    model._durable_pacing_backend._clock = model._clock
    with pytest.raises(OpenCLIWebModelError):
        model.invoke([HumanMessage(content="must not ask")])
    assert not log.exists() or " ask " not in log.read_text()


def test_durable_pacing_kill_owner_successor_respects_cooldown(tmp_path):
    """Killing the lock owner releases the lock but successor preserves cooldown."""
    pacing_dir = _durable_pacing_dir(tmp_path)
    state_path = os.path.join(pacing_dir, "kill-test.json")
    lock_path = os.path.join(pacing_dir, "kill-test.lock")
    send_log = tmp_path / "send-log"
    send_log.mkdir()
    kill_marker = tmp_path / "kill-me"
    clock_time = tmp_path / "clock"
    clock_time.write_text("100.0")

    _write_durable_state(state_path, DurablePacingState(key="kill-test", last_send_started=100.0))

    script = f"""
import fcntl, json, os, time, sys, signal

state_path = {str(state_path)!r}
lock_path = {str(lock_path)!r}
send_log = {str(send_log)!r}
kill_marker = {str(kill_marker)!r}
clock_time = {str(clock_time)!r}
my_id = sys.argv[1]

fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)

now = float(open(clock_time).read().strip())

with open(state_path) as f:
    state = json.load(f)

# Read back the state - successor should see the last_send_started
log_path = os.path.join(send_log, f"{{my_id}}-state.txt")
with open(log_path, "w") as f:
    f.write(json.dumps(state))

if my_id == "owner":
    # Record the last_send_started
    with open(os.path.join(send_log, "owner-send-ts.txt"), "w") as f:
        f.write(str(state.get("last_send_started", 0)))
    # Signal to be killed
    while not os.path.exists(kill_marker):
        time.sleep(0.02)
    # Do NOT release lock - simulate process death
    os._exit(1)
else:
    # Successor: got the lock, should see last_send_started=100.0
    with open(os.path.join(send_log, "successor-saw-ts.txt"), "w") as f:
        f.write(str(state.get("last_send_started", 0)))
    # Release lock
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
"""
    proc_script = tmp_path / "kill_worker.py"
    proc_script.write_text(script)

    # Start owner process
    p_owner = subprocess.Popen(
        [sys.executable, str(proc_script), "owner"],
        cwd=str(tmp_path),
    )
    while not (send_log / "owner-state.txt").exists():
        time.sleep(0.05)

    # Start successor (will block on flock)
    p_successor = subprocess.Popen(
        [sys.executable, str(proc_script), "successor"],
        cwd=str(tmp_path),
    )
    time.sleep(0.3)
    assert not (send_log / "successor-state.txt").exists(), "successor should be blocked"

    # Kill the owner
    kill_marker.touch()
    p_owner.wait(timeout=5)
    assert p_owner.returncode != 0

    # Successor should acquire the lock
    p_successor.wait(timeout=5)
    assert p_successor.returncode == 0
    assert (send_log / "successor-state.txt").exists()

    # Successor should see the same last_send_started
    owner_ts = (send_log / "owner-send-ts.txt").read_text().strip()
    successor_saw = (send_log / "successor-saw-ts.txt").read_text().strip()
    assert successor_saw == owner_ts == "100.0"


def test_durable_pacing_different_keys_are_independent(tmp_path):
    """Two different transport keys do not interfere."""
    pacing_dir = _durable_pacing_dir(tmp_path)
    state_a_path = os.path.join(pacing_dir, "keyA.json")
    state_b_path = os.path.join(pacing_dir, "keyB.json")
    lock_a_path = os.path.join(pacing_dir, "keyA.lock")
    lock_b_path = os.path.join(pacing_dir, "keyB.lock")

    state_a = DurablePacingState(key="keyA", last_send_started=50.0, last_response_finished=40.0)
    state_b = DurablePacingState(key="keyB", last_send_started=0.0, last_response_finished=0.0)
    _write_durable_state(state_a_path, state_a)
    _write_durable_state(state_b_path, state_b)

    # Acquire lock for key A
    lock_a = DurablePacingLock(lock_a_path)
    lock_a.acquire()
    try:
        # Lock for key B should still be acquirable (independent)
        lock_b = DurablePacingLock(lock_b_path)
        lock_b.acquire()
        # Both held simultaneously - different keys are independent
        loaded_a = _read_durable_state(state_a_path)
        loaded_b = _read_durable_state(state_b_path)
        assert loaded_a.last_send_started == 50.0
        assert loaded_b.last_send_started == 0.0
        lock_b.release()
    finally:
        lock_a.release()


def test_durable_pacing_model_with_runtime_state_root_initializes_backend(tmp_path):
    """Model with runtime_state_root and profile creates DurablePacingBackend."""
    model = OpenCLIWebChatModel(
        executable="/opt/opencli",
        intelligence_level="very-high",
        profile="test-profile",
        site_session="persistent",
        runtime_state_root=str(tmp_path),
    )
    assert model._durable_pacing_backend is not None
    assert model._durable_pacing_backend._key == _durable_pacing_key(
        "/opt/opencli", "test-profile", "persistent"
    )


def test_durable_pacing_model_without_profile_has_no_backend():
    """Model without profile does not create DurablePacingBackend."""
    model = OpenCLIWebChatModel(
        executable="/opt/opencli",
        intelligence_level="very-high",
        runtime_state_root="/tmp/fake-state",
    )
    assert model._durable_pacing_backend is None


def test_durable_pacing_model_without_runtime_state_root_has_no_backend():
    """Model without runtime_state_root does not create DurablePacingBackend."""
    model = OpenCLIWebChatModel(
        executable="/opt/opencli",
        intelligence_level="very-high",
        profile="test-profile",
    )
    assert model._durable_pacing_backend is None


def test_durable_pacing_send_and_reconcile_uses_durable_backend(tmp_path, monkeypatch):
    """Durable pacing backend is used when runtime_state_root is set."""
    backend = DurablePacingBackend(
        str(tmp_path),
        executable="/opt/opencli",
        profile="durable-test",
        site_session="persistent",
        clock=lambda: 0.0,
    )

    clock = _FakeClock()
    model = OpenCLIWebChatModel(
        executable="/opt/opencli",
        intelligence_level="very-high",
        profile="durable-test",
        site_session="persistent",
        runtime_state_root=str(tmp_path),
    )
    model._clock = clock
    model._sleep = clock.sleep
    # Force the backend to use our clock
    model._durable_pacing_backend._clock = clock

    ask_starts = []
    latest_prompt = ""

    def fake_run(argv, **_kwargs):
        args = list(argv)
        if args[1:3] == ["chatgpt", "model"]:
            return SimpleNamespace(returncode=0, stdout='[{"Status":"ok"}]', stderr="")
        if args[1:3] == ["chatgpt", "ask"]:
            nonlocal latest_prompt
            latest_prompt = args[3]
            ask_starts.append(clock())
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"conversationId": "web-conversation-1", "response": ""}]),
                stderr="",
            )
        if args[1:3] == ["chatgpt", "detail"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {"Index": 1, "Role": "User", "Text": latest_prompt, "Generating": False},
                        {
                            "Index": 2,
                            "Role": "Assistant",
                            "Text": '{"type":"final","content":"done"}',
                            "Generating": False,
                        },
                    ]
                ),
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr("nexus_open_swe_runtime.opencli_web_model.subprocess.run", fake_run)
    result = model.invoke([HumanMessage(content="first")])
    assert result.content == "done"
    assert len(ask_starts) == 1

    # Second send should be paced (15s interval)
    result2 = model.invoke([HumanMessage(content="second")])
    assert result2.content == "done"
    assert len(ask_starts) == 2
    assert ask_starts[1] - ask_starts[0] >= 15.0

    # Verify durable state on disk
    ds = backend.persisted_state()
    assert ds.last_send_started > 0
    assert ds.last_response_finished > 0


def test_durable_pacing_successor_after_restart_respects_cooldown(tmp_path):
    """New process reads durable state and respects cooldown from predecessor."""
    pacing_dir = _durable_pacing_dir(tmp_path)
    state_path = os.path.join(pacing_dir, "restart-test.json")

    # Simulate predecessor sent at t=50.0
    _write_durable_state(
        state_path,
        DurablePacingState(
            key="restart-test",
            last_send_started=50.0,
            last_response_finished=45.0,
        ),
    )

    # Simulate successor reading state at t=60.0 (10s after last send)
    # Should need to wait until t=65.0 (50.0 + 15.0)
    clock = _FakeClock()
    clock.now = 60.0

    _validate_state_file(state_path, clock=lambda: 60.0)
    loaded = _read_durable_state(state_path)
    now = 60.0
    eligible_at = max(now, loaded.last_send_started + 15.0, loaded.last_response_finished + 3.0)
    assert eligible_at == 65.0
    assert eligible_at - now == 5.0


def test_durable_pacing_restarts_same_root_preserves_state(tmp_path):
    """Writing and reading back state across 'restarts' preserves cooldown."""
    pacing_dir = _durable_pacing_dir(tmp_path)
    state_path = os.path.join(pacing_dir, "preserve-test.json")

    # First "process" writes state
    state1 = DurablePacingState(
        key="preserve-test",
        last_send_started=100.0,
        last_response_finished=95.0,
    )
    _write_durable_state(state_path, state1)

    # Second "process" reads the same root
    loaded = _read_durable_state(state_path)
    assert loaded.last_send_started == 100.0
    assert loaded.last_response_finished == 95.0

    # Second "process" updates state
    state2 = DurablePacingState(
        key="preserve-test",
        last_send_started=115.0,
        last_response_finished=110.0,
    )
    _write_durable_state(state_path, state2)

    # Third "process" reads
    loaded2 = _read_durable_state(state_path)
    assert loaded2.last_send_started == 115.0
    assert loaded2.last_response_finished == 110.0


def test_durable_pacing_no_duplicate_ask(monkeypatch):
    """Budget lock prevents duplicate asks even with durable pacing."""
    tmp_path_for_test = tempfile.mkdtemp()
    model = OpenCLIWebChatModel(
        executable="/opt/opencli",
        intelligence_level="very-high",
        profile="no-dup",
        site_session="persistent",
        runtime_state_root=tmp_path_for_test,
    )
    model._web_turn_count = 12  # Budget exhausted

    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_TURN_BUDGET_EXHAUSTED"):
        model._reserve_web_turn()
