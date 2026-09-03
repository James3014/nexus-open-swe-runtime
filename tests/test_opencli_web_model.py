from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from nexus_open_swe_runtime import cli
from nexus_open_swe_runtime import opencli_web_model as web_model
from nexus_open_swe_runtime.opencli_web_model import (
    OPENCLI_WEB_PROTOCOL,
    OpenCLIWebChatModel,
    OpenCLIWebModelError,
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
                stdout=json.dumps([
                    {
                        "conversationId": "web-conversation-1",
                        "conversationUrl": "https://chatgpt.com/c/web-conversation-1",
                        "tool": "chatgpt",
                        "response": "",
                    }
                ]),
                stderr="",
            )
        if args[1:3] == ["chatgpt", "detail"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([
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
                ]),
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
                stdout=json.dumps([
                    {"Index": 1, "Role": "User", "Text": latest_prompt, "Generating": False},
                    {
                        "Index": 2,
                        "Role": "Assistant",
                        "Text": '{"type":"final","content":"done"}',
                        "Generating": False,
                    },
                ]),
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
                stdout=json.dumps([
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
                ]),
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
                stdout=json.dumps([
                        {"Index": 1, "Role": "User", "Text": latest_prompt, "Generating": False},
                    {"Index": 2, "Role": "Assistant", "Text": '{"type":"final","content":"ok"}', "Generating": False},
                ]),
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
                stdout=json.dumps([
                    {
                        "conversationId": conversation_id,
                        "response": '{"type":"final","content":"done"}',
                    }
                ]),
                stderr="",
            )
        if args[1:3] == ["chatgpt", "detail"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([
                    {"Index": 1, "Role": "User", "Text": latest_prompt, "Generating": False},
                    {
                        "Index": 2,
                        "Role": "Assistant",
                        "Text": '{"type":"final","content":"done"}',
                        "Generating": False,
                    },
                ]),
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
                stdout=json.dumps([
                    {"Index": 1, "Role": "User", "Text": latest_prompt, "Generating": False},
                    {"Index": 2, "Role": "Assistant", "Text": text, "Generating": False},
                ]),
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
                stdout=json.dumps([
                    {"Index": 1, "Role": "User", "Text": latest_prompt, "Generating": False},
                    {"Index": 2, "Role": "Assistant", "Text": text, "Generating": False},
                ]),
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
                stdout=json.dumps([
                    {"Index": 1, "Role": "User", "Text": latest_prompt, "Generating": False},
                    {"Index": 2, "Role": "Assistant", "Text": "{", "Generating": False},
                ]),
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
                rows.append({
                    "Index": 2,
                    "Role": "Assistant",
                    "Text": '{"type":"final","content":"reconciled"}',
                    "Generating": False,
                })
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
                stdout=json.dumps([
                    {"Index": 1, "Role": "User", "Text": latest_prompt, "Generating": False},
                    {"Index": 2, "Role": "Assistant", "Text": '{"type":"final","content":"reconciled"}', "Generating": False},
                ]),
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
