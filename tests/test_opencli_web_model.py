from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from nexus_open_swe_runtime import cli
from nexus_open_swe_runtime.opencli_web_model import (
    OPENCLI_WEB_PROTOCOL,
    OpenCLIWebChatModel,
    OpenCLIWebModelError,
)


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
            "site_session": "session-a",
            "timeout_seconds": 240,
        },
    )

    assert isinstance(model, OpenCLIWebChatModel)
    assert model.executable == "/opt/opencli"
    assert model.profile == "profile-a"
    assert model.site_session == "session-a"
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
            "executable": "opencli",
            "profile": "correct-profile",
            "site_session": "correct-session",
            "timeout_seconds": 120,
        },
    )

    assert model.executable == "opencli"
    assert model.profile == "correct-profile"
    assert model.site_session == "correct-session"
    assert model.timeout_seconds == 120


def test_build_model_rejects_missing_unknown_and_out_of_range_transport_config():
    valid = {
        "executable": "opencli",
        "profile": "",
        "site_session": "ephemeral",
        "timeout_seconds": 30,
    }
    cli._build_model({}, "opencli_chatgpt", "very-high", valid)
    cli._build_model({}, "opencli_chatgpt", "very-high", {**valid, "timeout_seconds": 900})
    for config in (
        None,
        {**valid, "timeout_seconds": 29},
        {**valid, "timeout_seconds": 901},
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
    assert detail[detail.index("--stable") + 1] == "6"
    prompt = json.loads(ask[3])
    assert prompt["protocol"] == OPENCLI_WEB_PROTOCOL
    assert prompt["role"] == "model_transport_only"
    assert prompt["messages"][-1]["content"] == "inspect the repository"


def test_opencli_web_model_retries_idempotent_model_selection_once(
    monkeypatch: pytest.MonkeyPatch,
):
    model_attempts = 0
    ask_attempts = 0
    latest_prompt = ""

    def fake_run(argv, **_kwargs):
        nonlocal model_attempts, ask_attempts, latest_prompt
        args = list(argv)
        if args[1:3] == ["chatgpt", "model"]:
            model_attempts += 1
            if model_attempts == 1:
                return SimpleNamespace(returncode=1, stdout="", stderr="selector busy")
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

    assert model.invoke([HumanMessage(content="hello")]).content == "done"
    assert model_attempts == 2
    assert ask_attempts == 1


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
    result = model.bind_tools([read_file]).invoke("inspect README")

    assert ask_count == 2
    assert result.tool_calls[0]["name"] == "read_file"
    assert result.tool_calls[0]["args"] == {"file_path": "README.md"}


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
