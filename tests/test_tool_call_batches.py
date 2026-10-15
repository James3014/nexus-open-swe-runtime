"""Ordered multi-call responses: fewer web turns, same per-call safety."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from nexus_open_swe_runtime import cli
from nexus_open_swe_runtime import opencli_web_model as web_model
from nexus_open_swe_runtime.opencli_web_model import OpenCLIWebChatModel, OpenCLIWebModelError
from nexus_open_swe_runtime.recovery import RecoveryIdentity

_DECLARED = [
    {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
    for name in ("read_file", "edit_file", "write_file", "record_worker_result")
]


def _batch(*calls: dict) -> str:
    return json.dumps({"type": "tool_calls", "calls": list(calls)}, separators=(",", ":"))


def _identity(tmp_path: Path, allowed=("a.py",)) -> RecoveryIdentity:
    return RecoveryIdentity(
        operation_id="b" * 64,
        execution_material_sha256="e" * 64,
        workspace=str(tmp_path.resolve()),
        task_id="task-1",
        unit_id="unit-1",
        session_id="session-1",
        allowed_paths=tuple(allowed),
        provider_id="opencli_chatgpt",
        model_id="advanced",
        worker_identity_sha256="w" * 64,
        transport_config_sha256="t" * 64,
        runtime_identity_sha256="r" * 64,
    )


def _scripted_graph(tmp_path: Path, monkeypatch, responses: list[str]):
    runtime = cli._load_runtime()
    model = OpenCLIWebChatModel(executable="/opt/opencli")
    identity = _identity(tmp_path)
    journal = cli.DurableOperationJournal(tmp_path / "state", identity)
    journal.prepare()
    effect_journal = cli.DurableEffectJournal(tmp_path / "state", identity)
    journal.effect_journal = effect_journal
    model.configure_recovery_journal(journal)
    sends: list[dict] = []
    pending = list(responses)

    def send(prompt, **_kwargs):
        sends.append(json.loads(prompt))
        turn_id = json.loads(prompt)["turn_id"]
        model._journal_ask(turn_id, prompt)
        model._journal_bound("conversation-1")
        response = pending.pop(0)
        model._journal_response(turn_id, response)
        return response

    monkeypatch.setattr(model, "_send_and_reconcile", send)
    monkeypatch.setattr(model, "_select_intelligence_level", lambda: None)
    graph = cli.build_repair_graph(
        model,
        tmp_path,
        runtime,
        ("a.py",),
        "opencli_chatgpt:advanced",
        effect_journal=effect_journal,
    )
    return runtime, model, graph, effect_journal, sends


def _effects(effect_journal) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(effect_journal.root.glob("effect_*.json"))
    ]


def test_batch_read_edit_record_runs_in_one_web_turn_strictly_in_order(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    response = _batch(
        {"name": "read_file", "arguments": {"file_path": "/a.py"}},
        {"name": "edit_file", "arguments": {"file_path": "/a.py", "old_string": "VALUE = 1", "new_string": "VALUE = 2"}},
        # Only correct when run after the first edit: proves strict ordering.
        {"name": "edit_file", "arguments": {"file_path": "/a.py", "old_string": "VALUE = 2", "new_string": "VALUE = 3"}},
        {"name": "record_worker_result", "arguments": {"envelope": {"summary": "set VALUE"}}},
    )
    runtime, model, graph, effect_journal, sends = _scripted_graph(
        tmp_path, monkeypatch, [response]
    )

    output = graph.invoke(
        {"messages": [runtime["human_message"]("repair a.py")]},
        config={"configurable": {"thread_id": "t1"}, "recursion_limit": cli._AGENT_RECURSION_LIMIT},
    )

    assert len(sends) == 1
    assert model._web_turn_count == 1
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "VALUE = 3\n"
    messages = output["messages"]
    ai_calls = [m for m in messages if isinstance(m, AIMessage) and m.tool_calls]
    # Every call reached the tool executor as its own single-call message.
    assert [len(m.tool_calls) for m in ai_calls] == [1, 1, 1, 1]
    assert [m.tool_calls[0]["name"] for m in ai_calls] == [
        "read_file", "edit_file", "edit_file", "record_worker_result"
    ]
    ids = [m.tool_calls[0]["id"] for m in ai_calls]
    assert len(set(ids)) == 4
    # Each mutation was journaled under its own call id and reached RESULT.
    effects = _effects(effect_journal)
    assert sorted(e["tool_call_id"] for e in effects) == sorted(ids[1:3])
    assert all(e["status"] == "RESULT" for e in effects)
    assert cli._recorded_payload(output, "record_worker_result") == {"summary": "set VALUE"}
    assert messages[-1].content == "Terminal recorder completed."
    # The rendered history never re-sends the raw batch response per call.
    assert "opencli_batch" not in json.dumps(web_model._message_payload(ai_calls[1]))


def test_batch_failed_call_skips_terminal_recorder_and_returns_to_model(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    first = _batch(
        {"name": "edit_file", "arguments": {"file_path": "/a.py", "old_string": "VALUE  = 1", "new_string": "VALUE = 2"}},
        {"name": "record_worker_result", "arguments": {"envelope": {"summary": "premature"}}},
    )
    second = _batch(
        {"name": "edit_file", "arguments": {"file_path": "/a.py", "old_string": "VALUE = 1", "new_string": "VALUE = 2"}},
        {"name": "record_worker_result", "arguments": {"envelope": {"summary": "fixed"}}},
    )
    runtime, model, graph, effect_journal, sends = _scripted_graph(
        tmp_path, monkeypatch, [first, second]
    )

    output = graph.invoke(
        {"messages": [runtime["human_message"]("repair a.py")]},
        config={"configurable": {"thread_id": "t2"}, "recursion_limit": cli._AGENT_RECURSION_LIMIT},
    )

    assert len(sends) == 2
    second_prompt_messages = sends[1]["messages"]
    recorded_before_retry = [
        m for m in second_prompt_messages
        if m.get("role") == "assistant"
        and any(c["name"] == "record_worker_result" for c in m.get("tool_calls", []))
    ]
    assert recorded_before_retry == []
    tool_results = [m for m in second_prompt_messages if m.get("role") == "tool"]
    assert len(tool_results) == 1 and "String not found" in tool_results[0]["content"]
    # The edit miss names the exact closest text so one retry suffices.
    assert '"VALUE = 1\\n"' in tool_results[0]["content"]
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert cli._recorded_payload(output, "record_worker_result") == {"summary": "fixed"}
    assert len(_effects(effect_journal)) == 1


@pytest.mark.parametrize(
    "calls",
    [
        # terminal recorder not last
        [
            {"name": "record_worker_result", "arguments": {"envelope": {"summary": "x"}}},
            {"name": "read_file", "arguments": {"file_path": "a.py"}},
        ],
        # undeclared tool
        [{"name": "execute", "arguments": {"command": "rm -rf /"}}],
        # composite terminal is single-call only
        [{"name": "write_file_and_record_worker_result", "arguments": {"file_path": "a.py", "content": "", "envelope": {"summary": "x"}}}],
        # too many calls
        [{"name": "read_file", "arguments": {"file_path": "a.py"}}] * 9,
        # empty batch
        [],
        # extra keys on a call
        [{"name": "read_file", "arguments": {"file_path": "a.py"}, "id": "x"}],
        # terminal recorder with extra arguments
        [{"name": "record_worker_result", "arguments": {"envelope": {"summary": "x"}, "more": 1}}],
    ],
)
def test_batch_rejects_invalid_shapes_fail_closed(calls):
    declared = _DECLARED + [
        {"type": "function", "function": {"name": "write_file_and_record_worker_result", "parameters": {"type": "object"}}}
    ]
    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_TOOL_CALL_INVALID"):
        OpenCLIWebChatModel._response_message(_batch(*calls), declared)


def test_batch_accepts_direct_terminal_form_last_and_single_call_stays_valid():
    response = _batch(
        {"name": "read_file", "arguments": {"file_path": "a.py"}},
        {"type": "record_worker_result", "envelope": {"summary": "x"}},
    )
    message = OpenCLIWebChatModel._response_message(response, _DECLARED)
    assert message.tool_calls[0]["name"] == "read_file"
    assert message.additional_kwargs["opencli_batch"]["count"] == 2

    single = '{"type":"tool_call","name":"read_file","arguments":{"file_path":"a.py"}}'
    legacy = OpenCLIWebChatModel._response_message(single, _DECLARED)
    assert legacy.tool_calls[0]["id"] == web_model._tool_call_id(
        "read_file", {"file_path": "a.py"}, single
    )
    assert "opencli_batch" not in legacy.additional_kwargs


def test_batch_identical_calls_get_distinct_ids():
    response = _batch(
        {"name": "read_file", "arguments": {"file_path": "a.py"}},
        {"name": "read_file", "arguments": {"file_path": "a.py"}},
    )
    calls = web_model._batch_calls(json.loads(response), {"read_file"})
    first = web_model._batch_message(response, calls, 0)
    second = web_model._batch_message(response, calls, 1)
    assert first.tool_calls[0]["id"] != second.tool_calls[0]["id"]
    assert first.id != second.id


def _batch_pair(response: str, index: int, *, status="success", content="ok"):
    calls = web_model._batch_calls(json.loads(response), {"read_file", "record_worker_result"})
    request = web_model._batch_message(response, calls, index)
    result = ToolMessage(
        content=content, tool_call_id=request.tool_calls[0]["id"], name="read_file", status=status
    )
    return request, result


@pytest.mark.parametrize(
    ("status", "content"), [("error", "boom"), ("success", "Error: String not found")]
)
def test_batch_continuation_stops_after_failed_call(status, content):
    response = _batch(
        {"name": "read_file", "arguments": {"file_path": "a.py"}},
        {"name": "record_worker_result", "arguments": {"envelope": {"summary": "x"}}},
    )
    model = OpenCLIWebChatModel(executable="/opt/opencli")
    request, result = _batch_pair(response, 0, status=status, content=content)
    assert model._batch_continuation([HumanMessage("x"), request, result], _DECLARED) is None


def test_batch_continuation_releases_next_call_without_web_turn_or_budget():
    response = _batch(
        {"name": "read_file", "arguments": {"file_path": "a.py"}},
        {"name": "record_worker_result", "arguments": {"envelope": {"summary": "x"}}},
    )
    model = OpenCLIWebChatModel(executable="/opt/opencli")
    model._select_intelligence_level = lambda: (_ for _ in ()).throw(
        AssertionError("continuation must not start a web turn")
    )
    request, result = _batch_pair(response, 0)
    following = model._generate([HumanMessage("x"), request, result], tools=_DECLARED)
    message = following.generations[0].message
    assert message.tool_calls[0]["name"] == "record_worker_result"
    assert message.additional_kwargs["opencli_batch"]["index"] == 1
    assert model._web_turn_count == 0


def test_batch_continuation_rejects_tampered_checkpoint_message():
    response = _batch(
        {"name": "read_file", "arguments": {"file_path": "a.py"}},
        {"name": "record_worker_result", "arguments": {"envelope": {"summary": "x"}}},
    )
    model = OpenCLIWebChatModel(executable="/opt/opencli")
    request, _ = _batch_pair(response, 0)
    tampered = AIMessage(
        content="",
        id=request.id,
        tool_calls=[{**request.tool_calls[0], "args": {"file_path": "other.py"}}],
        additional_kwargs=request.additional_kwargs,
    )
    result = ToolMessage(content="ok", tool_call_id=request.tool_calls[0]["id"], name="read_file")
    with pytest.raises(OpenCLIWebModelError, match="OPENCLI_WEB_BATCH_STATE_INVALID"):
        model._batch_continuation([HumanMessage("x"), tampered, result], _DECLARED)


def test_batch_resume_message_continues_at_checkpointed_call():
    response = _batch(
        {"name": "read_file", "arguments": {"file_path": "a.py"}},
        {"name": "read_file", "arguments": {"file_path": "b.py"}},
        {"name": "record_worker_result", "arguments": {"envelope": {"summary": "x"}}},
    )
    first, first_result = _batch_pair(response, 0)
    second, _ = _batch_pair(response, 1)
    recovered = OpenCLIWebChatModel._response_message(response, _DECLARED)

    fresh = OpenCLIWebChatModel._batch_resume_message(recovered, [HumanMessage("x")], _DECLARED)
    assert fresh.tool_calls[0]["id"] == first.tool_calls[0]["id"]

    resumed = OpenCLIWebChatModel._batch_resume_message(
        recovered, [HumanMessage("x"), first, first_result, second], _DECLARED
    )
    assert resumed.id == second.id
    assert resumed.tool_calls == second.tool_calls


def test_batch_protocol_is_complete_and_prompt_documents_ordered_batches():
    response = _batch({"name": "read_file", "arguments": {"file_path": "a.py"}})
    assert OpenCLIWebChatModel._is_complete_protocol_response(response)
    assert not OpenCLIWebChatModel._is_complete_protocol_response('{"type":"tool_calls","calls":[]}')
    model = OpenCLIWebChatModel(executable="/opt/opencli")
    prompt = json.loads(model._render_prompt([HumanMessage("x")], _DECLARED, None))
    rules = " ".join(prompt["rules"])
    assert "type:tool_calls" in rules
    assert "strictly" in rules and "skipped" in rules
    assert "return one JSON object with type=tool_call" in rules


def test_preloaded_sources_reads_only_authorized_regular_text_files(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.py").write_text("print('a')\n", encoding="utf-8")
    (workspace / "bin.dat").write_bytes(b"\x00\x01binary")
    (workspace / "big.py").write_text("x" * (cli._PRELOAD_FILE_BYTES + 1), encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET", encoding="utf-8")
    os.symlink(secret, workspace / "link.py")
    (workspace / "other.py").write_text("not authorized", encoding="utf-8")

    text = cli._preloaded_sources(
        workspace, ("a.py", "bin.dat", "big.py", "link.py", "new.py", "../secret.txt")
    )

    assert "<<<FILE a.py bytes=11" in text and "print('a')\n" in text
    assert "binary" not in text
    assert "TOP-SECRET" not in text
    assert "not authorized" not in text
    assert "big.py: not preloaded (too large)" in text
    assert "new.py: does not exist yet" in text
    assert cli._preloaded_sources(workspace, ()) == ""


def test_preloaded_sources_respects_total_budget(tmp_path):
    names = []
    for index in range(4):
        name = f"f{index}.py"
        (tmp_path / name).write_text("y" * 20_000, encoding="utf-8")
        names.append(name)
    text = cli._preloaded_sources(tmp_path, tuple(names))
    assert text.count(" bytes=20000 ") == 3
    assert "f3.py: not preloaded (too large)" in text


def test_edit_mismatch_hint_shows_exact_whitespace_of_closest_lines():
    content = "def f():\n\treturn 1\n\ndef g():\n    return 2\n"
    hint = cli._edit_mismatch_hint(content, "def f():\n    return 1")
    assert "lines 1-2" in hint
    assert json.dumps("def f():\n\treturn 1\n") in hint
    assert "only in whitespace" in hint
    assert cli._edit_mismatch_hint(content, "completely unrelated text here") == ""


def test_batch_crash_mid_batch_resumes_at_checkpointed_call_without_rerunning_edits(
    tmp_path, monkeypatch
):
    """Mirror of _worker_reconcile: replaying the recovered batch must not restart it."""
    (tmp_path / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    response = _batch(
        {"name": "edit_file", "arguments": {"file_path": "/a.py", "old_string": "VALUE = 1", "new_string": "VALUE = 2"}},
        {"name": "edit_file", "arguments": {"file_path": "/a.py", "old_string": "VALUE = 2", "new_string": "VALUE = 3"}},
        {"name": "record_worker_result", "arguments": {"envelope": {"summary": "chained"}}},
    )
    runtime = cli._load_runtime()
    identity = _identity(tmp_path)
    state_root = tmp_path / "state"
    saver, _namespace = cli.create_checkpoint(state_root, identity.operation_id, "ns")
    journal = cli.DurableOperationJournal(state_root, identity)
    journal.prepare()
    effect_journal = cli.DurableEffectJournal(state_root, identity)
    journal.effect_journal = effect_journal
    model = OpenCLIWebChatModel(executable="/opt/opencli")
    model.configure_recovery_journal(journal)

    def send(prompt, **_kwargs):
        turn_id = json.loads(prompt)["turn_id"]
        model._journal_ask(turn_id, prompt)
        model._journal_bound("conversation-1")
        model._journal_response(turn_id, response)
        return response

    monkeypatch.setattr(model, "_send_and_reconcile", send)
    monkeypatch.setattr(model, "_select_intelligence_level", lambda: None)
    original_recover = effect_journal.recover_write
    writes = {"count": 0}

    def crash_on_second_write(effect):
        writes["count"] += 1
        if writes["count"] == 2:
            raise RuntimeError("simulated crash before second write")
        return original_recover(effect)

    monkeypatch.setattr(effect_journal, "recover_write", crash_on_second_write)
    config = {"configurable": {"thread_id": identity.operation_id}, "recursion_limit": cli._AGENT_RECURSION_LIMIT}
    graph = cli.build_repair_graph(
        model, tmp_path, runtime, ("a.py",), "opencli_chatgpt:advanced",
        checkpointer=saver, effect_journal=effect_journal,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        graph.invoke({"messages": [runtime["human_message"]("repair a.py")]}, config=config)
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "VALUE = 2\n"

    # Restart: fresh model and graph over the same checkpoint and journals.
    monkeypatch.setattr(effect_journal, "recover_write", original_recover)
    restarted = OpenCLIWebChatModel(executable="/opt/opencli")
    restarted.configure_recovery_journal(journal)
    monkeypatch.setattr(
        restarted, "_send_and_reconcile",
        lambda *_a, **_k: pytest.fail("resume must not start a web turn"),
    )
    monkeypatch.setattr(restarted, "_select_intelligence_level", lambda: None)
    graph = cli.build_repair_graph(
        restarted, tmp_path, runtime, ("a.py",), "opencli_chatgpt:advanced",
        checkpointer=saver, effect_journal=effect_journal,
    )
    tool_schemas = [
        web_model.convert_to_openai_tool(t)
        for t in graph.get_graph().nodes["tools"].data.tools_by_name.values()
    ]
    recovered = journal.read()["recovered_response"]
    message = restarted._response_message(recovered, tool_schemas)
    message = restarted._batch_resume_message(
        message, graph.get_state(config).values["messages"], tool_schemas
    )
    assert message.additional_kwargs["opencli_batch"]["index"] == 1
    effect_journal.bind_turn(str(journal.read()["turn_id"]), message.tool_calls[0]["id"])
    graph.update_state(config, {"messages": [message]}, as_node="model")
    output = graph.invoke(None, config=config)

    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "VALUE = 3\n"
    assert cli._recorded_payload(output, "record_worker_result") == {"summary": "chained"}
    effects = _effects(effect_journal)
    assert len(effects) == 2 and all(e["status"] == "RESULT" for e in effects)
    ai_calls = [m for m in output["messages"] if isinstance(m, AIMessage) and m.tool_calls]
    assert [m.additional_kwargs["opencli_batch"]["index"] for m in ai_calls] == [0, 1, 2]


def test_worker_run_first_repair_prompt_preloads_authorized_source(tmp_path):
    from test_runtime import FakeGraph, _record, _runtime, _worker_request

    request = _worker_request(tmp_path)
    diagnosis = FakeGraph(
        cli.DIAGNOSIS_TOOLS,
        _record(
            "record_diagnosis",
            {"status": "ROOT_CAUSE_SUPPORTED", "summary": "a.py value", "evidence_paths": ["a.py"]},
        ),
    )
    repair = FakeGraph(cli.REPAIR_TOOLS, _record("record_worker_result", {"summary": "ok"}))
    payloads: list = []
    original_invoke = repair.invoke

    def capture(payload, config=None):
        payloads.append((payload, config))
        return original_invoke(payload, config=config)

    repair.invoke = capture
    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: object(),
        diagnosis_factory=lambda *_args: diagnosis,
        repair_factory=lambda *_args: repair,
    )

    assert result["status"] == "COMPLETED"
    first_message = payloads[0][0]["messages"][0]
    assert "<<<FILE a.py bytes=10 sha256=" in first_message
    assert "VALUE = 1\n\n<<<END FILE a.py>>>" in first_message
    assert payloads[0][1]["recursion_limit"] == cli._AGENT_RECURSION_LIMIT


def test_edit_mismatch_hint_is_bounded_for_large_old_string():
    import time

    content = "".join(f"value_{index} = {index} * 2 + 1  # filler text\n" for index in range(5000))
    huge = "x = 1\n" * 2000
    started = time.monotonic()
    assert cli._edit_mismatch_hint(content, huge) == ""
    assert cli._edit_mismatch_hint(content, "y" * 5000) == ""
    wide = "".join(f"value_{index} = {index} * 3\n" for index in range(60))
    cli._edit_mismatch_hint(content, wide)
    assert time.monotonic() - started < 10
