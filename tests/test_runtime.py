from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from nexus_open_swe_runtime import cli
from nexus_open_swe_runtime.opencli_web_model import OpenCLIWebChatModel, _tool_call_id
from nexus_open_swe_runtime.recovery import (
    DurableEffectJournal,
    DurableOperationJournal,
    RecoveryIdentity,
    create_checkpoint,
)


def test_root_checkpoint_config_reads_sqlite_checkpoint_without_subgraph_namespace(tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        value: str

    connection = sqlite3.connect(str(tmp_path / "checkpoint.sqlite"), check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    builder = StateGraph(State)
    builder.add_node("step", lambda _state: {"value": "recovered"})
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "operation-1"}}

    assert graph.invoke({"value": "pending"}, config=config) == {"value": "recovered"}
    assert graph.get_state(config).values == {"value": "recovered"}
    assert connection.execute("SELECT DISTINCT checkpoint_ns FROM checkpoints").fetchall() == [
        ("",)
    ]
    with pytest.raises(ValueError, match="Subgraph open-swe-repair-v1 not found"):
        graph.get_state({
            "configurable": {
                "thread_id": "operation-1",
                "checkpoint_ns": "open-swe-repair-v1",
            }
        })


def test_worker_reconcile_real_graph_replays_write_effect_once_after_restart(tmp_path):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    request = _worker_request(tmp_path)
    request.update(operation="worker_reconcile", provider_id="google_genai", model_id="test-model")
    workspace = Path(request["workspace_path"])
    state_root = Path(request["runtime_state_root"])
    identity = RecoveryIdentity(
        operation_id=request["operation_id"],
        execution_material_sha256="b" * 64,
        workspace=str(workspace.resolve()),
        task_id="task-1",
        unit_id="u1",
        session_id="session-1",
        allowed_paths=("a.py",),
        provider_id="google_genai",
        model_id="test-model",
        worker_identity_sha256=request["worker_identity_sha256"],
        transport_config_sha256=cli._sha256("{}"),
        runtime_identity_sha256=cli._sha256(
            cli._canonical_json({
                "module_sha256": cli._sha256(Path(cli.__file__).read_bytes()),
                "deepagents": cli._deepagents_version(),
                "checkpoint_namespace": "open-swe-repair-v1",
            })
        ),
    )
    journal = DurableOperationJournal(state_root, identity)
    journal.prepare()
    journal.ask_dispatching(turn_id="turn-1", prompt="repair", ordinal=0)
    journal.conversation_bound("conversation-1")
    cli._atomic_json(
        cli._operation_path(request),
        {
            **cli._write_started(request, "worker"),
            "status": "OPEN_SWE_OUTCOME_UNKNOWN",
            "outcome_unknown": True,
            "retry_safe": False,
            "execution_material_sha256": identity.execution_material_sha256,
            "session_id": identity.session_id,
        },
    )
    checkpoint, _ = cli.create_checkpoint(
        state_root, request["operation_id"], identity.checkpoint_namespace
    )
    checkpoint.conn.close()

    class RecoveryModel(BaseChatModel):
        model_name: str = "test-model"
        continuation_conversations: list[str] = []

        @property
        def _llm_type(self):
            return "test-recovery-model"

        def bind_tools(self, tools, **kwargs):
            return self

        def configure_recovery_journal(self, value):
            return None

        def _detail_response(self, conversation_id, *, wait, turn_id):
            assert (conversation_id, wait, turn_id) == ("conversation-1", False, "turn-1")
            return '{"type":"tool_call","name":"write_file","arguments":{"file_path":"a.py","content":"done\\n"}}'

        def _response_message(self, _response, _tools):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "a.py", "content": "done\n"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )

        def _generate(self, messages, **kwargs):
            if any(
                getattr(message, "name", None) == "record_worker_result" for message in messages
            ):
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])
            self.continuation_conversations.append("conversation-1")
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "record_worker_result",
                                    "args": {"envelope": {"summary": "done"}},
                                    "id": "record-1",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    )
                ]
            )

    model = RecoveryModel()
    result = cli._worker_reconcile(
        request,
        runtime_loader=cli._load_runtime,
        model_builder=lambda *_args: model,
        graph_builder=cli.build_repair_graph,
    )
    assert result["status"] == "COMPLETED"
    assert (workspace / "a.py").read_text(encoding="utf-8") == "done\n"
    effects = list((state_root / "recovery" / "effects").glob("*.json"))
    assert len(effects) == 1
    assert json.loads(effects[0].read_text(encoding="utf-8"))["status"] == "RESULT"
    assert model.continuation_conversations == ["conversation-1"]


class FakeGraph:
    def __init__(self, surface, output=None, effect=None, error=None):
        self.surface = tuple(surface)
        self.output = output
        self.effect = effect
        self.error = error
        self.calls = 0

    def get_graph(self):
        tools = {name: object() for name in self.surface}
        return SimpleNamespace(
            nodes={"tools": SimpleNamespace(data=SimpleNamespace(tools_by_name=tools))}
        )

    def invoke(self, _payload, config=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.effect is not None:
            self.effect()
        return self.output


def _record(name: str, envelope: dict):
    return {
        "messages": [SimpleNamespace(tool_calls=[{"name": name, "args": {"envelope": envelope}}])]
    }


def _runtime():
    return {"human_message": lambda content: content}


def test_cli_rejects_non_object_json_fail_closed() -> None:
    process = subprocess.run(
        [sys.executable, str(Path(cli.__file__).resolve())],
        input="[]\n",
        capture_output=True,
        text=True,
        check=False,
    )
    result = json.loads(process.stdout)
    assert process.returncode == 0
    assert result["status"] == "OPEN_SWE_RUNTIME_PROTOCOL_FAILED"
    assert result["provider_id"] == ""
    assert result["model_id"] == ""


def _semantic_request(tmp_path: Path) -> dict:
    repo = tmp_path / "repo"
    repo.mkdir()
    return {
        "schema": cli.REQUEST_SCHEMA,
        "operation": "semantic_run",
        "operation_id": "a" * 64,
        "provider_id": "google_genai",
        "model_id": "gemini-test",
        "repository_root": str(repo),
        "runtime_state_root": str(tmp_path / "state"),
        "prompt": "bounded semantic prompt",
    }


def _worker_request(tmp_path: Path) -> dict:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    artifact = tmp_path / "evidence.json"
    artifact.write_text(
        '{"schema":"external_execution_envelope.v1","failure":"VALUE must be 2"}\n',
        encoding="utf-8",
    )
    return {
        "schema": cli.REQUEST_SCHEMA,
        "operation": "worker_run",
        "operation_id": "b" * 64,
        "provider_id": "google_genai",
        "model_id": "gemini-test",
        "runtime_state_root": str(tmp_path / "state"),
        "workspace_path": str(workspace),
        "artifact_path": str(artifact),
        "prompt": "\n".join([
            "task_id=task-1",
            "unit_id=u1",
            'authorized_mutation_paths=["a.py"]',
            "bounded repair",
        ]),
        "session_id": "",
        "worker_identity_sha256": "c" * 64,
    }


@pytest.mark.parametrize(
    "raw_state",
    [b"[]", b"null", b'"string"', b"123", b"true", b"{malformed"],
)
def test_corrupt_semantic_operation_state_fails_closed_without_effect_or_overwrite(
    tmp_path, raw_state
):
    request = _semantic_request(tmp_path)
    state_path = cli._operation_path(request)
    state_path.parent.mkdir(parents=True)
    state_path.write_bytes(raw_state)
    graph = FakeGraph(
        cli.SEMANTIC_TOOLS,
        _record("record_finding", {"schema": "external_execution_envelope.v1"}),
    )

    result = cli._semantic_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: object(),
        graph_factory=lambda *_args: graph,
    )

    assert result["status"] == "OPEN_SWE_OPERATION_STATE_CORRUPT"
    assert graph.calls == 0
    assert state_path.read_bytes() == raw_state


@pytest.mark.parametrize(
    "raw_state",
    [b"[]", b"null", b'"string"', b"123", b"true", b"{malformed"],
)
def test_corrupt_worker_operation_state_fails_closed_without_effect_or_overwrite(
    tmp_path, raw_state
):
    request = _worker_request(tmp_path)
    state_path = cli._operation_path(request)
    state_path.parent.mkdir(parents=True)
    state_path.write_bytes(raw_state)
    graph = FakeGraph(cli.DIAGNOSIS_TOOLS)

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: object(),
        diagnosis_factory=lambda *_args: graph,
        repair_factory=lambda *_args: graph,
    )

    assert result["status"] == "OPEN_SWE_OPERATION_STATE_CORRUPT"
    assert graph.calls == 0
    assert state_path.read_bytes() == raw_state


def test_semantic_terminal_result_is_durable_and_reconcile_never_redispatches(tmp_path):
    request = _semantic_request(tmp_path)
    graph = FakeGraph(
        cli.SEMANTIC_TOOLS,
        _record("record_finding", {"schema": "external_execution_envelope.v1", "binding": {}}),
    )
    result = cli._semantic_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: object(),
        graph_factory=lambda *_args: graph,
    )

    assert result["status"] == "INTELLIGENCE_COMPLETED"
    assert graph.calls == 1
    request["operation"] = "semantic_reconcile"
    reconciled = cli.dispatch(request)
    assert reconciled == result
    assert graph.calls == 1


def test_semantic_started_without_terminal_reconciles_to_unknown(tmp_path):
    request = _semantic_request(tmp_path)
    cli._write_started(request, "semantic")
    request["operation"] = "semantic_reconcile"

    result = cli.dispatch(request)

    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert result["outcome_unknown"] is True
    assert result["process_started"] is False


def test_worker_supported_diagnosis_produces_bounded_result_and_workspace_index(tmp_path):
    request = _worker_request(tmp_path)
    workspace = Path(request["workspace_path"])
    diagnosis = FakeGraph(
        cli.DIAGNOSIS_TOOLS,
        _record(
            "record_diagnosis",
            {
                "status": "ROOT_CAUSE_SUPPORTED",
                "summary": "a.py contains the failing value",
                "evidence_paths": ["a.py"],
            },
        ),
    )
    repair = FakeGraph(
        cli.REPAIR_TOOLS,
        _record("record_worker_result", {"summary": "repaired a.py"}),
        effect=lambda: (workspace / "a.py").write_text("VALUE = 2\n", encoding="utf-8"),
    )

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: object(),
        diagnosis_factory=lambda *_args: diagnosis,
        repair_factory=lambda *_args: repair,
    )

    assert result["status"] == "COMPLETED"
    assert result["diagnosis_status"] == "ROOT_CAUSE_SUPPORTED"
    assert result["repair_admitted"] is True
    assert result["repair_phase_count"] == 1
    assert result["worker_identity_sha256"] == "c" * 64
    assert (workspace / "a.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    request["operation"] = "worker_reconcile"
    request["prompt"] = ""
    request["artifact_path"] = ""
    reconciled = cli.dispatch(request)
    assert reconciled == result
    assert diagnosis.calls == 1
    assert repair.calls == 1


def test_worker_supported_diagnosis_and_repair_use_distinct_phase_models(tmp_path):
    request = _worker_request(tmp_path)
    events: list[str] = []
    diagnosis = FakeGraph(
        cli.DIAGNOSIS_TOOLS,
        _record(
            "record_diagnosis",
            {
                "status": "ROOT_CAUSE_SUPPORTED",
                "summary": "supported",
                "evidence_paths": ["a.py"],
            },
        ),
        effect=lambda: events.append("diagnosis_invoke"),
    )
    repair = FakeGraph(
        cli.REPAIR_TOOLS,
        _record("record_worker_result", {"summary": "repaired"}),
        effect=lambda: events.append("repair_invoke"),
    )
    models: list[object] = []
    diagnosis_models: list[object] = []
    repair_models: list[object] = []
    model_args: list[tuple[object, ...]] = []

    def model_factory(*args):
        model = object()
        models.append(model)
        model_args.append(args)
        events.append("model")
        return model

    def diagnosis_factory(model, *_args):
        diagnosis_models.append(model)
        events.append("diagnosis_graph")
        return diagnosis

    def repair_factory(model, *_args):
        repair_models.append(model)
        events.append("repair_graph")
        return repair

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=model_factory,
        diagnosis_factory=diagnosis_factory,
        repair_factory=repair_factory,
    )

    assert result["status"] == "COMPLETED"
    assert len(models) == 2
    assert diagnosis_models == [models[0]]
    assert repair_models == [models[1]]
    assert models[0] is not models[1]
    assert len(model_args) == 2
    assert model_args[0] == model_args[1]
    assert model_args[0][1:] == (
        request["provider_id"],
        request["model_id"],
        request.get("transport_config"),
        request["runtime_state_root"],
    )
    assert events == [
        "model",
        "diagnosis_graph",
        "diagnosis_invoke",
        "model",
        "repair_graph",
        "repair_invoke",
    ]


def test_worker_opencli_repair_phase_starts_without_diagnosis_conversation(tmp_path):
    request = _worker_request(tmp_path)
    request.update({
        "provider_id": "opencli_chatgpt",
        "model_id": "very-high",
        "transport_config": {
            "executable": "/opt/opencli",
            "profile": "balanced",
            "site_session": "ephemeral",
            "timeout_seconds": 120,
        },
    })
    diagnosis = FakeGraph(
        cli.DIAGNOSIS_TOOLS,
        _record(
            "record_diagnosis",
            {
                "status": "ROOT_CAUSE_SUPPORTED",
                "summary": "supported",
                "evidence_paths": ["a.py"],
            },
        ),
    )
    repair = FakeGraph(
        cli.REPAIR_TOOLS,
        _record("record_worker_result", {"summary": "repaired"}),
    )

    class PhaseModel:
        def __init__(self):
            self._conversation_id = None

    models: list[PhaseModel] = []

    def model_factory(*_args):
        model = PhaseModel()
        models.append(model)
        return model

    def diagnosis_factory(model, *_args):
        model._conversation_id = "diagnosis-conversation"
        return diagnosis

    def repair_factory(model, *_args):
        assert model is models[1]
        assert model._conversation_id is None
        return repair

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=model_factory,
        diagnosis_factory=diagnosis_factory,
        repair_factory=repair_factory,
    )

    assert result["status"] == "COMPLETED"
    assert len(models) == 2


def test_worker_inconclusive_diagnosis_does_not_construct_repair_phase(tmp_path):
    request = _worker_request(tmp_path)
    diagnosis = FakeGraph(
        cli.DIAGNOSIS_TOOLS,
        _record(
            "record_diagnosis",
            {
                "status": "INCONCLUSIVE",
                "summary": "evidence is insufficient",
                "evidence_paths": [],
            },
        ),
    )
    model_calls = 0
    repair_calls = 0

    def model_factory(*_args):
        nonlocal model_calls
        model_calls += 1
        return SimpleNamespace(_conversation_id=None)

    def repair_factory(*_args):
        nonlocal repair_calls
        repair_calls += 1
        raise AssertionError("repair graph must not be constructed")

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=model_factory,
        diagnosis_factory=lambda *_args: diagnosis,
        repair_factory=repair_factory,
    )

    assert result["status"] == "COMPLETED"
    assert result["diagnosis_status"] == "INCONCLUSIVE"
    assert result["repair_admitted"] is False
    assert result["repair_phase_count"] == 0
    assert model_calls == 1
    assert repair_calls == 0


def test_worker_repair_model_construction_failure_is_unknown_without_repair_invocation(tmp_path):
    request = _worker_request(tmp_path)
    diagnosis = FakeGraph(
        cli.DIAGNOSIS_TOOLS,
        _record(
            "record_diagnosis",
            {
                "status": "ROOT_CAUSE_SUPPORTED",
                "summary": "supported",
                "evidence_paths": ["a.py"],
            },
        ),
    )
    model_calls = 0
    repair_calls = 0

    def model_factory(*_args):
        nonlocal model_calls
        model_calls += 1
        if model_calls == 2:
            raise RuntimeError("repair construction failed")
        return object()

    def repair_factory(*_args):
        nonlocal repair_calls
        repair_calls += 1
        raise AssertionError("repair graph must not be invoked")

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=model_factory,
        diagnosis_factory=lambda *_args: diagnosis,
        repair_factory=repair_factory,
    )

    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert result["outcome_unknown"] is True
    assert result["repair_admitted"] is True
    assert result["repair_phase_count"] == 1
    assert model_calls == 2
    assert repair_calls == 0


def test_worker_repair_graph_construction_failure_is_unknown_without_repair_invocation(tmp_path):
    request = _worker_request(tmp_path)
    diagnosis = FakeGraph(
        cli.DIAGNOSIS_TOOLS,
        _record(
            "record_diagnosis",
            {
                "status": "ROOT_CAUSE_SUPPORTED",
                "summary": "supported",
                "evidence_paths": ["a.py"],
            },
        ),
    )
    models: list[object] = []

    def model_factory(*_args):
        model = object()
        models.append(model)
        return model

    def repair_factory(*_args):
        raise RuntimeError("repair graph construction failed")

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=model_factory,
        diagnosis_factory=lambda *_args: diagnosis,
        repair_factory=repair_factory,
    )

    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert result["outcome_unknown"] is True
    assert result["repair_admitted"] is True
    assert result["repair_phase_count"] == 1
    assert len(models) == 2
    assert models[0] is not models[1]


@pytest.mark.parametrize("evidence_paths", [[], ["missing.py"]])
def test_worker_invalid_diagnosis_evidence_never_constructs_repair(
    tmp_path, evidence_paths: list[str]
):
    request = _worker_request(tmp_path)
    diagnosis = FakeGraph(
        cli.DIAGNOSIS_TOOLS,
        _record(
            "record_diagnosis",
            {
                "status": "ROOT_CAUSE_SUPPORTED",
                "summary": "unsupported evidence",
                "evidence_paths": evidence_paths,
            },
        ),
    )
    model_calls = 0
    repair_calls = 0

    def model_factory(*_args):
        nonlocal model_calls
        model_calls += 1
        return object()

    def repair_factory(*_args):
        nonlocal repair_calls
        repair_calls += 1
        raise AssertionError("invalid evidence must block repair construction")

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=model_factory,
        diagnosis_factory=lambda *_args: diagnosis,
        repair_factory=repair_factory,
    )

    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert result["repair_admitted"] is False
    assert result["repair_phase_count"] == 0
    assert model_calls == 1
    assert repair_calls == 0


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("provider_id", "other_provider"),
        ("model_id", "other_model"),
        ("worker_identity_sha256", "d" * 64),
        ("workspace_path", "/tmp/other-workspace"),
    ],
)
def test_retained_worker_session_rejects_identity_substitution(tmp_path, field, replacement):
    request = _worker_request(tmp_path)
    task_id, unit_id, allowed_paths, session_id = cli._worker_context(request, request["prompt"])
    assert task_id == "task-1"
    assert unit_id == "u1"
    assert allowed_paths == ("a.py",)

    resumed = dict(request)
    resumed["session_id"] = session_id
    resumed[field] = replacement
    with pytest.raises(cli.RuntimeErrorBounded, match="SESSION_BINDING_MISMATCH"):
        cli._worker_context(resumed, resumed["prompt"])


def test_worker_ambiguous_repair_is_durable_unknown_and_not_reexecuted(tmp_path):
    request = _worker_request(tmp_path)
    diagnosis = FakeGraph(
        cli.DIAGNOSIS_TOOLS,
        _record(
            "record_diagnosis",
            {
                "status": "ROOT_CAUSE_SUPPORTED",
                "summary": "supported",
                "evidence_paths": ["a.py"],
            },
        ),
    )
    repair = FakeGraph(cli.REPAIR_TOOLS, error=TimeoutError("ambiguous"))

    first = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: object(),
        diagnosis_factory=lambda *_args: diagnosis,
        repair_factory=lambda *_args: repair,
    )
    second = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: object(),
        diagnosis_factory=lambda *_args: diagnosis,
        repair_factory=lambda *_args: repair,
    )

    assert first["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert first["outcome_unknown"] is True
    assert second == first
    assert diagnosis.calls == 1
    assert repair.calls == 1


def test_worker_reconcile_does_not_substitute_completed_operation_from_stale_workspace_index(
    tmp_path,
):
    """A workspace index entry for operation A cannot complete target operation B."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state_root = tmp_path / "state"
    operation_a = "a" * 64
    operation_b = "b" * 64
    completed_a = {
        "schema": cli.RESULT_SCHEMA,
        "kind": "worker",
        "status": "COMPLETED",
        "operation_id": operation_a,
        "directory": str(workspace.resolve()),
        "process_started": True,
        "outcome_unknown": False,
        "retry_safe": False,
    }
    cli._atomic_json(state_root / "operations" / f"{operation_a}.json", completed_a)
    cli._atomic_json(
        state_root / "workspaces" / f"{cli._sha256(str(workspace.resolve()))}.json",
        completed_a,
    )
    cli._atomic_json(
        state_root / "operations" / f"{operation_b}.json",
        {
            **completed_a,
            "operation_id": operation_b,
            "status": "STARTED",
            "outcome_unknown": False,
        },
    )

    result = cli.dispatch({
        "schema": cli.REQUEST_SCHEMA,
        "operation": "worker_reconcile",
        "operation_id": operation_b,
        "runtime_state_root": str(state_root),
        "workspace_path": str(workspace),
    })

    assert result["operation_id"] == operation_b
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert result["outcome_unknown"] is True


def test_worker_reconcile_fails_closed_on_same_operation_material_mismatch(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    other_workspace = tmp_path / "other-workspace"
    other_workspace.mkdir()
    state_root = tmp_path / "state"
    operation_id = "c" * 64
    cli._atomic_json(
        state_root / "operations" / f"{operation_id}.json",
        {
            "schema": cli.RESULT_SCHEMA,
            "kind": "worker",
            "status": "COMPLETED",
            "operation_id": operation_id,
            "directory": str(workspace.resolve()),
            "provider_id": "provider-a",
            "model_id": "model-a",
            "worker_identity_sha256": "d" * 64,
            "process_started": True,
            "outcome_unknown": False,
            "retry_safe": False,
        },
    )

    result = cli.dispatch({
        "schema": cli.REQUEST_SCHEMA,
        "operation": "worker_reconcile",
        "operation_id": operation_id,
        "runtime_state_root": str(state_root),
        "workspace_path": str(other_workspace),
        "provider_id": "provider-a",
        "model_id": "model-a",
        "worker_identity_sha256": "d" * 64,
    })

    assert result["operation_id"] == operation_id
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert result["outcome_unknown"] is True


def test_worker_reconcile_direct_restart_trace_has_one_winner_and_zero_call_loser(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "a.py"
    target.write_text("old\n", encoding="utf-8")
    state_root = tmp_path / "state"
    operation_id = "r" * 64
    runtime_identity = cli._sha256(
        cli._canonical_json({
            "module_sha256": cli._sha256(Path(cli.__file__).read_bytes()),
            "deepagents": cli._deepagents_version(),
            "checkpoint_namespace": "open-swe-repair-v1",
        })
    )
    identity = RecoveryIdentity(
        operation_id=operation_id,
        execution_material_sha256="b" * 64,
        workspace=str(workspace.resolve()),
        task_id="task-1",
        unit_id="unit-1",
        session_id="session-1",
        allowed_paths=("a.py",),
        provider_id="google_genai",
        model_id="test-model",
        worker_identity_sha256="c" * 64,
        transport_config_sha256=cli._sha256("{}"),
        runtime_identity_sha256=runtime_identity,
    )
    journal = DurableOperationJournal(state_root, identity)
    journal.prepare()
    journal.ask_dispatching(turn_id="turn-1", prompt="repair", ordinal=0)
    journal.conversation_bound("conversation-1")
    op_state = {
        "schema": cli.RESULT_SCHEMA,
        "kind": "worker",
        "status": "OPEN_SWE_OUTCOME_UNKNOWN",
        "operation_id": operation_id,
        "directory": str(workspace.resolve()),
        "provider_id": "google_genai",
        "model_id": "test-model",
        "worker_identity_sha256": "c" * 64,
        "process_started": True,
        "outcome_unknown": True,
        "retry_safe": False,
    }
    cli._atomic_json(
        cli._operation_path({"runtime_state_root": str(state_root), "operation_id": operation_id}),
        op_state,
    )
    checkpoint, _ = create_checkpoint(state_root, operation_id, "open-swe-repair-v1")
    request = {
        "schema": cli.REQUEST_SCHEMA,
        "operation": "worker_reconcile",
        "operation_id": operation_id,
        "runtime_state_root": str(state_root),
        "workspace_path": str(workspace),
        "provider_id": "google_genai",
        "model_id": "test-model",
        "worker_identity_sha256": "c" * 64,
        "transport_config": {},
    }
    counters = {"detail": 0, "update": 0, "invoke": 0, "effect": 0, "continuation": 0}
    barrier = __import__("threading").Barrier(2)

    class Model:
        _conversation_id = "conversation-1"

        def configure_recovery_journal(self, journal):
            self.journal = journal

        def _detail_response(self, conversation_id, *, wait, turn_id):
            counters["detail"] += 1
            assert (conversation_id, wait, turn_id) == ("conversation-1", False, "turn-1")
            return '{"type":"tool_call","name":"write_file","arguments":{"file_path":"a.py","content":"done\\n"}}'

        def _response_message(self, response, tools):
            from langchain_core.messages import AIMessage

            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "a.py", "content": "done\n"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )

        def continue_same_conversation(self):
            assert self._conversation_id == "conversation-1"
            counters["continuation"] += 1

    class Delegate:
        def write(self, file_path, content):
            target.write_text(content, encoding="utf-8")

    class Graph:
        def __init__(self, model, effect_journal):
            self.model = model
            self.effect_journal = effect_journal

        def get_graph(self):
            from langchain_core.tools import tool

            @tool
            def write_file(file_path: str, content: str) -> str:
                """Write one file."""
                return file_path + content

            return type(
                "G",
                (),
                {
                    "nodes": {
                        "tools": type(
                            "T",
                            (),
                            {
                                "data": type(
                                    "D", (), {"tools_by_name": {"write_file": write_file}}
                                )()
                            },
                        )()
                    }
                },
            )()

        def update_state(self, config, values, *, as_node):
            counters["update"] += 1
            assert as_node == "model"
            assert config["configurable"] == {"thread_id": operation_id}
            assert values["messages"][0].tool_calls[0]["id"] == "call-1"

        def invoke(self, payload, *, config):
            counters["invoke"] += 1
            assert payload is None
            backend = cli.ScopedRepairBackend(Delegate(), workspace, ("a.py",), self.effect_journal)
            backend.write("a.py", "done\n")
            counters["effect"] += 1
            self.model.continue_same_conversation()
            return _record("record_worker_result", {"summary": "recovered"})

    def run():
        barrier.wait()
        return cli._worker_reconcile(
            request,
            runtime_loader=lambda: {},
            model_builder=lambda *_args: Model(),
            graph_builder=lambda model, *_args: Graph(model, _args[-1]),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(lambda _ignored: run(), range(2))
    assert first["status"] == second["status"] == "COMPLETED"
    assert counters == {"detail": 1, "update": 1, "invoke": 1, "effect": 1, "continuation": 1}
    assert target.read_text(encoding="utf-8") == "done\n"


def test_worker_reconcile_repairs_malformed_write_before_checkpoint_resume(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "a.py"
    target.write_text("old\n", encoding="utf-8")
    state_root = tmp_path / "state"
    operation_id = "q" * 64
    runtime_identity = cli._sha256(
        cli._canonical_json({
            "module_sha256": cli._sha256(Path(cli.__file__).read_bytes()),
            "deepagents": cli._deepagents_version(),
            "checkpoint_namespace": "open-swe-repair-v1",
        })
    )
    identity = RecoveryIdentity(
        operation_id=operation_id,
        execution_material_sha256="b" * 64,
        workspace=str(workspace.resolve()),
        task_id="task-1",
        unit_id="unit-1",
        session_id="session-1",
        allowed_paths=("a.py",),
        provider_id="google_genai",
        model_id="test-model",
        worker_identity_sha256="c" * 64,
        transport_config_sha256=cli._sha256("{}"),
        runtime_identity_sha256=runtime_identity,
    )
    journal = DurableOperationJournal(state_root, identity)
    journal.prepare()
    journal.ask_dispatching(turn_id="turn-1", prompt="repair", ordinal=0)
    journal.conversation_bound("conversation-1")
    cli._atomic_json(
        cli._operation_path({"runtime_state_root": str(state_root), "operation_id": operation_id}),
        {
            "schema": cli.RESULT_SCHEMA,
            "kind": "worker",
            "status": "OPEN_SWE_OUTCOME_UNKNOWN",
            "operation_id": operation_id,
            "directory": str(workspace.resolve()),
            "provider_id": "google_genai",
            "model_id": "test-model",
            "worker_identity_sha256": "c" * 64,
            "process_started": True,
            "outcome_unknown": True,
            "retry_safe": False,
        },
    )
    checkpoint_file = cli.checkpoint_path(state_root, operation_id)
    checkpoint_file.write_bytes(b"checkpoint")
    checkpoint_file.chmod(0o600)
    monkeypatch.setattr(cli, "validate_checkpoint", lambda _path: True)
    monkeypatch.setattr(cli, "create_checkpoint", lambda *_args: (object(), "open-swe-repair-v1"))
    request = {
        "schema": cli.REQUEST_SCHEMA,
        "operation": "worker_reconcile",
        "operation_id": operation_id,
        "runtime_state_root": str(state_root),
        "workspace_path": str(workspace),
        "provider_id": "google_genai",
        "model_id": "test-model",
        "worker_identity_sha256": "c" * 64,
        "transport_config": {},
    }
    malformed = (
        '{"type":"tool_call","name":"write_file","arguments":'
        '{"file_path":"a.py","content":"VALUE = ' + '"2"' + '\\n"}}'
    )
    repaired = (
        '{"type":"tool_call","name":"write_file","arguments":'
        '{"file_path":"a.py","content":"VALUE = \\"2\\"\\n"}}'
    )
    calls = {"repair": 0, "update": 0, "invoke": 0, "write": 0}

    class Model:
        _conversation_id = "conversation-1"

        def configure_recovery_journal(self, value):
            self.journal = value

        def _detail_response(self, *_args, **_kwargs):
            return malformed

        def _refresh_protocol_response(self, response, *, turn_id):
            return response

        _is_complete_protocol_response = staticmethod(
            OpenCLIWebChatModel._is_complete_protocol_response
        )
        _repair_matches_invalid_response = staticmethod(
            lambda invalid_response, repaired_response, _journal=None: (
                OpenCLIWebChatModel._repair_matches_invalid_response(
                    invalid_response, repaired_response, _journal
                )
            )
        )

        def _repair_protocol_response(self, response):
            calls["repair"] += 1
            turn_id = "turn_repair_once"
            self.journal.protocol_repair_started(
                origin=response, origin_sha256=cli._sha256(response), turn_id=turn_id
            )
            self.journal.ask_dispatching(turn_id=turn_id, prompt="repair", ordinal=1)
            self.journal.conversation_bound("conversation-repair")
            return repaired

        def _response_message(self, *_args):
            from langchain_core.messages import AIMessage

            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "a.py", "content": 'VALUE = "2"\n'},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )

    class Graph:
        def __init__(self, effect_journal):
            self.effect_journal = effect_journal

        def get_graph(self):
            return type(
                "G",
                (),
                {
                    "nodes": {
                        "tools": type("T", (), {"data": type("D", (), {"tools_by_name": {}})()})()
                    }
                },
            )()

        def update_state(self, *_args, **_kwargs):
            calls["update"] += 1

        def invoke(self, *_args, **_kwargs):
            calls["invoke"] += 1

            class Delegate:
                def write(_self, file_path, content):
                    calls["write"] += 1
                    target.write_text(content, encoding="utf-8")

            cli.ScopedRepairBackend(Delegate(), workspace, ("a.py",), self.effect_journal).write(
                "a.py", 'VALUE = "2"\n'
            )
            return _record("record_worker_result", {"summary": "recovered"})

    result = cli._worker_reconcile(
        request,
        runtime_loader=lambda: {},
        model_builder=lambda *_args: Model(),
        graph_builder=lambda *_args: Graph(_args[-1]),
    )
    assert result["status"] == "COMPLETED"
    assert calls == {"repair": 1, "update": 1, "invoke": 1, "write": 0}
    assert target.read_text(encoding="utf-8") == 'VALUE = "2"\n'
    effects = list((state_root / "recovery" / "effects").glob("*.json"))
    assert len(effects) == 1
    assert json.loads(effects[0].read_text(encoding="utf-8"))["status"] == "RESULT"


"""REMOVED_PENDING_TEST"""


def test_worker_reconcile_pending_repair_projects_locally_then_continues_original_conversation(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("old\n", encoding="utf-8")
    state_root = tmp_path / "state"
    operation_id = "h" * 64
    runtime_identity = cli._sha256(
        cli._canonical_json({
            "module_sha256": cli._sha256(Path(cli.__file__).read_bytes()),
            "deepagents": cli._deepagents_version(),
            "checkpoint_namespace": "open-swe-repair-v1",
        })
    )
    identity = RecoveryIdentity(
        operation_id=operation_id,
        execution_material_sha256="b" * 64,
        workspace=str(workspace.resolve()),
        task_id="task-1",
        unit_id="unit-1",
        session_id="session-1",
        allowed_paths=("a.py",),
        provider_id="opencli_chatgpt",
        model_id="balanced",
        worker_identity_sha256="c" * 64,
        transport_config_sha256=cli._sha256("{}"),
        runtime_identity_sha256=runtime_identity,
    )
    journal = DurableOperationJournal(state_root, identity)
    journal.prepare()
    journal.ask_dispatching(turn_id="turn_repair_1", prompt="repair", ordinal=1)
    journal.conversation_bound("conversation-old")
    origin = (
        '{"type":"tool_call","name":"write_file","arguments":'
        '{"file_path":"a.py","content":"new ' + '"v"' + '"}}'
    )
    journal.protocol_repair_started(
        origin=origin,
        origin_sha256="x" * 64,
        turn_id="turn_repair_1",
    )
    # Preserve the old conversation as the pre-repair binding and dispatch state.
    journal._transition(
        protocol_repair_origin_sha256=cli._sha256(origin),
        turn_id="turn_repair_1",
        status="ASK_DISPATCHING",
    )
    cli._atomic_json(
        cli._operation_path({"runtime_state_root": str(state_root), "operation_id": operation_id}),
        {
            "schema": cli.RESULT_SCHEMA,
            "kind": "worker",
            "status": "OPEN_SWE_OUTCOME_UNKNOWN",
            "operation_id": operation_id,
            "directory": str(workspace.resolve()),
            "provider_id": "opencli_chatgpt",
            "model_id": "balanced",
            "worker_identity_sha256": "c" * 64,
            "process_started": True,
            "outcome_unknown": True,
            "retry_safe": False,
        },
    )
    checkpoint_file = cli.checkpoint_path(state_root, operation_id)
    checkpoint_file.write_bytes(b"checkpoint")
    checkpoint_file.chmod(0o600)
    monkeypatch.setattr(cli, "validate_checkpoint", lambda _path: True)
    monkeypatch.setattr(cli, "create_checkpoint", lambda *_args: (object(), "open-swe-repair-v1"))
    request = {
        "schema": cli.REQUEST_SCHEMA,
        "operation": "worker_reconcile",
        "operation_id": operation_id,
        "runtime_state_root": str(state_root),
        "workspace_path": str(workspace),
        "provider_id": "opencli_chatgpt",
        "model_id": "balanced",
        "worker_identity_sha256": "c" * 64,
        "transport_config": {},
    }
    repaired = '{"type":"tool_call","name":"write_file","arguments":{"file_path":"a.py","content":"new \\"v\\""}}'
    counters = {"history": 0, "detail": 0, "repair": 0, "ask": 0}
    latest_prompt = json.dumps({"turn_id": "turn_repair_1"})

    model = OpenCLIWebChatModel(executable="/opt/opencli", intelligence_level="balanced")
    from langchain_core.messages import AIMessage

    model._response_message = lambda *_args: AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"file_path": "a.py", "content": 'new "v"'},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )

    class Graph:
        def __init__(self, model):
            self.model = model

        def get_graph(self):
            from langchain_core.tools import tool

            @tool
            def write_file(file_path: str, content: str) -> str:
                """Write one file."""
                return file_path + content

            return type(
                "G",
                (),
                {
                    "nodes": {
                        "tools": type(
                            "T",
                            (),
                            {
                                "data": type(
                                    "D", (), {"tools_by_name": {"write_file": write_file}}
                                )()
                            },
                        )()
                    }
                },
            )()

        def update_state(self, *_args, **_kwargs):
            pass

        def invoke(self, *_args, **_kwargs):
            self.model._send_and_reconcile('{"turn_id":"continuation"}')
            return _record("record_worker_result", {"summary": "terminal"})

    def fake_run(argv, **_kwargs):
        nonlocal latest_prompt
        args = list(argv)
        if args[1:3] == ["chatgpt", "history"]:
            counters["history"] += 1
            raise AssertionError("local projection must not scan history")
        if args[1:3] == ["chatgpt", "detail"]:
            counters["detail"] += 1
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([
                    {"Role": "User", "Text": latest_prompt, "Generating": False},
                    {"Role": "Assistant", "Text": repaired, "Generating": False},
                ]),
                stderr="",
            )
        if args[1:3] == ["chatgpt", "ask"]:
            counters["ask"] += 1
            latest_prompt = args[3]
            assert "--conversation" in args
            assert args[args.index("--conversation") + 1] == "conversation-old"
            assert "--new" not in args
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"conversationId": "conversation-old", "response": ""}]),
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr("nexus_open_swe_runtime.opencli_web_model.subprocess.run", fake_run)
    result = cli._worker_reconcile(
        request,
        runtime_loader=lambda: {},
        model_builder=lambda *_args: model,
        graph_builder=lambda model, *_args: Graph(model),
    )
    assert result["status"] == "COMPLETED"
    assert counters == {"history": 0, "detail": 1, "repair": 0, "ask": 1}


def test_fresh_repair_timeout_with_non_projectable_pending_origin_uses_history_fallback(
    monkeypatch,
):
    origin = (
        '{"type":"tool_call","name":"write_file","arguments":'
        '{"file_path":"a.py","content":"new "quoted""}}'
    )

    class Journal:
        def __init__(self):
            self.state = {
                "protocol_repair_origin": origin,
                "protocol_repair_origin_sha256": "f" * 64,
                "protocol_repair_original_conversation_id": "conversation-original",
                "protocol_repair_status": "DISPATCHING",
            }

        def read(self):
            return dict(self.state)

        def conversation_bound(self, conversation_id):
            self.state["conversation_id"] = conversation_id

        def response_recovered(self, _turn_id, _response):
            pass

        def protocol_repair_recovered(self, _response):
            pass

    journal = Journal()
    model = OpenCLIWebChatModel(executable="/opt/opencli")
    model._recovery_journal = journal
    model._conversation_id = "conversation-original"
    calls = []
    detail = json.dumps([
        {"Role": "User", "Text": json.dumps({"turn_id": "turn_repair_fallback"})},
    ])
    repaired = '{"type":"final","content":"fallback"}'

    def fake_run(argv):
        calls.append(list(argv))
        if argv[1:3] == ["chatgpt", "history"]:
            return json.dumps([{"Id": "conversation-original"}, {"Id": "conversation-repair"}])
        if argv[1:3] == ["chatgpt", "detail"] and argv[3] == "conversation-repair":
            return detail
        raise AssertionError(argv)

    monkeypatch.setattr(model, "_run", fake_run)
    monkeypatch.setattr(model, "_detail_response", lambda *_args, **_kwargs: repaired)

    assert (
        model._reconcile_fresh_repair_timeout(
            "turn_repair_fallback", "conversation-original", "resume"
        )
        == repaired
    )
    assert [call[2] for call in calls] == ["history", "detail"]
    assert calls[1][3] == "conversation-repair"
    assert model._conversation_id == "conversation-repair"


def test_worker_run_operation_fence_allows_one_initial_external_execution(tmp_path):
    request = _worker_request(tmp_path)
    workspace = Path(request["workspace_path"])
    diagnosis = FakeGraph(
        cli.DIAGNOSIS_TOOLS,
        _record(
            "record_diagnosis",
            {
                "status": "ROOT_CAUSE_SUPPORTED",
                "summary": "supported",
                "evidence_paths": ["a.py"],
            },
        ),
    )
    calls = {"models": 0, "effects": 0}

    class Model:
        def configure_recovery_journal(self, journal):
            self.journal = journal

    class Repair(FakeGraph):
        def invoke(self, payload, config=None):
            time.sleep(0.05)
            calls["effects"] += 1
            (workspace / "a.py").write_text("VALUE = 2\n", encoding="utf-8")
            return super().invoke(payload, config)

    repair = Repair(
        cli.REPAIR_TOOLS,
        _record("record_worker_result", {"summary": "repaired"}),
    )

    def model_factory(*_args):
        calls["models"] += 1
        return Model()

    def run():
        return cli._worker_run(
            request,
            runtime_loader=_runtime,
            model_factory=model_factory,
            diagnosis_factory=lambda *_args: diagnosis,
            repair_factory=lambda *_args: repair,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(lambda _ignored: run(), range(2))
    assert first["status"] == second["status"] == "COMPLETED"
    assert calls == {"models": 2, "effects": 1}
    assert diagnosis.calls == 1
    assert repair.calls == 1
    assert workspace.joinpath("a.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_restart_trace_uses_real_opencli_conversation_for_continuation(tmp_path, monkeypatch):
    state_root = tmp_path / "state"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    identity = RecoveryIdentity(
        operation_id="s" * 64,
        execution_material_sha256="a" * 64,
        workspace=str(workspace.resolve()),
        task_id="task-1",
        unit_id="unit-1",
        session_id="session-1",
        allowed_paths=("a.py",),
        provider_id="opencli_chatgpt",
        model_id="balanced",
        worker_identity_sha256="b" * 64,
        transport_config_sha256=cli._sha256("{}"),
        runtime_identity_sha256="c" * 64,
    )
    journal = DurableOperationJournal(state_root, identity)
    journal.prepare()
    calls = []
    ask_count = 0
    latest_prompt = ""

    def fake_run(argv, **kwargs):
        nonlocal ask_count, latest_prompt
        args = list(argv)
        calls.append((args, dict(kwargs)))
        if args[1:3] == ["chatgpt", "model"]:
            return SimpleNamespace(returncode=0, stdout='[{"Status":"ok"}]', stderr="")
        if args[1:3] == ["chatgpt", "ask"]:
            ask_count += 1
            latest_prompt = args[3]
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"conversationId": "conversation-1", "response": ""}]),
                stderr="",
            )
        if args[1:3] == ["chatgpt", "detail"]:
            response = (
                '{"type":"tool_call","name":"write_file","arguments":'
                '{"file_path":"a.py","content":"done\\n"}}'
                if ask_count == 1
                else '{"type":"final","content":"continued"}'
            )
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

    monkeypatch.setattr("nexus_open_swe_runtime.opencli_web_model.subprocess.run", fake_run)
    first_model = OpenCLIWebChatModel(executable="/opt/opencli", intelligence_level="balanced")
    first_model.configure_recovery_journal(journal)
    tools = [
        {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object"}}}
    ]
    first = (
        first_model._generate([HumanMessage(content="initial")], tools=tools).generations[0].message
    )
    assert first.tool_calls[0]["id"]
    assert journal.read()["conversation_id"] == "conversation-1"

    restarted = OpenCLIWebChatModel(executable="/opt/opencli", intelligence_level="balanced")
    restarted.configure_recovery_journal(journal)
    assert restarted._detail_response is not None
    recovered = cli.reconcile_bound_turn(journal, restarted)
    assert json.loads(recovered)["type"] == "tool_call"
    continuation = (
        restarted
        ._generate(
            [
                first,
                ToolMessage(
                    content="written", tool_call_id=first.tool_calls[0]["id"], name="write_file"
                ),
            ],
            tools=tools,
        )
        .generations[0]
        .message
    )
    assert continuation.content == "continued"
    asks = [argv for argv, _kwargs in calls if argv[1:3] == ["chatgpt", "ask"]]
    assert len(asks) == 2
    assert sum("--new" in argv for argv in asks) == 1
    assert "--conversation" not in asks[0]
    assert asks[1][asks[1].index("--conversation") + 1] == "conversation-1"
    assert "--new" not in asks[1]


def test_worker_replay_with_changed_material_does_not_return_cached_terminal_or_redispatch(
    tmp_path,
):
    request = _worker_request(tmp_path)
    workspace = Path(request["workspace_path"])
    operation_id = request["operation_id"]
    terminal = {
        "schema": cli.RESULT_SCHEMA,
        "kind": "worker",
        "status": "COMPLETED",
        "operation_id": operation_id,
        "directory": str(workspace.resolve()),
        "provider_id": request["provider_id"],
        "model_id": request["model_id"],
        "worker_identity_sha256": request["worker_identity_sha256"],
        "process_started": True,
        "outcome_unknown": False,
        "retry_safe": False,
    }
    cli._atomic_json(
        Path(request["runtime_state_root"]) / "operations" / f"{operation_id}.json",
        terminal,
    )
    changed = dict(request)
    changed["workspace_path"] = str(tmp_path / "changed-workspace")
    graph = FakeGraph(cli.REPAIR_TOOLS)

    result = cli._worker_run(
        changed,
        runtime_loader=_runtime,
        model_factory=lambda *_args: object(),
        diagnosis_factory=lambda *_args: graph,
        repair_factory=lambda *_args: graph,
    )

    assert result["operation_id"] == operation_id
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert result["outcome_unknown"] is True
    assert graph.calls == 0


@pytest.mark.parametrize(
    "material",
    [
        "prompt",
        "artifact",
        "session",
        "operation",
        "provider_id",
        "model_id",
        "worker_identity_sha256",
    ],
)
def test_worker_replay_changed_execution_material_fails_closed(tmp_path, material):
    request = _worker_request(tmp_path)
    started = cli._write_started(request, "worker")
    terminal = {**started, "status": "COMPLETED", "finished_at": cli._now()}
    cli._atomic_json(
        Path(request["runtime_state_root"]) / "operations" / f"{request['operation_id']}.json",
        terminal,
    )
    changed = dict(request)
    if material == "prompt":
        changed["prompt"] = request["prompt"] + " changed"
    elif material == "artifact":
        Path(request["artifact_path"]).write_text('{"failure":"changed"}\n', encoding="utf-8")
    elif material == "session":
        changed["session_id"] = "ses_open_swe_changed"
    elif material == "operation":
        changed["operation"] = "worker_continue"
    elif material == "provider_id":
        changed["provider_id"] = ""
    elif material == "model_id":
        changed["model_id"] = ""
    else:
        changed["worker_identity_sha256"] = ""

    graph = FakeGraph(cli.REPAIR_TOOLS)
    result = cli._worker_run(
        changed,
        runtime_loader=_runtime,
        model_factory=lambda *_args: object(),
        diagnosis_factory=lambda *_args: graph,
        repair_factory=lambda *_args: graph,
    )

    assert result["operation_id"] == request["operation_id"]
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert result["outcome_unknown"] is True
    assert graph.calls == 0


def test_scoped_repair_backend_rejects_out_of_scope_write(tmp_path):
    class Delegate:
        def write(self, file_path, content):
            target = tmp_path / file_path.lstrip("/")
            target.write_text(content, encoding="utf-8")
            return None

    scoped = cli.ScopedRepairBackend(Delegate(), tmp_path, ("a.py",))
    with pytest.raises(PermissionError, match="OPEN_SWE_MUTATION_PATH_FORBIDDEN"):
        scoped.write("b.py", "forbidden\n")
    assert not (tmp_path / "b.py").exists()
    scoped.write("a.py", "allowed\n")
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "allowed\n"


@pytest.mark.parametrize(
    ("content", "old_string", "replace_all", "error_fragment"),
    [
        ("alpha\n", "missing", False, "String not found"),
        ("same\nsame\n", "same", False, "appears 2 times"),
    ],
)
def test_scoped_repair_backend_edit_recovery_preserves_replace_contract(
    tmp_path, content, old_string, replace_all, error_fragment
):
    target = tmp_path / "a.py"
    target.write_text(content, encoding="utf-8")
    identity = RecoveryIdentity(
        operation_id="o" * 64,
        execution_material_sha256="e" * 64,
        workspace=str(tmp_path.resolve()),
        task_id="task-1",
        unit_id="unit-1",
        session_id="session-1",
        allowed_paths=("a.py",),
        provider_id="test",
        model_id="test",
        worker_identity_sha256="w" * 64,
        transport_config_sha256="t" * 64,
        runtime_identity_sha256="r" * 64,
    )
    journal = cli.DurableEffectJournal(tmp_path, identity)
    journal.bind_turn("turn-1", "call-1")
    backend = cli.ScopedRepairBackend(object(), tmp_path, ("a.py",), journal)

    result = backend.edit("a.py", old_string, "replacement", replace_all=replace_all)

    assert result.error is not None and error_fragment in result.error
    assert target.read_text(encoding="utf-8") == content
    assert list(journal.root.glob("*.json")) == []


def test_real_deepagents_graphs_expose_only_qualified_surfaces(tmp_path):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class SurfaceModel(BaseChatModel):
        model_name: str = "surface"

        @property
        def _llm_type(self):
            return "external-open-swe-surface"

        def _get_ls_params(self, *args, **kwargs):
            return {"ls_provider": "test", "ls_model_name": self.model_name}

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])

    runtime = cli._load_runtime()
    model = SurfaceModel()
    semantic = cli.build_semantic_graph(model, tmp_path, runtime, "test:surface")
    diagnosis = cli.build_diagnosis_graph(model, tmp_path, runtime, "test:surface")
    repair = cli.build_repair_graph(model, tmp_path, runtime, ("a.py",), "test:surface")

    assert set(cli.executable_tool_surface(semantic)) == cli.SEMANTIC_TOOLS
    assert set(cli.executable_tool_surface(diagnosis)) == cli.DIAGNOSIS_TOOLS
    assert set(cli.executable_tool_surface(repair)) == cli.REPAIR_TOOLS
    forbidden = {"execute", "shell", "task", "delete_file", "http_request", "git_push"}
    assert forbidden.isdisjoint(cli.executable_tool_surface(semantic))
    assert forbidden.isdisjoint(cli.executable_tool_surface(diagnosis))
    assert forbidden.isdisjoint(cli.executable_tool_surface(repair))


def test_admitted_composite_graph_writes_receipt_and_terminates_locally(tmp_path, monkeypatch):
    """The compiled admitted graph performs one composite effect and one web send."""
    runtime = cli._load_runtime()
    model = OpenCLIWebChatModel(executable="/opt/opencli")
    identity = RecoveryIdentity(
        operation_id="o" * 64,
        execution_material_sha256="e" * 64,
        workspace=str(tmp_path.resolve()),
        task_id="task-1",
        unit_id="unit-1",
        session_id="session-1",
        allowed_paths=("a.py",),
        provider_id="opencli_chatgpt",
        model_id="advanced",
        worker_identity_sha256="w" * 64,
        transport_config_sha256="t" * 64,
        runtime_identity_sha256="r" * 64,
        composite_admitted=True,
    )
    journal = cli.DurableOperationJournal(tmp_path, identity)
    journal.prepare()
    effect_journal = cli.DurableEffectJournal(tmp_path, identity)
    journal.effect_journal = effect_journal
    model.configure_recovery_journal(journal)
    sends: list[str] = []

    def one_model_send(prompt, **_kwargs):
        sends.append(prompt)
        turn_id = json.loads(prompt)["turn_id"]
        model._journal_ask(turn_id, prompt)
        model._journal_bound("conversation-1")
        args = {
            "file_path": "a.py",
            "content": "VALUE = 2\n",
            "envelope": {"summary": "repaired a.py"},
        }
        response = json.dumps(
            {
                "type": "tool_call",
                "name": "write_file_and_record_worker_result",
                "arguments": args,
            },
            separators=(",", ":"),
        )
        model._journal_response(turn_id, response)
        return response

    monkeypatch.setattr(model, "_send_and_reconcile", one_model_send)
    monkeypatch.setattr(model, "_select_intelligence_level", lambda: None)
    graph = cli.build_repair_graph(
        model,
        tmp_path,
        runtime,
        ("a.py",),
        "opencli_chatgpt:advanced",
        effect_journal=effect_journal,
        composite=True,
    )
    assert "write_file_and_record_worker_result" in cli.executable_tool_surface(graph)

    result = graph.invoke(
        {"messages": [runtime["human_message"]("repair a.py")]},
        config={"configurable": {"thread_id": identity.operation_id}, "recursion_limit": 60},
    )

    assert len(sends) == 1
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    effects = list(effect_journal.root.glob("effect_*.json"))
    assert len(effects) == 1
    effect = json.loads(effects[0].read_text(encoding="utf-8"))
    assert effect["status"] == "RESULT"
    receipt = effect["worker_result"]
    assert receipt["schema"] == "nexus.open_swe_runtime.worker_result.v1"
    assert receipt["status"] == "IMPLEMENTATION_EFFECT_COMPLETE"
    assert result["messages"][-1].content == "Terminal recorder completed."


@pytest.mark.parametrize(
    "allowed_paths, composite",
    [((), False), (("a.py", "b.py"), False)],
)
def test_fallback_and_multipath_graphs_do_not_expose_composite(tmp_path, allowed_paths, composite):
    runtime = cli._load_runtime()
    model = OpenCLIWebChatModel(executable="/opt/opencli")
    graph = cli.build_repair_graph(
        model,
        tmp_path,
        runtime,
        allowed_paths,
        "test:unqualified",
        composite=composite,
    )
    assert "write_file_and_record_worker_result" not in cli.executable_tool_surface(graph)


def test_runtime_state_files_are_restrictive_and_contain_no_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sentinel-provider-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "sentinel-github-secret")
    request = _semantic_request(tmp_path)
    graph = FakeGraph(
        cli.SEMANTIC_TOOLS,
        _record("record_finding", {"schema": "external_execution_envelope.v1", "binding": {}}),
    )
    cli._semantic_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: object(),
        graph_factory=lambda *_args: graph,
    )
    state_file = (
        Path(request["runtime_state_root"]) / "operations" / f"{request['operation_id']}.json"
    )
    text = state_file.read_text(encoding="utf-8")
    assert "GEMINI_API_KEY" not in text
    assert "GITHUB_TOKEN" not in text
    assert "GH_TOKEN" not in text
    assert "sentinel-provider-secret" not in text
    assert "sentinel-github-secret" not in text
    assert state_file.stat().st_mode & 0o077 == 0


def test_runtime_identity_dispatch():
    request = {
        "schema": cli.REQUEST_SCHEMA,
        "operation": "identity",
    }
    result = cli.dispatch(request)
    assert result["schema"] == cli.RESULT_SCHEMA
    assert result["kind"] == "identity"
    assert result["status"] == "IDENTIFIED"
    assert result["distribution_name"] == "nexus-open-swe-runtime"
    assert "distribution_version" in result
    assert result["runtime_protocol_version"] == cli.REQUEST_SCHEMA
    assert result["authority_boundary"] == "execution_runtime_only"
    assert result["process_started"] is False
    assert result["outcome_unknown"] is False
    assert result["retry_safe"] is True
    assert "artifact_identity" in result
    assert "module_file" in result["artifact_identity"]
    assert "module_sha256" in result["artifact_identity"]


def test_protocol_schema_mismatch_fails_closed():
    invalid_request = {
        "schema": "nexus.open_swe_runtime.request.v0_legacy",
        "operation": "identity",
    }
    with pytest.raises(cli.RuntimeErrorBounded, match="OPEN_SWE_PROTOCOL_SCHEMA_INVALID"):
        cli.dispatch(invalid_request)


def test_persisted_operation_state_preserves_identity_across_restart(tmp_path: Path):
    state_root = tmp_path / "runtime_state"
    op_id = "f" * 64
    request = {
        "schema": cli.REQUEST_SCHEMA,
        "operation": "semantic_run",
        "operation_id": op_id,
        "provider_id": "google_genai",
        "model_id": "gemini-test",
        "repository_root": str(tmp_path),
        "runtime_state_root": str(state_root),
        "prompt": "inspect repo",
    }
    graph = FakeGraph(
        cli.SEMANTIC_TOOLS,
        _record("record_finding", {"schema": "external_execution_envelope.v1", "binding": {}}),
    )
    res1 = cli._semantic_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: object(),
        graph_factory=lambda *_args: graph,
    )
    assert res1["status"] == "INTELLIGENCE_COMPLETED"
    assert graph.calls == 1

    # Simulate restart / upgrade: new dispatch with reconcile
    reconcile_req = {
        "schema": cli.REQUEST_SCHEMA,
        "operation": "semantic_reconcile",
        "operation_id": op_id,
        "runtime_state_root": str(state_root),
    }
    res2 = cli.dispatch(reconcile_req)
    assert res2 == res1
    assert graph.calls == 1  # No re-dispatch


def test_child_environment_containment_excludes_controller_and_github_tokens(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "leak-github-token")
    monkeypatch.setenv("GH_TOKEN", "leak-gh-token")
    monkeypatch.setenv("CONTROLLER_TOKEN", "leak-controller-token")
    monkeypatch.setenv("OPENCLI_ALLOWED", "kept-value")

    from nexus_open_swe_runtime.opencli_web_model import OpenCLIWebChatModel

    model = OpenCLIWebChatModel(model="chatgpt")
    env = model._environment()
    assert "GITHUB_TOKEN" not in env
    assert "GH_TOKEN" not in env
    assert "CONTROLLER_TOKEN" not in env
    assert "leak-github-token" not in env.values()
    assert "leak-gh-token" not in env.values()
    assert env.get("OPENCLI_ALLOWED") == "kept-value"


def test_semantic_ambiguous_timeout_hostile_reconciliation_no_second_send(tmp_path: Path):
    """Verify that a timeout or lost ack marks outcome_unknown=True, retry_safe=False,
    and hostile retry/reconciliation NEVER causes a second semantic send.
    Later authoritative result is read back idempotently without duplicate send.
    """
    request = _semantic_request(tmp_path)
    op_id = request["operation_id"]
    state_root = Path(request["runtime_state_root"])

    # Simulate timeout during graph invoke
    timeout_graph = FakeGraph(cli.SEMANTIC_TOOLS, error=TimeoutError("remote gateway timeout"))

    res1 = cli._semantic_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: object(),
        graph_factory=lambda *_args: timeout_graph,
    )

    # 1. First send resulted in timeout error during processing
    assert timeout_graph.calls == 1
    assert res1["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert res1["outcome_unknown"] is True
    assert res1["retry_safe"] is False
    assert res1["process_started"] is True
    assert res1["operation_id"] == op_id

    # 2. Hostile resend with semantic_run on same operation_id MUST NOT dispatch a second send
    res2 = cli._semantic_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: object(),
        graph_factory=lambda *_args: timeout_graph,
    )
    assert timeout_graph.calls == 1  # ZERO additional send
    assert res2["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert res2["outcome_unknown"] is True
    assert res2["retry_safe"] is False

    # 3. Hostile reconcile MUST NOT dispatch a second send
    reconcile_req = {
        "schema": cli.REQUEST_SCHEMA,
        "operation": "semantic_reconcile",
        "operation_id": op_id,
        "runtime_state_root": str(state_root),
    }
    res3 = cli.dispatch(reconcile_req)
    assert timeout_graph.calls == 1  # Still 1
    assert res3["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert res3["outcome_unknown"] is True
    assert res3["retry_safe"] is False

    # 4. Later authoritative state discovery writes terminal result
    authoritative_payload = {
        **res1,
        "status": "INTELLIGENCE_COMPLETED",
        "raw": cli._canonical_json({"finding": "authoritative_completed"}),
        "outcome_unknown": False,
        "finished_at": cli._now(),
    }
    cli._atomic_json(state_root / "operations" / f"{op_id}.json", authoritative_payload)

    # 5. Subsequent reconcile seamlessly transitions without sending again
    res4 = cli.dispatch(reconcile_req)
    assert timeout_graph.calls == 1  # Still exactly 1
    assert res4["status"] == "INTELLIGENCE_COMPLETED"
    assert res4["outcome_unknown"] is False
    assert res4["operation_id"] == op_id


def test_legacy_client_contract_frozen_fixture(tmp_path: Path):
    """Verify repository-local contract compatibility with frozen client request fixture.
    Fixture origin: Nexus-new revision, nexus/services/open_swe_external_intelligence.py.
    Protocol: nexus.open_swe_runtime.request.v1 / nexus.open_swe_runtime.result.v1.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("print('hello')\n", encoding="utf-8")
    state_root = tmp_path / "state"

    # Frozen client-shaped request as constructed by Nexus-new OpenSWEExternalIntelligenceTransport
    frozen_client_request = {
        "schema": "nexus.open_swe_runtime.request.v1",
        "operation": "semantic_run",
        "operation_id": "e" * 64,
        "provider_id": "google_genai",
        "model_id": "gemini-test",
        "repository_root": str(repo),
        "runtime_state_root": str(state_root),
        "prompt": "Find all print calls in sample.py",
        "transport_config": None,
    }

    graph = FakeGraph(
        cli.SEMANTIC_TOOLS,
        _record(
            "record_finding",
            {
                "schema": "external_execution_envelope.v1",
                "findings": [{"file": "sample.py", "line": 1}],
            },
        ),
    )

    result = cli._semantic_run(
        frozen_client_request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: object(),
        graph_factory=lambda *_args: graph,
    )

    # Assert exact v1 result schema contract expected by Nexus-new
    assert result["schema"] == "nexus.open_swe_runtime.result.v1"
    assert result["kind"] == "semantic"
    assert result["status"] == "INTELLIGENCE_COMPLETED"
    assert result["operation_id"] == "e" * 64
    assert result["process_started"] is True
    assert result["outcome_unknown"] is False
    assert result["retry_safe"] is False
    assert "raw" in result
    envelope = json.loads(result["raw"])
    assert envelope["schema"] == "external_execution_envelope.v1"


def test_worker_admits_repair_when_absent_create_target_matches_authorized_mutation_path(tmp_path):
    request = _worker_request(tmp_path)
    workspace = Path(request["workspace_path"])
    target_rel = "tests/ops/new_canary.py"
    target = workspace / target_rel
    assert not target.exists()

    request["prompt"] = "\n".join([
        "task_id=task-1",
        "unit_id=u1",
        f'authorized_mutation_paths=["{target_rel}"]',
        "bounded repair create target",
    ])

    diagnosis = FakeGraph(
        cli.DIAGNOSIS_TOOLS,
        _record(
            "record_diagnosis",
            {
                "status": "ROOT_CAUSE_SUPPORTED",
                "summary": "absent create target canary file",
                "evidence_paths": [f"/{target_rel}"],
            },
        ),
    )

    def _create_target():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("CANARY = True\n", encoding="utf-8")

    repair = FakeGraph(
        cli.REPAIR_TOOLS,
        _record("record_worker_result", {"summary": "created canary file"}),
        effect=_create_target,
    )

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: object(),
        diagnosis_factory=lambda *_args: diagnosis,
        repair_factory=lambda *_args: repair,
    )

    assert result["status"] == "COMPLETED"
    assert result["diagnosis_status"] == "ROOT_CAUSE_SUPPORTED"
    assert result["repair_admitted"] is True
    assert result["repair_phase_count"] == 1
    assert target.read_text(encoding="utf-8") == "CANARY = True\n"
    assert diagnosis.calls == 1
    assert repair.calls == 1


def test_worker_blocks_repair_when_absent_evidence_path_outside_authorized_mutation_paths(tmp_path):
    request = _worker_request(tmp_path)
    workspace = Path(request["workspace_path"])
    initial_files = {
        p.relative_to(workspace): p.read_bytes() for p in workspace.rglob("*") if p.is_file()
    }
    unauthorized_rel = "tests/ops/unauthorized_absent.py"
    target = workspace / unauthorized_rel
    assert not target.exists()

    request["prompt"] = "\n".join([
        "task_id=task-1",
        "unit_id=u1",
        'authorized_mutation_paths=["tests/ops/new_canary.py"]',
        "bounded repair unauthorized",
    ])

    diagnosis = FakeGraph(
        cli.DIAGNOSIS_TOOLS,
        _record(
            "record_diagnosis",
            {
                "status": "ROOT_CAUSE_SUPPORTED",
                "summary": "unauthorized absent file",
                "evidence_paths": [f"/{unauthorized_rel}"],
            },
        ),
    )

    def _create_unauthorized():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("SHOULD_NOT_EXIST = True\n", encoding="utf-8")

    repair = FakeGraph(
        cli.REPAIR_TOOLS,
        _record("record_worker_result", {"summary": "should not be called"}),
        effect=_create_unauthorized,
    )

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: object(),
        diagnosis_factory=lambda *_args: diagnosis,
        repair_factory=lambda *_args: repair,
    )

    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert result["outcome_unknown"] is True
    assert result["repair_admitted"] is False
    assert result["repair_phase_count"] == 0
    assert repair.calls == 0
    assert diagnosis.calls == 1
    current_files = {
        p.relative_to(workspace): p.read_bytes() for p in workspace.rglob("*") if p.is_file()
    }
    assert current_files == initial_files
    assert not target.exists()


def test_worker_blocks_repair_when_authorized_absent_path_is_dangling_symlink(tmp_path):
    request = _worker_request(tmp_path)
    workspace = Path(request["workspace_path"])
    target_rel = "tests/ops/new_canary.py"
    target = workspace / target_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to("missing-target.py")
    missing_target = target.parent / "missing-target.py"
    assert target.is_symlink()
    assert not target.exists()

    request["prompt"] = "\n".join([
        "task_id=task-1",
        "unit_id=u1",
        f'authorized_mutation_paths=["{target_rel}"]',
        "bounded repair dangling symlink",
    ])

    diagnosis = FakeGraph(
        cli.DIAGNOSIS_TOOLS,
        _record(
            "record_diagnosis",
            {
                "status": "ROOT_CAUSE_SUPPORTED",
                "summary": "authorized path is a dangling symlink",
                "evidence_paths": [f"/{target_rel}"],
            },
        ),
    )
    repair = FakeGraph(
        cli.REPAIR_TOOLS,
        _record("record_worker_result", {"summary": "must not run"}),
        effect=lambda: missing_target.write_text("SHOULD_NOT_EXIST = True\n", encoding="utf-8"),
    )

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: object(),
        diagnosis_factory=lambda *_args: diagnosis,
        repair_factory=lambda *_args: repair,
    )

    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert result["outcome_unknown"] is True
    assert result["repair_admitted"] is False
    assert result["repair_phase_count"] == 0
    assert diagnosis.calls == 1
    assert repair.calls == 0
    assert target.is_symlink()
    assert target.readlink() == Path("missing-target.py")
    assert not missing_target.exists()


def _v2_request(tmp_path: Path, *, status: str = "PROVEN") -> dict:
    request = _worker_request(tmp_path)
    request.update({
        "provider_id": "opencli_chatgpt",
        "model_id": "advanced",
        "transport_config": {
            "executable": "/usr/bin/opencli",
            "profile": "r16",
            "site_session": "ephemeral",
            "timeout_seconds": 30,
        },
    })
    workspace = Path(request["workspace_path"])
    (workspace / "a.py").unlink()
    card_ref = "tasks/task-card.md"
    card = workspace / card_ref
    card.parent.mkdir(parents=True)
    card.write_text(
        "\n".join([
            "- task_id: `task-1`",
            "- status: `ACTIVE`",
            "- worker_may_approve: `false`",
            "- worker_may_integrate: `false`",
            "- worker_may_push: `false`",
            "- AUTO_CHAIN: `false`",
            "- allow_deletions: `false`",
            "\n## Allowed files\n",
            "- `a.py`",
        ]),
        encoding="utf-8",
    )
    card_hash = cli._sha256(card.read_bytes())
    envelope = {
        "binding": {
            "context_pack_sha256": "c" * 64,
            "item_id": "task-1",
            "item_type": "issue",
            "main_sha": "b" * 40,
            "repository": "James3014/Nexus-new",
            "revision": "r16",
            "task_card_ref": card_ref,
            "task_card_hash": card_hash,
        },
        "diagnosis": {
            "status": "UNKNOWN" if status == "INCONCLUSIVE" else status,
            "hypothesis": "the required target is absent",
            "next_probe": "inspect the absent target",
        },
        "definition_of_done": ["one bounded test"],
        "evidence_refs": [
            f"task_card:{card_ref}@" + "a" * 16,
            "source_absence:a.py@" + "b" * 16,
        ],
        "inspect_first": [
            f"task_card:{card_ref}@" + "a" * 16,
            "source_absence:a.py@" + "b" * 16,
        ],
        "failure_guards": ["no other paths"],
        "implementation_direction": ["create one test"],
        "objective": "bounded repair",
        "required_semantics": ["one deterministic test"],
        "schema": "external_execution_envelope.v2",
        "scope_signal": {
            "conditional_migration_paths": [],
            "forbidden_paths": [],
            "max_files": 1,
            "production_edit_paths": [],
            "read_only_authorities": [card_ref],
            "required_test_edit_paths": ["a.py"],
            "scope_block_conditions": ["no extra files"],
            "scope_confidence": "HIGH",
            "verification_only_paths": ["a.py"],
        },
        "selected_worker": {
            "admission_evidence_hash": "a" * 64,
            "admission_evidence_ref": "admission.json",
            "model": request["model_id"],
            "provider": request["provider_id"],
            "role_ceiling": "bounded_candidate_generation",
            "selection_evidence_hash": "d" * 64,
            "selection_evidence_ref": "selection.json",
            "worker_id": "worker-1",
        },
        "stop_and_escalate": ["stop"],
        "verification_focus": ["pytest"],
    }
    raw = cli._canonical_json(envelope)
    Path(request["artifact_path"]).write_text(raw, encoding="utf-8")
    selected = envelope["selected_worker"]
    request["worker_identity"] = dict(selected)
    request["worker_identity_sha256"] = cli._sha256(cli._canonical_json(selected))
    request["prompt"] += "\n" + "\n".join([
        f"envelope_sha256={cli._sha256(raw)}",
        "expected_base_sha=" + "b" * 40,
    ])
    return request


def test_worker_admits_strict_v2_without_diagnosis_model_or_graph(tmp_path, monkeypatch):
    request = _v2_request(tmp_path)
    monkeypatch.setattr(
        cli,
        "_git_output",
        lambda _workspace, *args: {
            ("rev-parse", "HEAD"): "b" * 40,
            ("status", "--porcelain"): "",
            ("remote", "get-url", "origin"): "git@github.com:James3014/Nexus-new.git",
        }[args],
    )
    model_calls = 0
    diagnosis_calls = 0
    repair = FakeGraph(cli.REPAIR_TOOLS, _record("record_worker_result", {"summary": "done"}))

    def model_factory(_runtime, provider, model_id, transport_config, _state_root):
        nonlocal model_calls
        model_calls += 1
        assert provider == "opencli_chatgpt"
        assert model_id == "advanced"
        assert transport_config["site_session"] == "ephemeral"
        return SimpleNamespace(_conversation_id=None)

    def diagnosis_factory(*_args):
        nonlocal diagnosis_calls
        diagnosis_calls += 1
        raise AssertionError("admitted v2 must skip diagnosis")

    def repair_factory(model, *_args):
        assert model._conversation_id is None
        return repair

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=model_factory,
        diagnosis_factory=diagnosis_factory,
        repair_factory=repair_factory,
    )

    assert result["status"] == "COMPLETED"
    assert result["diagnosis_status"] == "ROOT_CAUSE_SUPPORTED"
    assert result["repair_admitted"] is True
    assert model_calls == 1
    assert diagnosis_calls == 0
    assert repair.calls == 1


def test_r24_issue_metadata_authority_is_admitted(tmp_path, monkeypatch):
    request = _v2_request(tmp_path)
    envelope = json.loads(Path(request["artifact_path"]).read_text(encoding="utf-8"))
    envelope["binding"].update(item_id="853", item_type="issue")
    envelope["scope_signal"]["read_only_authorities"].append("GitHub Issue 853 binding metadata")
    envelope["evidence_refs"].append(
        "github_issue:github://James3014/Nexus-new/issues/853@" + "e" * 16
    )
    _write_v2_mutation(request, envelope)
    monkeypatch.setattr(
        cli,
        "_git_output",
        lambda _workspace, *args: {
            ("rev-parse", "HEAD"): "b" * 40,
            ("status", "--porcelain"): "",
            ("remote", "get-url", "origin"): "git@github.com:James3014/Nexus-new.git",
        }[args],
    )
    admission = cli._semantic_v2_admission(
        request,
        Path(request["workspace_path"]),
        Path(request["artifact_path"]),
        request["prompt"],
        ("a.py",),
    )
    assert admission.decision == cli.ADMIT, admission


def test_r24_worker_selects_composite_terminal_path(tmp_path, monkeypatch):
    request = _v2_request(tmp_path)
    envelope = json.loads(Path(request["artifact_path"]).read_text(encoding="utf-8"))
    envelope["binding"].update(item_id="853", item_type="issue")
    envelope["scope_signal"]["read_only_authorities"].append("GitHub Issue 853 binding metadata")
    envelope["evidence_refs"].append(
        "github_issue:github://James3014/Nexus-new/issues/853@" + "e" * 16
    )
    _write_v2_mutation(request, envelope)
    monkeypatch.setattr(
        cli,
        "_git_output",
        lambda _workspace, *args: {
            ("rev-parse", "HEAD"): "b" * 40,
            ("status", "--porcelain"): "",
            ("remote", "get-url", "origin"): "git@github.com:James3014/Nexus-new.git",
        }[args],
    )
    captured: dict[str, object] = {}
    original_admission = cli._semantic_v2_admission

    def admission_probe(*args, **kwargs):
        admission = original_admission(*args, **kwargs)
        captured["admitted"] = admission.decision == cli.ADMIT
        captured["scope"] = admission.envelope["scope_signal"]
        captured["allowed"] = args[4]
        return admission

    monkeypatch.setattr(cli, "_semantic_v2_admission", admission_probe)

    def repair_factory(
        _model, _workspace, _runtime, _allowed_paths, _key, *, composite=False, **_kwargs
    ):
        return FakeGraph(
            cli.REPAIR_TOOLS | {"write_file_and_record_worker_result"},
            error=RuntimeError("terminal-path probe"),
        )

    original_repair_graph = cli._repair_graph

    def repair_probe(factory, *args, **kwargs):
        captured["composite"] = kwargs.get("composite")
        return original_repair_graph(factory, *args, **kwargs)

    monkeypatch.setattr(cli, "_repair_graph", repair_probe)

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: SimpleNamespace(_conversation_id=None),
        diagnosis_factory=lambda *_args: pytest.fail("admitted r24 must skip diagnosis"),
        repair_factory=repair_factory,
    )
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert result["repair_admitted"] is True
    assert captured["admitted"] is True
    assert captured["composite"] is True


@pytest.mark.parametrize("variant", ["empty", "prose", "exact_refs"])
def test_r17_prose_inspect_first_is_admitted_as_advisory(tmp_path, monkeypatch, variant):
    request = _v2_request(tmp_path)
    envelope = json.loads(Path(request["artifact_path"]).read_text(encoding="utf-8"))
    envelope["inspect_first"] = {
        "empty": [],
        "prose": ["Read the Task Card before making the repair."],
        "exact_refs": list(envelope["evidence_refs"]),
    }[variant]
    raw = cli._canonical_json(envelope)
    Path(request["artifact_path"]).write_text(raw, encoding="utf-8")
    request["prompt"] = request["prompt"].replace(
        next(
            line for line in request["prompt"].splitlines() if line.startswith("envelope_sha256=")
        ),
        f"envelope_sha256={cli._sha256(raw)}",
    )
    monkeypatch.setattr(
        cli,
        "_git_output",
        lambda _workspace, *args: {
            ("rev-parse", "HEAD"): "b" * 40,
            ("status", "--porcelain"): "",
            ("remote", "get-url", "origin"): "git@github.com:James3014/Nexus-new.git",
        }[args],
    )
    repair = FakeGraph(cli.REPAIR_TOOLS, _record("record_worker_result", {"summary": "done"}))
    assert cli._read_regular_artifact(Path(request["artifact_path"])) == raw.encode()
    assert cli._parse_unique_json(raw.encode())["inspect_first"] == envelope["inspect_first"]
    admission = cli._semantic_v2_admission(
        request,
        Path(request["workspace_path"]).resolve(),
        Path(request["artifact_path"]),
        request["prompt"],
        ("a.py",),
        raw.encode(),
    )
    assert admission.decision == cli.ADMIT, admission
    model_calls = 0

    def model_factory(*_args):
        nonlocal model_calls
        model_calls += 1
        return SimpleNamespace(_conversation_id=None)

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=model_factory,
        diagnosis_factory=lambda *_args: pytest.fail("r17 admission must skip diagnosis"),
        repair_factory=lambda *_args: repair,
    )
    assert result["status"] == "COMPLETED"
    assert result["repair_admitted"] is True
    assert model_calls == 1
    assert repair.calls == 1


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda _r, e: e.__setitem__("inspect_first", "read first"), id="non-list"),
        pytest.param(
            lambda _r, e: e.__setitem__("inspect_first", ["read first", 1]),
            id="non-string-item",
        ),
        pytest.param(
            lambda r, e: _mutate_card(r, e, "- `a.py`", "- `a.py"),
            id="malformed-task-card",
        ),
        pytest.param(
            lambda _r, e: e["evidence_refs"].__setitem__(1, "source_absence:a.py@not-an-anchor"),
            id="malformed-source-evidence-anchor",
        ),
        pytest.param(
            lambda _r, e: e["evidence_refs"].__setitem__(
                0, "task_card:tasks/task-card.md@not-an-anchor"
            ),
            id="malformed-task-card-evidence-anchor",
        ),
    ],
)
def test_r17_malformed_admission_inputs_reject_without_calls(tmp_path, monkeypatch, mutate):
    request = _v2_matrix_request(tmp_path, mutate)
    monkeypatch.setattr(
        cli,
        "_git_output",
        lambda _workspace, *args: {
            ("rev-parse", "HEAD"): "b" * 40,
            ("status", "--porcelain"): "",
            ("remote", "get-url", "origin"): "git@github.com:James3014/Nexus-new.git",
        }[args],
    )
    calls = {"model": 0, "diagnosis": 0, "repair": 0}
    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: calls.__setitem__("model", calls["model"] + 1),
        diagnosis_factory=lambda *_args: calls.__setitem__("diagnosis", calls["diagnosis"] + 1),
        repair_factory=lambda *_args: calls.__setitem__("repair", calls["repair"] + 1),
    )
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert calls == {"model": 0, "diagnosis": 0, "repair": 0}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda _r, e: e["evidence_refs"].__setitem__(0, "source_absence:a.py@" + "a" * 16),
        lambda _r, e: e["evidence_refs"].__setitem__(1, "source_absence:b.py@" + "b" * 16),
        lambda _r, e: e["scope_signal"].update(read_only_authorities=[]),
    ],
)
def test_r17_strict_evidence_and_authority_negatives_reject_without_calls(
    tmp_path, monkeypatch, mutate
):
    request = _v2_matrix_request(tmp_path, mutate)
    monkeypatch.setattr(
        cli,
        "_git_output",
        lambda _workspace, *args: {
            ("rev-parse", "HEAD"): "b" * 40,
            ("status", "--porcelain"): "",
            ("remote", "get-url", "origin"): "git@github.com:James3014/Nexus-new.git",
        }[args],
    )
    calls = {"model": 0, "diagnosis": 0, "repair": 0}
    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: calls.__setitem__("model", calls["model"] + 1),
        diagnosis_factory=lambda *_args: calls.__setitem__("diagnosis", calls["diagnosis"] + 1),
        repair_factory=lambda *_args: calls.__setitem__("repair", calls["repair"] + 1),
    )
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert calls == {"model": 0, "diagnosis": 0, "repair": 0}


def test_worker_admit_r16_opencli_repair_uses_new_without_conversation(tmp_path, monkeypatch):
    request = _v2_request(tmp_path)
    monkeypatch.setattr(
        cli,
        "_git_output",
        lambda _workspace, *args: {
            ("rev-parse", "HEAD"): "b" * 40,
            ("status", "--porcelain"): "",
            ("remote", "get-url", "origin"): "git@github.com:James3014/Nexus-new.git",
        }[args],
    )
    ask_commands: list[list[str]] = []
    latest_prompt = ""

    def fake_run(argv, **_kwargs):
        nonlocal latest_prompt
        args = list(argv)
        if args[1:3] == ["chatgpt", "model"]:
            return SimpleNamespace(returncode=0, stdout='[{"Status":"ok"}]', stderr="")
        if args[1:3] == ["chatgpt", "ask"]:
            ask_commands.append(args)
            latest_prompt = args[3]
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([{"conversationId": "r16-repair", "response": ""}]),
                stderr="",
            )
        if args[1:3] == ["chatgpt", "detail"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([
                    {"Role": "User", "Text": latest_prompt, "Generating": False},
                    {
                        "Role": "Assistant",
                        "Text": '{"type":"final","content":"repair ok"}',
                        "Generating": False,
                    },
                ]),
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr("nexus_open_swe_runtime.opencli_web_model.subprocess.run", fake_run)
    repair_model_calls = 0

    def model_factory(_runtime, provider, model_id, transport_config, state_root):
        nonlocal repair_model_calls
        repair_model_calls += 1
        return OpenCLIWebChatModel(
            executable=transport_config["executable"],
            intelligence_level=model_id,
            opencli_profile=transport_config["profile"],
            timeout_seconds=transport_config["timeout_seconds"],
            site_session=transport_config["site_session"],
            runtime_state_root=state_root,
        )

    class RepairGraph(FakeGraph):
        def __init__(self, model):
            super().__init__(cli.REPAIR_TOOLS)
            self.model = model

        def invoke(self, _payload, config=None):
            self.calls += 1
            self.model.invoke([HumanMessage(content="repair")])
            return _record("record_worker_result", {"summary": "repair ok"})

    def repair_factory(model, *_args):
        assert model._conversation_id is None
        return RepairGraph(model)

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=model_factory,
        diagnosis_factory=lambda *_args: pytest.fail("admitted v2 must skip diagnosis"),
        repair_factory=repair_factory,
    )
    assert result["status"] == "COMPLETED"
    assert repair_model_calls == 1
    assert len(ask_commands) == 1
    assert "--new" in ask_commands[0]
    assert "--conversation" not in ask_commands[0]


def test_worker_invalid_v2_hash_rejects_before_any_model_or_graph(tmp_path):
    request = _v2_request(tmp_path)
    request["prompt"] = request["prompt"].replace(
        "envelope_sha256=", "envelope_sha256=" + "0" * 64 + "\n#"
    )
    model_calls = 0

    def model_factory(*_args):
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("invalid v2 must not construct a model")

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=model_factory,
        diagnosis_factory=lambda *_args: pytest.fail("invalid v2 diagnosis"),
        repair_factory=lambda *_args: pytest.fail("invalid v2 repair"),
    )
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert (
        cli._semantic_v2_admission(
            request,
            Path(request["workspace_path"]),
            Path(request["artifact_path"]),
            request["prompt"],
            ("a.py",),
        ).decision
        == cli.REJECT
    )
    assert model_calls == 0


def test_malformed_v2_rejects_without_fallback_model(tmp_path):
    request = _v2_request(tmp_path)
    Path(request["artifact_path"]).write_text(
        '{"schema":"external_execution_envelope.v2",', encoding="utf-8"
    )
    model_calls = 0

    def model_factory(*_args):
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("malformed v2 must not construct a model")

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=model_factory,
        diagnosis_factory=lambda *_args: pytest.fail("malformed v2 diagnosis"),
        repair_factory=lambda *_args: pytest.fail("malformed v2 repair"),
    )
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert model_calls == 0


def test_v2_artifact_symlink_rejects(tmp_path):
    request = _v2_request(tmp_path)
    artifact = Path(request["artifact_path"])
    target = artifact.with_name("artifact-target.json")
    artifact.replace(target)
    artifact.symlink_to(target.name)
    assert (
        cli._semantic_v2_admission(
            request,
            Path(request["workspace_path"]),
            artifact,
            request["prompt"],
            ("a.py",),
        ).decision
        == cli.REJECT
    )


def test_v2_missing_full_worker_identity_rejects(tmp_path):
    request = _v2_request(tmp_path)
    request.pop("worker_identity")
    assert (
        cli._semantic_v2_admission(
            request,
            Path(request["workspace_path"]),
            Path(request["artifact_path"]),
            request["prompt"],
            ("a.py",),
        ).decision
        == cli.REJECT
    )


def test_v2_card_symlink_rejects(tmp_path):
    request = _v2_request(tmp_path)
    workspace = Path(request["workspace_path"])
    card = workspace / "tasks/task-card.md"
    target = workspace / "tasks/card-target.md"
    card.replace(target)
    card.symlink_to(target.name)
    assert (
        cli._semantic_v2_admission(
            request,
            workspace,
            Path(request["artifact_path"]),
            request["prompt"],
            ("a.py",),
        ).decision
        == cli.REJECT
    )


def test_v2_target_symlink_rejects(tmp_path):
    request = _v2_request(tmp_path)
    workspace = Path(request["workspace_path"])
    (workspace / "a.py").symlink_to("tasks/task-card.md")
    assert (
        cli._semantic_v2_admission(
            request,
            workspace,
            Path(request["artifact_path"]),
            request["prompt"],
            ("a.py",),
        ).decision
        == cli.REJECT
    )


def test_v2_intermediate_parent_symlink_rejects_without_model_or_graph(tmp_path, monkeypatch):
    request = _v2_request(tmp_path)
    workspace = Path(request["workspace_path"])
    card = workspace / "tasks/task-card.md"
    card_content = card.read_text(encoding="utf-8").replace("- `a.py`", "- `nested/new.py`")
    card.write_text(card_content, encoding="utf-8")
    envelope = json.loads(Path(request["artifact_path"]).read_text(encoding="utf-8"))
    envelope["binding"]["task_card_hash"] = cli._sha256(card.read_bytes())
    envelope["scope_signal"]["required_test_edit_paths"] = ["nested/new.py"]
    envelope["scope_signal"]["verification_only_paths"] = ["nested/new.py"]
    envelope["evidence_refs"][1] = "source_absence:nested/new.py@" + "b" * 16
    envelope["inspect_first"][1] = envelope["evidence_refs"][1]
    workspace.joinpath("nested").symlink_to("tasks")
    request["prompt"] = request["prompt"].replace(
        'authorized_mutation_paths=["a.py"]',
        'authorized_mutation_paths=["nested/new.py"]',
    )
    _write_v2_mutation(request, envelope)
    monkeypatch.setattr(
        cli,
        "_git_output",
        lambda _workspace, *args: {
            ("rev-parse", "HEAD"): "b" * 40,
            ("status", "--porcelain"): "",
            ("remote", "get-url", "origin"): "git@github.com:James3014/Nexus-new.git",
        }[args],
    )
    calls = {"model": 0, "diagnosis": 0, "repair": 0}
    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=lambda *_args: calls.__setitem__("model", calls["model"] + 1),
        diagnosis_factory=lambda *_args: calls.__setitem__("diagnosis", calls["diagnosis"] + 1),
        repair_factory=lambda *_args: calls.__setitem__("repair", calls["repair"] + 1),
    )
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert calls == {"model": 0, "diagnosis": 0, "repair": 0}


def _write_v2_mutation(request: dict, envelope: dict, *, refresh_hash: bool = True) -> None:
    artifact = Path(request["artifact_path"])
    canonical = cli._canonical_json(envelope)
    artifact.write_text(canonical, encoding="utf-8")
    if refresh_hash:
        digest = cli._sha256(canonical)
        request["prompt"] = "\n".join(
            f"envelope_sha256={digest}" if line.startswith("envelope_sha256=") else line
            for line in request["prompt"].splitlines()
        )
        assert cli._prompt_field(request["prompt"], "envelope_sha256") == digest
    assert cli._sha256(artifact.read_bytes()) == cli._prompt_field(
        request["prompt"], "envelope_sha256"
    )


def _v2_matrix_request(tmp_path: Path, mutate, *, refresh_hash: bool = True) -> dict:
    request = _v2_request(tmp_path)
    envelope = json.loads(Path(request["artifact_path"]).read_text(encoding="utf-8"))
    mutate(request, envelope)
    _write_v2_mutation(request, envelope, refresh_hash=refresh_hash)
    return request


def _mutate_card(request: dict, envelope: dict, old: str, new: str) -> None:
    card = Path(request["workspace_path"]) / envelope["binding"]["task_card_ref"]
    content = card.read_text(encoding="utf-8").replace(old, new, 1)
    card.write_text(content, encoding="utf-8")
    envelope["binding"]["task_card_hash"] = cli._sha256(card.read_bytes())


@pytest.mark.parametrize(
    "name,mutate",
    [
        ("binding_base", lambda _r, e: e["binding"].update(main_sha="a" * 40)),
        ("physical_head", lambda _r, _e: None),
        ("dirty_workspace", lambda _r, _e: None),
        ("card_ref", lambda _r, e: e["binding"].update(task_card_ref="tasks/other.md")),
        ("card_hash", lambda _r, e: e["binding"].update(task_card_hash="a" * 64)),
        ("card_backtick", lambda r, e: _mutate_card(r, e, "- `a.py`", "- `a.py")),
        ("card_status", lambda r, e: _mutate_card(r, e, "`ACTIVE`", "`PAUSED`")),
        (
            "card_auto_chain",
            lambda r, e: _mutate_card(r, e, "- AUTO_CHAIN: `false`", "- AUTO_CHAIN: `true`"),
        ),
        (
            "card_deletions",
            lambda r, e: _mutate_card(
                r, e, "- allow_deletions: `false`", "- allow_deletions: `true`"
            ),
        ),
        (
            "card_approve",
            lambda r, e: _mutate_card(
                r, e, "- worker_may_approve: `false`", "- worker_may_approve: `true`"
            ),
        ),
        (
            "card_integrate",
            lambda r, e: _mutate_card(
                r, e, "- worker_may_integrate: `false`", "- worker_may_integrate: `true`"
            ),
        ),
        (
            "card_push",
            lambda r, e: _mutate_card(
                r, e, "- worker_may_push: `false`", "- worker_may_push: `true`"
            ),
        ),
        ("scope_paths", lambda _r, e: e["scope_signal"].update(required_test_edit_paths=["b.py"])),
        ("scope_max_files", lambda _r, e: e["scope_signal"].update(max_files=2)),
        ("scope_read_only", lambda _r, e: e["scope_signal"].update(read_only_authorities=[])),
        (
            "scope_extra_path",
            lambda _r, e: e["scope_signal"]["read_only_authorities"].append("tasks/other.md"),
        ),
        (
            "scope_marker_only",
            lambda _r, e: e["scope_signal"]["read_only_authorities"].append(
                "GitHub Issue 853 binding metadata"
            ),
        ),
        (
            "scope_mismatched_issue_evidence",
            lambda _r, e: (
                e["binding"].update(item_id="853", item_type="issue"),
                e["scope_signal"]["read_only_authorities"].append(
                    "GitHub Issue 853 binding metadata"
                ),
                e["evidence_refs"].append(
                    "github_issue:github://James3014/Nexus-new/issues/854@" + "e" * 16
                ),
            ),
        ),
        (
            "scope_mismatched_issue_repository",
            lambda _r, e: (
                e["binding"].update(item_id="853", item_type="issue"),
                e["scope_signal"]["read_only_authorities"].append(
                    "GitHub Issue 853 binding metadata"
                ),
                e["evidence_refs"].append(
                    "github_issue:github://other/repo/issues/853@" + "e" * 16
                ),
            ),
        ),
        (
            "scope_non_issue_item_type",
            lambda _r, e: (
                e["binding"].update(item_id="853", item_type="task"),
                e["scope_signal"]["read_only_authorities"].append(
                    "GitHub Issue 853 binding metadata"
                ),
                e["evidence_refs"].append(
                    "github_issue:github://James3014/Nexus-new/issues/853@" + "e" * 16
                ),
            ),
        ),
        (
            "scope_zero_issue",
            lambda _r, e: (
                e["binding"].update(item_id="0", item_type="issue"),
                e["scope_signal"]["read_only_authorities"].append(
                    "GitHub Issue 0 binding metadata"
                ),
                e["evidence_refs"].append(
                    "github_issue:github://James3014/Nexus-new/issues/0@" + "e" * 16
                ),
            ),
        ),
        (
            "scope_leading_zero_issue",
            lambda _r, e: (
                e["binding"].update(item_id="0853", item_type="issue"),
                e["scope_signal"]["read_only_authorities"].append(
                    "GitHub Issue 0853 binding metadata"
                ),
                e["evidence_refs"].append(
                    "github_issue:github://James3014/Nexus-new/issues/0853@" + "e" * 16
                ),
            ),
        ),
        (
            "scope_broad_authority",
            lambda _r, e: e["scope_signal"]["read_only_authorities"].append("repository contents"),
        ),
        (
            "scope_path_like_marker",
            lambda _r, e: e["scope_signal"]["read_only_authorities"].append(
                "GitHub Issue 853 binding metadata/tasks/other.md"
            ),
        ),
        (
            "scope_marker_wrong_order",
            lambda _r, e: e["scope_signal"].update(
                read_only_authorities=[
                    "GitHub Issue 853 binding metadata",
                    e["binding"]["task_card_ref"],
                ]
            ),
        ),
        (
            "scope_third_authority",
            lambda _r, e: e["scope_signal"]["read_only_authorities"].extend([
                "GitHub Issue 853 binding metadata",
                "repository contents",
            ]),
        ),
        (
            "scope_duplicate_marker",
            lambda _r, e: (
                e["binding"].update(item_id="853", item_type="issue"),
                e["scope_signal"]["read_only_authorities"].extend([
                    "GitHub Issue 853 binding metadata",
                    "GitHub Issue 853 binding metadata",
                ]),
                e["evidence_refs"].append(
                    "github_issue:github://James3014/Nexus-new/issues/853@" + "e" * 16
                ),
            ),
        ),
        (
            "scope_production",
            lambda _r, e: e["scope_signal"].update(production_edit_paths=["a.py"]),
        ),
        (
            "scope_migration",
            lambda _r, e: e["scope_signal"].update(conditional_migration_paths=["a.py"]),
        ),
        (
            "task_evidence",
            lambda _r, e: e["evidence_refs"].__setitem__(0, "source_absence:a.py@" + "a" * 16),
        ),
        (
            "source_evidence",
            lambda _r, e: e["evidence_refs"].__setitem__(1, "source_absence:b.py@" + "b" * 16),
        ),
        ("worker_mapping", lambda _r, _e: _r["worker_identity"].update(worker_id="other")),
        ("worker_hash", lambda _r, _e: _r.update(worker_identity_sha256="0" * 64)),
        ("context_digest_length", lambda _r, e: e["binding"].update(context_pack_sha256="c" * 16)),
        (
            "selected_digest_length",
            lambda _r, e: e["selected_worker"].update(admission_evidence_hash="a" * 16),
        ),
        ("top_keyset", lambda _r, e: e.update(extra=True)),
        ("nested_keyset", lambda _r, e: e["scope_signal"].update(extra=True)),
        ("unknown_schema", lambda _r, e: e.update(schema="external_execution_envelope.unknown")),
    ],
)
def test_v2_hostile_single_fault_rejects_without_model_or_graph(
    tmp_path, monkeypatch, name, mutate
):
    request = _v2_matrix_request(tmp_path, mutate)

    def fake_git(_workspace, *args):
        if name == "physical_head" and args == ("rev-parse", "HEAD"):
            return "a" * 40
        if name == "dirty_workspace" and args == ("status", "--porcelain"):
            return " M changed.py"
        return {
            ("rev-parse", "HEAD"): "b" * 40,
            ("status", "--porcelain"): "",
            ("remote", "get-url", "origin"): "git@github.com:James3014/Nexus-new.git",
        }[args]

    monkeypatch.setattr(cli, "_git_output", fake_git)
    calls = {"model": 0, "diagnosis": 0, "repair": 0}

    def model_factory(*_args):
        calls["model"] += 1
        raise AssertionError(f"{name} must reject before model construction")

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=model_factory,
        diagnosis_factory=lambda *_args: calls.__setitem__("diagnosis", calls["diagnosis"] + 1),
        repair_factory=lambda *_args: calls.__setitem__("repair", calls["repair"] + 1),
    )
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert calls == {"model": 0, "diagnosis": 0, "repair": 0}


def test_v2_duplicate_json_key_rejects_without_model_or_graph(tmp_path, monkeypatch):
    request = _v2_request(tmp_path)
    Path(request["artifact_path"]).write_text(
        '{"schema":"external_execution_envelope.v2","schema":"external_execution_envelope.v2"}',
        encoding="utf-8",
    )
    calls = {"model": 0, "diagnosis": 0, "repair": 0}

    def model_factory(*_args):
        calls["model"] += 1
        return object()

    def diagnosis_factory(*_args):
        calls["diagnosis"] += 1
        return None

    def repair_factory(*_args):
        calls["repair"] += 1
        return None

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=model_factory,
        diagnosis_factory=diagnosis_factory,
        repair_factory=repair_factory,
    )
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert calls == {"model": 0, "diagnosis": 0, "repair": 0}


def test_v2_bare_binding_and_strict_origin_are_distinct(tmp_path, monkeypatch):
    request = _v2_request(tmp_path)
    monkeypatch.setattr(
        cli,
        "_git_output",
        lambda _workspace, *args: {
            ("rev-parse", "HEAD"): "b" * 40,
            ("status", "--porcelain"): "",
            ("remote", "get-url", "origin"): "James3014/Nexus-new",
        }[args],
    )
    assert (
        cli._semantic_v2_admission(
            request,
            Path(request["workspace_path"]),
            Path(request["artifact_path"]),
            request["prompt"],
            ("a.py",),
        ).decision
        == cli.REJECT
    )


def test_worker_well_formed_inconclusive_v2_falls_back_to_diagnosis(tmp_path, monkeypatch):
    request = _v2_request(tmp_path, status="INCONCLUSIVE")
    diagnosis = FakeGraph(
        cli.DIAGNOSIS_TOOLS,
        _record(
            "record_diagnosis",
            {"status": "INCONCLUSIVE", "summary": "unclear", "evidence_paths": []},
        ),
    )
    model_calls = 0

    def model_factory(*_args):
        nonlocal model_calls
        model_calls += 1
        return object()

    result = cli._worker_run(
        request,
        runtime_loader=_runtime,
        model_factory=model_factory,
        diagnosis_factory=lambda *_args: diagnosis,
        repair_factory=lambda *_args: pytest.fail("fallback must not repair"),
    )
    assert result["status"] == "COMPLETED"
    assert (
        cli._semantic_v2_admission(
            request,
            Path(request["workspace_path"]),
            Path(request["artifact_path"]),
            request["prompt"],
            ("a.py",),
        ).decision
        == cli.FALLBACK
    )
    assert model_calls == 1
    assert diagnosis.calls == 1


def test_composite_projector_has_strict_literal_inverse():
    from nexus_open_swe_runtime.opencli_web_model import OpenCLIWebChatModel

    raw = '{"type":"tool_call","name":"write_file_and_record_worker_result","arguments":{"file_path":"a.py","content":"hello \\"world\\"","envelope":{"summary":"done"}}}'
    projected = OpenCLIWebChatModel._project_unescaped_composite_response(raw)
    assert projected == raw
    assert OpenCLIWebChatModel._inverse_repaired_composite_response(projected) == raw
    assert (
        OpenCLIWebChatModel._project_unescaped_composite_response(
            raw.replace('"name":"write_file_and_record_worker_result"', '"name":"write_file"')
        )
        is None
    )


def _composite_worker_model(
    model: OpenCLIWebChatModel,
    sends: list[str],
    *,
    response_override: str | None = None,
) -> None:
    """Make the real web model return one valid composite tool call."""

    def one_model_send(prompt, **_kwargs):
        sends.append(prompt)
        turn_id = json.loads(prompt)["turn_id"]
        model._journal_ask(turn_id, prompt)
        model._journal_bound("conversation-1")
        response = response_override or json.dumps(
            {
                "type": "tool_call",
                "name": "write_file_and_record_worker_result",
                "arguments": {
                    "file_path": "a.py",
                    "content": "VALUE = 2\n",
                    "envelope": {"summary": "repaired a.py"},
                },
            },
            separators=(",", ":"),
        )
        model._journal_response(turn_id, response)
        return response

    model._send_and_reconcile = one_model_send


def _composite_recovery_fixture(
    tmp_path: Path, *, direct: bool = False
) -> tuple[dict, DurableOperationJournal, DurableEffectJournal]:
    request = _worker_request(tmp_path)
    request.update(
        operation="worker_reconcile",
        provider_id="opencli_chatgpt",
        model_id="advanced",
        transport_config={
            "executable": "/usr/bin/opencli",
            "profile": "r16",
            "site_session": "ephemeral",
            "timeout_seconds": 30,
        },
    )
    workspace = Path(request["workspace_path"])
    target = workspace / "a.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    runtime_identity = cli._sha256(
        cli._canonical_json({
            "module_sha256": cli._sha256(Path(cli.__file__).read_bytes()),
            "deepagents": cli._deepagents_version(),
            "checkpoint_namespace": "open-swe-repair-v1",
        })
    )
    identity = RecoveryIdentity(
        operation_id=request["operation_id"],
        execution_material_sha256="b" * 64,
        workspace=str(workspace.resolve()),
        task_id="task-1",
        unit_id="u1",
        session_id="session-1",
        allowed_paths=("a.py",),
        provider_id=request["provider_id"],
        model_id=request["model_id"],
        worker_identity_sha256=request["worker_identity_sha256"],
        transport_config_sha256=cli._sha256(cli._canonical_json(request["transport_config"])),
        runtime_identity_sha256=runtime_identity,
        composite_admitted=True,
    )
    journal = DurableOperationJournal(request["runtime_state_root"], identity)
    journal.prepare()
    journal.ask_dispatching(turn_id="turn-1", prompt="repair", ordinal=0)
    journal.conversation_bound("conversation-1")
    recovered_arguments = {
        "file_path": "a.py",
        "content": "VALUE = 2\n",
        "envelope": {"summary": "repaired a.py"},
    }
    recovered_response = json.dumps(
        {
            "type": "write_file_and_record_worker_result",
            "file_path": "a.py",
            "content": "VALUE = 2\n",
            "envelope": {
                "schema": "external_intelligence_worker_result.v1",
                "status": "IMPLEMENTATION_COMPLETED",
                "task_id": "task-1",
                "unit_id": "u1",
                "summary": "repaired a.py",
            },
        }
        if direct
        else {
            "type": "tool_call",
            "name": "write_file_and_record_worker_result",
            "arguments": recovered_arguments,
        },
        separators=(",", ":"),
    )
    journal.response_recovered("turn-1", recovered_response)
    effects = DurableEffectJournal(request["runtime_state_root"], identity)
    journal.effect_journal = effects
    if direct:
        from nexus_open_swe_runtime.opencli_web_model import _canonicalize_direct_composite_response

        recovered_response_for_effect = _canonicalize_direct_composite_response(
            recovered_response, journal
        )
    else:
        recovered_response_for_effect = recovered_response
    tool_call_id = _tool_call_id(
        "write_file_and_record_worker_result", recovered_arguments, recovered_response_for_effect
    )
    effects.bind_turn("turn-1", tool_call_id)
    effect = effects.intent(
        turn_id="turn-1",
        tool_call_id=tool_call_id,
        tool_name="write_file_and_record_worker_result",
        arguments=recovered_arguments,
        path=target,
        preimage="VALUE = 1\n",
        postimage="VALUE = 2\n",
    )
    effects.recover_write(effect)
    effects.record_worker_result(effect, {"summary": "repaired a.py"})
    cli._atomic_json(
        cli._operation_path(request),
        {
            **cli._write_started(request, "worker"),
            "status": "OPEN_SWE_OUTCOME_UNKNOWN",
            "outcome_unknown": True,
            "retry_safe": False,
            "execution_material_sha256": identity.execution_material_sha256,
            "session_id": identity.session_id,
        },
    )
    return request, journal, effects


def test_worker_run_admitted_v2_executes_real_composite_graph_once(tmp_path, monkeypatch):
    request = _v2_request(tmp_path)
    monkeypatch.setattr(
        cli,
        "_git_output",
        lambda _workspace, *args: {
            ("rev-parse", "HEAD"): "b" * 40,
            ("status", "--porcelain"): "",
            ("remote", "get-url", "origin"): "git@github.com:James3014/Nexus-new.git",
        }[args],
    )
    monkeypatch.setattr(
        cli, "create_checkpoint", lambda _state_root, _operation_id, namespace: (None, namespace)
    )
    sends: list[str] = []

    def model_factory(_runtime, provider, model_id, transport_config, state_root):
        model = OpenCLIWebChatModel(
            executable=transport_config["executable"],
            intelligence_level=model_id,
            opencli_profile=transport_config["profile"],
            site_session=transport_config["site_session"],
            timeout_seconds=transport_config["timeout_seconds"],
            runtime_state_root=state_root,
        )
        assert (provider, model_id) == ("opencli_chatgpt", "advanced")
        _composite_worker_model(model, sends)
        return model

    result = cli._worker_run(
        request,
        runtime_loader=cli._load_runtime,
        model_factory=model_factory,
        repair_factory=cli.build_repair_graph,
    )

    assert result["status"] == "COMPLETED"
    assert result["diagnosis_status"] == "ROOT_CAUSE_SUPPORTED"
    assert len(sends) == 1
    assert "write_file_and_record_worker_result" in sends[0]
    assert (
        "For the final mutation and result, call write_file_and_record_worker_result exactly once"
        in sends[0]
    )
    assert "Never use write_file_and_record_worker_result" not in sends[0]
    assert (Path(request["workspace_path"]) / "a.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    effects = list(
        (Path(request["runtime_state_root"]) / "recovery" / "effects").glob("effect_*.json")
    )
    assert len(effects) == 1
    assert (
        json.loads(effects[0].read_text(encoding="utf-8"))["worker_result"]["summary"]
        == "repaired a.py"
    )


def test_worker_run_admitted_v2_executes_direct_r25_composite_graph_once(tmp_path, monkeypatch):
    request = _v2_request(tmp_path)
    monkeypatch.setattr(
        cli,
        "_git_output",
        lambda _workspace, *args: {
            ("rev-parse", "HEAD"): "b" * 40,
            ("status", "--porcelain"): "",
            ("remote", "get-url", "origin"): "git@github.com:James3014/Nexus-new.git",
        }[args],
    )
    monkeypatch.setattr(
        cli, "create_checkpoint", lambda _state_root, _operation_id, namespace: (None, namespace)
    )
    sends: list[str] = []
    direct_response = json.dumps(
        {
            "type": "write_file_and_record_worker_result",
            "file_path": "a.py",
            "content": "VALUE = 2\n",
            "envelope": {
                "schema": "external_intelligence_worker_result.v1",
                "status": "IMPLEMENTATION_COMPLETED",
                "task_id": "task-1",
                "unit_id": "u1",
                "summary": "repaired a.py",
            },
        },
        separators=(",", ":"),
    )

    def model_factory(_runtime, _provider, _model_id, transport_config, state_root):
        model = OpenCLIWebChatModel(
            executable=transport_config["executable"],
            intelligence_level="advanced",
            opencli_profile=transport_config["profile"],
            site_session=transport_config["site_session"],
            timeout_seconds=transport_config["timeout_seconds"],
            runtime_state_root=state_root,
        )
        _composite_worker_model(model, sends, response_override=direct_response)
        return model

    result = cli._worker_run(
        request,
        runtime_loader=cli._load_runtime,
        model_factory=model_factory,
        repair_factory=cli.build_repair_graph,
    )
    assert result["status"] == "COMPLETED"
    assert len(sends) == 1
    assert (Path(request["workspace_path"]) / "a.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_worker_run_fallback_and_multipath_do_not_admit_composite(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli,
        "_git_output",
        lambda _workspace, *args: {
            ("rev-parse", "HEAD"): "b" * 40,
            ("status", "--porcelain"): "",
            ("remote", "get-url", "origin"): "git@github.com:James3014/Nexus-new.git",
        }[args],
    )
    monkeypatch.setattr(
        cli, "create_checkpoint", lambda _state_root, _operation_id, namespace: (None, namespace)
    )
    fallback_root = tmp_path / "fallback"
    fallback_root.mkdir()
    cases: list[dict] = [{"request": _worker_request(fallback_root)}]
    multipath_root = tmp_path / "multipath"
    multipath_root.mkdir()
    request = _v2_request(multipath_root)
    workspace = Path(request["workspace_path"])
    card = workspace / "tasks/task-card.md"
    card.write_text(
        card.read_text(encoding="utf-8").replace("- `a.py`", "- `a.py`\n- `b.py`"), encoding="utf-8"
    )
    envelope = json.loads(Path(request["artifact_path"]).read_text(encoding="utf-8"))
    envelope["binding"]["task_card_hash"] = cli._sha256(card.read_bytes())
    envelope["scope_signal"].update(
        required_test_edit_paths=["a.py", "b.py"],
        verification_only_paths=["a.py", "b.py"],
        max_files=2,
    )
    envelope["evidence_refs"].append("source_absence:b.py@" + "b" * 16)
    envelope["inspect_first"].append(envelope["evidence_refs"][-1])
    Path(request["artifact_path"]).write_text(cli._canonical_json(envelope), encoding="utf-8")
    request["prompt"] = request["prompt"].replace(
        'authorized_mutation_paths=["a.py"]', 'authorized_mutation_paths=["a.py","b.py"]'
    )
    request["prompt"] = "\n".join(
        f"envelope_sha256={cli._sha256(Path(request['artifact_path']).read_bytes())}"
        if line.startswith("envelope_sha256=")
        else line
        for line in request["prompt"].splitlines()
    )
    assert (
        cli._semantic_v2_admission(
            request,
            workspace.resolve(),
            Path(request["artifact_path"]),
            request["prompt"],
            ("a.py", "b.py"),
        ).decision
        == cli.ADMIT
    )
    cases.append({"request": request})
    admitted_flags: list[bool] = []

    for case in cases:
        current = case["request"]
        diagnosis = FakeGraph(
            cli.DIAGNOSIS_TOOLS,
            _record(
                "record_diagnosis",
                {
                    "status": "ROOT_CAUSE_SUPPORTED",
                    "summary": "supported",
                    "evidence_paths": ["a.py"],
                },
            ),
        )
        repair = FakeGraph(cli.REPAIR_TOOLS, _record("record_worker_result", {"summary": "done"}))

        def repair_factory(_model, *_args, composite=False, **_kwargs):
            admitted_flags.append(composite)
            assert composite is False
            return repair

        result = cli._worker_run(
            current,
            runtime_loader=_runtime,
            model_factory=lambda *_args: SimpleNamespace(_conversation_id=None),
            diagnosis_factory=lambda *_args: diagnosis,
            repair_factory=repair_factory,
        )
        assert result["status"] == "COMPLETED"

    assert admitted_flags == [False, False]


def test_worker_reconcile_terminalizes_valid_durable_composite_without_model_or_write(tmp_path):
    request, _journal, effects = _composite_recovery_fixture(tmp_path)
    other_workspace = tmp_path / "other-workspace"
    other_workspace.mkdir()
    other_target = other_workspace / "a.py"
    other_target.write_text("VALUE = 1\n", encoding="utf-8")
    other_identity = RecoveryIdentity(
        operation_id="c" * 64,
        execution_material_sha256="d" * 64,
        workspace=str(other_workspace.resolve()),
        task_id="task-2",
        unit_id="u2",
        session_id="session-2",
        allowed_paths=("a.py",),
        provider_id="opencli_chatgpt",
        model_id="advanced",
        worker_identity_sha256="e" * 64,
        transport_config_sha256=effects.identity.transport_config_sha256,
        runtime_identity_sha256=effects.identity.runtime_identity_sha256,
        composite_admitted=True,
    )
    other_effects = DurableEffectJournal(request["runtime_state_root"], other_identity)
    other_effects.bind_turn("turn-2", "call-2")
    other_effect = other_effects.intent(
        turn_id="turn-2",
        tool_call_id="call-2",
        tool_name="write_file_and_record_worker_result",
        arguments={
            "file_path": "a.py",
            "content": "VALUE = other\n",
            "envelope": {"summary": "other operation"},
        },
        path=other_target,
        preimage="VALUE = 1\n",
        postimage="VALUE = other\n",
    )
    other_effects.recover_write(other_effect)
    other_effects.record_worker_result(other_effect, {"summary": "other operation"})
    calls = {"model": 0, "graph": 0}

    def forbidden_model(*_args):
        calls["model"] += 1
        raise AssertionError("durable composite receipt must not redispatch or extract model args")

    def forbidden_graph(*_args, **_kwargs):
        calls["graph"] += 1
        raise AssertionError("durable composite receipt must terminalize locally")

    result = cli._worker_reconcile(
        request,
        runtime_loader=_runtime,
        model_builder=forbidden_model,
        graph_builder=forbidden_graph,
    )

    assert result["status"] == "COMPLETED"
    assert result["outcome_unknown"] is False
    assert json.loads(result["response_text"])["summary"] == "repaired a.py"
    assert (Path(request["workspace_path"]) / "a.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert calls == {"model": 0, "graph": 0}
    assert len(list(effects.root.glob("effect_*.json"))) == 2


def test_worker_reconcile_terminalizes_direct_r25_composite_without_model_or_write(tmp_path):
    request, _journal, effects = _composite_recovery_fixture(tmp_path, direct=True)
    calls = {"model": 0, "graph": 0}

    def forbidden_model(*_args):
        calls["model"] += 1
        raise AssertionError("direct r25 recovery must not redispatch or extract model args")

    def forbidden_graph(*_args, **_kwargs):
        calls["graph"] += 1
        raise AssertionError("direct r25 recovery must terminalize locally")

    result = cli._worker_reconcile(
        request,
        runtime_loader=_runtime,
        model_builder=forbidden_model,
        graph_builder=forbidden_graph,
    )
    assert result["status"] == "COMPLETED"
    assert result["outcome_unknown"] is False
    assert calls == {"model": 0, "graph": 0}
    assert (Path(request["workspace_path"]) / "a.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert len(list(effects.root.glob("effect_*.json"))) == 1


def test_worker_reconcile_flat_direct_response_with_checkpoint_has_one_local_effect(
    tmp_path,
):
    """A flat r25 response is local projection, not an external protocol repair."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    request = _v2_request(tmp_path)
    request["operation"] = "worker_reconcile"
    workspace = Path(request["workspace_path"])
    state_root = Path(request["runtime_state_root"])
    identity = RecoveryIdentity(
        operation_id=request["operation_id"],
        execution_material_sha256="b" * 64,
        workspace=str(workspace.resolve()),
        task_id="task-1",
        unit_id="u1",
        session_id="session-1",
        allowed_paths=("a.py",),
        provider_id=request["provider_id"],
        model_id=request["model_id"],
        worker_identity_sha256=request["worker_identity_sha256"],
        transport_config_sha256=cli._sha256(cli._canonical_json(request["transport_config"])),
        runtime_identity_sha256=cli._sha256(
            cli._canonical_json({
                "module_sha256": cli._sha256(Path(cli.__file__).read_bytes()),
                "deepagents": cli._deepagents_version(),
                "checkpoint_namespace": "open-swe-repair-v1",
            })
        ),
        composite_admitted=True,
    )
    journal = DurableOperationJournal(state_root, identity)
    journal.prepare()
    journal.ask_dispatching(turn_id="turn-1", prompt="repair", ordinal=0)
    journal.conversation_bound("conversation-1")
    raw_flat = json.dumps(
        {
            "type": "write_file_and_record_worker_result",
            "file_path": "/a.py",
            "content": "VALUE = 2\n",
            "envelope": {
                "schema": "external_intelligence_worker_result.v1",
                "status": "IMPLEMENTATION_COMPLETED",
                "task_id": "task-1",
                "unit_id": "u1",
                "summary": "repaired a.py",
            },
        },
        separators=(",", ":"),
    )
    journal.response_recovered("turn-1", raw_flat)
    cli._atomic_json(
        cli._operation_path(request),
        {
            **cli._write_started(request, "worker"),
            "status": "OPEN_SWE_OUTCOME_UNKNOWN",
            "outcome_unknown": True,
            "retry_safe": False,
            "execution_material_sha256": identity.execution_material_sha256,
            "session_id": identity.session_id,
        },
    )
    checkpoint, _ = cli.create_checkpoint(
        state_root, request["operation_id"], identity.checkpoint_namespace
    )
    checkpoint.conn.close()

    class LocalRecoveryModel(BaseChatModel):
        model_name: str = "advanced"
        detail_calls: int = 0
        repair_calls: int = 0
        generate_calls: int = 0

        @property
        def _llm_type(self):
            return "test-local-recovery-model"

        def bind_tools(self, tools, **kwargs):
            return self

        def configure_recovery_journal(self, value):
            return None

        def _detail_response(self, conversation_id, *, wait, turn_id):
            self.detail_calls += 1
            raise AssertionError("flat RESPONSE_RECOVERED must not call detail")

        def _repair_protocol_response(self, response):
            self.repair_calls += 1
            raise AssertionError("flat r25 response must not call protocol repair")

        _repair_matches_invalid_response = staticmethod(
            OpenCLIWebChatModel._repair_matches_invalid_response
        )

        def _response_message(self, _response, _tools):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file_and_record_worker_result",
                        "args": {
                            "file_path": "a.py",
                            "content": "VALUE = 2\n",
                            "envelope": {"summary": "repaired a.py"},
                        },
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )

        def _generate(self, messages, **kwargs):
            self.generate_calls += 1
            if not any(
                isinstance(message, ToolMessage)
                and getattr(message, "name", None) == "write_file_and_record_worker_result"
                for message in messages
            ):
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "write_file_and_record_worker_result",
                                        "args": {
                                            "file_path": "a.py",
                                            "content": "VALUE = 2\n",
                                            "envelope": {"summary": "repaired a.py"},
                                        },
                                        "id": "call-1",
                                        "type": "tool_call",
                                    }
                                ],
                            )
                        )
                    ]
                )
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])

    model = LocalRecoveryModel()
    graph_calls = 0

    def graph_builder(
        model,
        root,
        runtime,
        allowed_paths,
        key,
        checkpointer=None,
        effect_journal=None,
        composite=False,
    ):
        nonlocal graph_calls
        graph_calls += 1
        graph = cli.build_repair_graph(
            model,
            root,
            runtime,
            allowed_paths,
            key,
            checkpointer=checkpointer,
            effect_journal=effect_journal,
            composite=composite,
        )
        original_invoke = graph.invoke

        def invoke(*invoke_args, **invoke_kwargs):
            output = original_invoke(*invoke_args, **invoke_kwargs)
            return output

        graph.invoke = invoke
        return graph

    result = cli._worker_reconcile(
        request,
        runtime_loader=cli._load_runtime,
        model_builder=lambda *_args: model,
        graph_builder=graph_builder,
    )
    assert result["status"] == "COMPLETED"
    assert model.detail_calls == 0
    assert model.repair_calls == 0
    assert model.generate_calls == 1
    assert graph_calls == 1
    assert (workspace / "a.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    effects = list((state_root / "recovery" / "effects").glob("effect_*.json"))
    assert len(effects) == 1
    assert json.loads(effects[0].read_text(encoding="utf-8"))["status"] == "RESULT"
    state = journal.read()
    assert state["protocol_repair_origin"] == raw_flat
    assert state["protocol_repair_origin_sha256"] == cli._sha256(raw_flat)
    assert state["protocol_repair_status"] == "RECOVERED"
    assert state["recovered_response"] == json.dumps(
        {
            "type": "tool_call",
            "name": "write_file_and_record_worker_result",
            "arguments": {
                "file_path": "a.py",
                "content": "VALUE = 2\n",
                "envelope": {"summary": "repaired a.py"},
            },
        },
        separators=(",", ":"),
    )
    assert state["response_sha256"] == cli._sha256(state["recovered_response"])

    second = cli._worker_reconcile(
        request,
        runtime_loader=lambda: pytest.fail("second reconcile must not load runtime"),
        model_builder=lambda *_args: pytest.fail("second reconcile must not build model"),
        graph_builder=lambda *_args, **_kwargs: pytest.fail(
            "second reconcile must not build graph"
        ),
    )
    assert second == result
    assert graph_calls == 1
    assert len(list((state_root / "recovery" / "effects").glob("effect_*.json"))) == 1


@pytest.mark.parametrize(
    "field,value", [("task_id", "other"), ("unit_id", "other"), ("file_path", "../a.py")]
)
def test_worker_reconcile_rejects_hostile_direct_r25_identity_or_path(tmp_path, field, value):
    request, journal, _effects = _composite_recovery_fixture(tmp_path, direct=True)
    state = journal.read()
    raw = json.loads(state["recovered_response"])
    if field in {"task_id", "unit_id"}:
        raw["envelope"][field] = value
    else:
        raw[field] = value
    journal.response_recovered("turn-1", json.dumps(raw, separators=(",", ":")))
    result = cli._worker_reconcile(
        request,
        runtime_loader=lambda: pytest.fail("hostile direct response must not build model"),
        model_builder=lambda *_args: pytest.fail("hostile direct response must not build model"),
        graph_builder=lambda *_args, **_kwargs: pytest.fail(
            "hostile direct response must not build graph"
        ),
    )
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"


def test_composite_terminal_predicate_rejects_error_tool_message_without_effect_file(
    tmp_path,
):
    from langchain_core.messages import AIMessage

    from nexus_open_swe_runtime.opencli_web_model import _composite_terminal_completed

    _request, journal, effects = _composite_recovery_fixture(tmp_path)
    effect_path = next(effects.root.glob("effect_*.json"))
    receipt = json.loads(effect_path.read_text(encoding="utf-8"))["worker_result"]
    effect_path.unlink()
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file_and_record_worker_result",
                    "args": {
                        "file_path": "a.py",
                        "content": "VALUE = 2\n",
                        "envelope": {"summary": "repaired a.py"},
                    },
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=json.dumps(receipt),
            tool_call_id="call-1",
            name="write_file_and_record_worker_result",
            status="error",
        ),
    ]
    tools = [{"type": "function", "function": {"name": "write_file_and_record_worker_result"}}]

    assert not _composite_terminal_completed(messages, tools, journal)


@pytest.mark.parametrize("tamper", ["receipt", "hash"])
def test_worker_reconcile_tampered_durable_composite_receipt_fails_closed(tmp_path, tamper):
    request, _journal, effects = _composite_recovery_fixture(tmp_path)
    effect_path = next(effects.root.glob("effect_*.json"))
    record = json.loads(effect_path.read_text(encoding="utf-8"))
    if tamper == "receipt":
        record["worker_result"]["summary"] = "tampered"
    else:
        record["worker_result_sha256"] = "0" * 64
    effect_path.write_text(json.dumps(record), encoding="utf-8")

    result = cli._worker_reconcile(
        request,
        runtime_loader=_runtime,
        model_builder=lambda *_args: pytest.fail("tampered receipt must not build model"),
        graph_builder=lambda *_args: pytest.fail("tampered receipt must not build graph"),
    )

    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert result["outcome_unknown"] is True


@pytest.mark.parametrize("tamper", ["summary", "path", "turn", "tool", "effect_id"])
def test_worker_reconcile_rehashed_composite_identity_mutations_remain_unknown(tmp_path, tamper):
    """A receipt remains bound to its durable effect and operation identity."""
    request, _journal, effects = _composite_recovery_fixture(tmp_path)
    effect_path = next(effects.root.glob("effect_*.json"))
    record = json.loads(effect_path.read_text(encoding="utf-8"))
    receipt = dict(record["worker_result"])
    arguments = dict(record["arguments"])
    envelope = dict(arguments["envelope"])

    if tamper == "summary":
        envelope["summary"] = "rehashed but unbound"
        arguments["envelope"] = envelope
        receipt["summary"] = envelope["summary"]
        receipt["summary_sha256"] = cli._sha256(receipt["summary"])
    elif tamper == "path":
        arguments["file_path"] = "other.py"
        record["path"] = str(Path(request["workspace_path"]) / "other.py")
        receipt["path"] = record["path"]
    elif tamper == "turn":
        record["turn_id"] = "turn-tampered"
        receipt["turn_id"] = record["turn_id"]
    elif tamper == "tool":
        record["tool_name"] = "write_file"
        receipt["tool_name"] = record["tool_name"]
    else:
        receipt["effect_id"] = "effect_" + "f" * 64

    record["arguments"] = arguments
    effect_material = {
        "operation_id": record["operation_id"],
        "turn_id": record["turn_id"],
        "tool_call_id": record["tool_call_id"],
        "tool_name": record["tool_name"],
        "arguments": arguments,
    }
    expected_effect_id = "effect_" + cli._sha256(cli._canonical_json(effect_material))
    if tamper != "effect_id":
        receipt["effect_id"] = expected_effect_id
    record["effect_id"] = receipt["effect_id"]
    receipt_material = dict(receipt)
    receipt_material.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = cli._sha256(cli._canonical_json(receipt_material))
    record["worker_result"] = receipt
    record["worker_result_sha256"] = cli._sha256(cli._canonical_json(receipt))

    replacement = effects.root / f"{record['effect_id']}.json"
    effect_path.rename(replacement)
    replacement.write_text(cli._canonical_json(record), encoding="utf-8")

    result = cli._worker_reconcile(
        request,
        runtime_loader=_runtime,
        model_builder=lambda *_args: pytest.fail("rehashed identity mutation must not build model"),
        graph_builder=lambda *_args: pytest.fail("rehashed identity mutation must not build graph"),
    )

    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert result["outcome_unknown"] is True


def test_worker_run_malformed_r23_composite_completes_once_then_reconcile_is_local(
    tmp_path, monkeypatch
):
    """Malformed r23 free text is repaired once; restart does no second write or web call."""
    request = _v2_request(tmp_path)
    monkeypatch.setattr(
        cli,
        "_git_output",
        lambda _workspace, *args: {
            ("rev-parse", "HEAD"): "b" * 40,
            ("status", "--porcelain"): "",
            ("remote", "get-url", "origin"): "git@github.com:James3014/Nexus-new.git",
        }[args],
    )
    monkeypatch.setattr(
        cli, "create_checkpoint", lambda _state_root, _operation_id, namespace: (None, namespace)
    )
    sends: list[str] = []
    malformed_r23 = (
        '{"type":"tool_call","name":"write_file_and_record_worker_result",'
        '"arguments":{"file_path":"a.py","content":"VALUE = "2"\\n",'
        '"envelope":{"summary":"repaired "a.py""}}}'
    )

    def model_factory(_runtime, provider, model_id, transport_config, state_root):
        model = OpenCLIWebChatModel(
            executable=transport_config["executable"],
            intelligence_level=model_id,
            opencli_profile=transport_config["profile"],
            site_session=transport_config["site_session"],
            timeout_seconds=transport_config["timeout_seconds"],
            runtime_state_root=state_root,
        )
        assert (provider, model_id) == ("opencli_chatgpt", "advanced")
        _composite_worker_model(model, sends, response_override=malformed_r23)
        return model

    result = cli._worker_run(
        request,
        runtime_loader=cli._load_runtime,
        model_factory=model_factory,
        repair_factory=cli.build_repair_graph,
    )

    assert result["status"] == "COMPLETED"
    assert len(sends) == 1
    assert (Path(request["workspace_path"]) / "a.py").read_text(encoding="utf-8") == 'VALUE = "2"\n'

    replace_calls: list[tuple[object, object]] = []
    original_replace = cli.os.replace

    def forbidden_reconcile_replace(source, target):
        replace_calls.append((source, target))
        return original_replace(source, target)

    monkeypatch.setattr(cli.os, "replace", forbidden_reconcile_replace)
    model_calls = 0

    def forbidden_model(*_args):
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("completed malformed composite must not call model on reconcile")

    reconciled = cli._worker_reconcile(
        request,
        runtime_loader=_runtime,
        model_builder=forbidden_model,
        graph_builder=lambda *_args: pytest.fail(
            "completed malformed composite must not build graph"
        ),
    )
    assert reconciled["status"] == "COMPLETED"
    assert model_calls == 0
    assert replace_calls == []
