from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from nexus_open_swe_runtime import cli
from nexus_open_swe_runtime.opencli_web_model import OpenCLIWebChatModel


class FakeGraph:
    def __init__(self, surface, output=None, effect=None, error=None):
        self.surface = tuple(surface)
        self.output = output
        self.effect = effect
        self.error = error
        self.calls = 0

    def get_graph(self):
        tools = {name: object() for name in self.surface}
        return SimpleNamespace(nodes={"tools": SimpleNamespace(data=SimpleNamespace(tools_by_name=tools))})

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
        "prompt": "\n".join(
            [
                "task_id=task-1",
                "unit_id=u1",
                'authorized_mutation_paths=["a.py"]',
                "bounded repair",
            ]
        ),
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
    assert events == ["model", "diagnosis_graph", "diagnosis_invoke", "model", "repair_graph", "repair_invoke"]


def test_worker_opencli_repair_phase_starts_without_diagnosis_conversation(tmp_path):
    request = _worker_request(tmp_path)
    request.update(
        {
            "provider_id": "opencli_chatgpt",
            "model_id": "very-high",
            "transport_config": {
                "executable": "/opt/opencli",
                "profile": "balanced",
                "site_session": "ephemeral",
                "timeout_seconds": 120,
            },
        }
    )
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


def test_worker_reconcile_does_not_substitute_completed_operation_from_stale_workspace_index(tmp_path):
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
    cli._atomic_json(
        state_root / "operations" / f"{operation_a}.json", completed_a
    )
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

    result = cli.dispatch(
        {
            "schema": cli.REQUEST_SCHEMA,
            "operation": "worker_reconcile",
            "operation_id": operation_b,
            "runtime_state_root": str(state_root),
            "workspace_path": str(workspace),
        }
    )

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

    result = cli.dispatch(
        {
            "schema": cli.REQUEST_SCHEMA,
            "operation": "worker_reconcile",
            "operation_id": operation_id,
            "runtime_state_root": str(state_root),
            "workspace_path": str(other_workspace),
            "provider_id": "provider-a",
            "model_id": "model-a",
            "worker_identity_sha256": "d" * 64,
        }
    )

    assert result["operation_id"] == operation_id
    assert result["status"] == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert result["outcome_unknown"] is True


def test_worker_replay_with_changed_material_does_not_return_cached_terminal_or_redispatch(tmp_path):
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
    ["prompt", "artifact", "session", "operation", "provider_id", "model_id", "worker_identity_sha256"],
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
    state_file = Path(request["runtime_state_root"]) / "operations" / f"{request['operation_id']}.json"
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
        _record("record_finding", {
            "schema": "external_execution_envelope.v1",
            "findings": [{"file": "sample.py", "line": 1}],
        }),
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

    request["prompt"] = "\n".join(
        [
            "task_id=task-1",
            "unit_id=u1",
            f'authorized_mutation_paths=["{target_rel}"]',
            "bounded repair create target",
        ]
    )

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
    initial_files = {p.relative_to(workspace): p.read_bytes() for p in workspace.rglob("*") if p.is_file()}
    unauthorized_rel = "tests/ops/unauthorized_absent.py"
    target = workspace / unauthorized_rel
    assert not target.exists()

    request["prompt"] = "\n".join(
        [
            "task_id=task-1",
            "unit_id=u1",
            'authorized_mutation_paths=["tests/ops/new_canary.py"]',
            "bounded repair unauthorized",
        ]
    )

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
    current_files = {p.relative_to(workspace): p.read_bytes() for p in workspace.rglob("*") if p.is_file()}
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

    request["prompt"] = "\n".join(
        [
            "task_id=task-1",
            "unit_id=u1",
            f'authorized_mutation_paths=["{target_rel}"]',
            "bounded repair dangling symlink",
        ]
    )

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
    request.update(
        {
            "provider_id": "opencli_chatgpt",
            "model_id": "advanced",
            "transport_config": {
                "executable": "/usr/bin/opencli",
                "profile": "r16",
                "site_session": "ephemeral",
                "timeout_seconds": 30,
            },
        }
    )
    workspace = Path(request["workspace_path"])
    (workspace / "a.py").unlink()
    card_ref = "tasks/task-card.md"
    card = workspace / card_ref
    card.parent.mkdir(parents=True)
    card.write_text(
        "\n".join(
            [
                "- task_id: `task-1`",
                "- status: `ACTIVE`",
                "- worker_may_approve: `false`",
                "- worker_may_integrate: `false`",
                "- worker_may_push: `false`",
                "- AUTO_CHAIN: `false`",
                "- allow_deletions: `false`",
                "\n## Allowed files\n",
                "- `a.py`",
            ]
        ),
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
    request["prompt"] += "\n" + "\n".join(
        [
            f"envelope_sha256={cli._sha256(raw)}",
            "expected_base_sha=" + "b" * 40,
        ]
    )
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
        next(line for line in request["prompt"].splitlines() if line.startswith("envelope_sha256=")),
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
def test_r17_strict_evidence_and_authority_negatives_reject_without_calls(tmp_path, monkeypatch, mutate):
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
                stdout=json.dumps(
                    [
                        {"Role": "User", "Text": latest_prompt, "Generating": False},
                        {
                            "Role": "Assistant",
                            "Text": '{"type":"final","content":"repair ok"}',
                            "Generating": False,
                        },
                    ]
                ),
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
    request["prompt"] = request["prompt"].replace("envelope_sha256=", "envelope_sha256=" + "0" * 64 + "\n#")
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
    assert cli._semantic_v2_admission(
        request,
        Path(request["workspace_path"]),
        Path(request["artifact_path"]),
        request["prompt"],
        ("a.py",),
    ).decision == cli.REJECT
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
    assert cli._semantic_v2_admission(
        request,
        Path(request["workspace_path"]),
        artifact,
        request["prompt"],
        ("a.py",),
    ).decision == cli.REJECT


def test_v2_missing_full_worker_identity_rejects(tmp_path):
    request = _v2_request(tmp_path)
    request.pop("worker_identity")
    assert cli._semantic_v2_admission(
        request,
        Path(request["workspace_path"]),
        Path(request["artifact_path"]),
        request["prompt"],
        ("a.py",),
    ).decision == cli.REJECT


def test_v2_card_symlink_rejects(tmp_path):
    request = _v2_request(tmp_path)
    workspace = Path(request["workspace_path"])
    card = workspace / "tasks/task-card.md"
    target = workspace / "tasks/card-target.md"
    card.replace(target)
    card.symlink_to(target.name)
    assert cli._semantic_v2_admission(
        request,
        workspace,
        Path(request["artifact_path"]),
        request["prompt"],
        ("a.py",),
    ).decision == cli.REJECT


def test_v2_target_symlink_rejects(tmp_path):
    request = _v2_request(tmp_path)
    workspace = Path(request["workspace_path"])
    (workspace / "a.py").symlink_to("tasks/task-card.md")
    assert cli._semantic_v2_admission(
        request,
        workspace,
        Path(request["artifact_path"]),
        request["prompt"],
        ("a.py",),
    ).decision == cli.REJECT


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
        ("card_auto_chain", lambda r, e: _mutate_card(r, e, "- AUTO_CHAIN: `false`", "- AUTO_CHAIN: `true`")),
        ("card_deletions", lambda r, e: _mutate_card(r, e, "- allow_deletions: `false`", "- allow_deletions: `true`")),
        ("card_approve", lambda r, e: _mutate_card(r, e, "- worker_may_approve: `false`", "- worker_may_approve: `true`")),
        ("card_integrate", lambda r, e: _mutate_card(r, e, "- worker_may_integrate: `false`", "- worker_may_integrate: `true`")),
        ("card_push", lambda r, e: _mutate_card(r, e, "- worker_may_push: `false`", "- worker_may_push: `true`")),
        ("scope_paths", lambda _r, e: e["scope_signal"].update(required_test_edit_paths=["b.py"])),
        ("scope_max_files", lambda _r, e: e["scope_signal"].update(max_files=2)),
        ("scope_read_only", lambda _r, e: e["scope_signal"].update(read_only_authorities=[])),
        ("scope_production", lambda _r, e: e["scope_signal"].update(production_edit_paths=["a.py"])),
        ("scope_migration", lambda _r, e: e["scope_signal"].update(conditional_migration_paths=["a.py"])),
        ("task_evidence", lambda _r, e: e["evidence_refs"].__setitem__(0, "source_absence:a.py@" + "a" * 16)),
        ("source_evidence", lambda _r, e: e["evidence_refs"].__setitem__(1, "source_absence:b.py@" + "b" * 16)),
        ("worker_mapping", lambda _r, _e: _r["worker_identity"].update(worker_id="other")),
        ("worker_hash", lambda _r, _e: _r.update(worker_identity_sha256="0" * 64)),
        ("context_digest_length", lambda _r, e: e["binding"].update(context_pack_sha256="c" * 16)),
        ("selected_digest_length", lambda _r, e: e["selected_worker"].update(admission_evidence_hash="a" * 16)),
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
    assert cli._semantic_v2_admission(
        request,
        Path(request["workspace_path"]),
        Path(request["artifact_path"]),
        request["prompt"],
        ("a.py",),
    ).decision == cli.REJECT


def test_worker_well_formed_inconclusive_v2_falls_back_to_diagnosis(tmp_path, monkeypatch):
    request = _v2_request(tmp_path, status="INCONCLUSIVE")
    diagnosis = FakeGraph(
        cli.DIAGNOSIS_TOOLS,
        _record("record_diagnosis", {"status": "INCONCLUSIVE", "summary": "unclear", "evidence_paths": []}),
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
    assert cli._semantic_v2_admission(
        request,
        Path(request["workspace_path"]),
        Path(request["artifact_path"]),
        request["prompt"],
        ("a.py",),
    ).decision == cli.FALLBACK
    assert model_calls == 1
    assert diagnosis.calls == 1
