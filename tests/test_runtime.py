from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from nexus_open_swe_runtime import cli


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
    artifact.write_text('{"failure":"VALUE must be 2"}\n', encoding="utf-8")
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
    request["operation_id"] = "d" * 64
    request["prompt"] = ""
    request["artifact_path"] = ""
    reconciled = cli.dispatch(request)
    assert reconciled == result
    assert diagnosis.calls == 1
    assert repair.calls == 1


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
    assert "sentinel-provider-secret" not in text
    assert "sentinel-github-secret" not in text
    assert state_file.stat().st_mode & 0o077 == 0
