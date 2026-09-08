from __future__ import annotations

import json
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
