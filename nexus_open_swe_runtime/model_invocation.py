from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

"""Model-invocation execution evidence.

A stable, hash-bound receipt per physical provider/model attempt.  Deliberately
carries no routing, selection, correctness, or acceptance authority: it is
``DERIVED_EXECUTION_EVIDENCE_ONLY`` evidence for one external model transport
attempt.  Missing telemetry is recorded as ``NOT_MEASURED`` and is never coerced
to a numeric value.
"""

MODEL_INVOCATION_RECEIPT_SCHEMA = "nexus.open_swe_runtime.model_invocation_receipt.v1"
MODEL_INVOCATION_AUTHORITY_KIND = "DERIVED_EXECUTION_EVIDENCE_ONLY"
MODEL_INVOCATION_BACKEND_ID = "opencli-chatgpt-web"
NOT_MEASURED = "NOT_MEASURED"

TRANSPORT_OUTCOME_OBSERVED_OK = "OBSERVED_OK"
TRANSPORT_OUTCOME_PROVIDER_STARTUP_FAILURE = "PROVIDER_STARTUP_FAILURE"
TRANSPORT_OUTCOME_PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
TRANSPORT_OUTCOME_TRANSPORT_FAILURE = "TRANSPORT_FAILURE"
TRANSPORT_OUTCOME_INVALID_RESPONSE = "INVALID_RESPONSE"
TRANSPORT_OUTCOME_OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"

MODEL_INVOCATION_BINDING_KEYS = (
    "operation_id",
    "provider_id",
    "model_id",
    "worker_identity_sha256",
    "transport_config_sha256",
    "runtime_identity_sha256",
    "core_binding_hash",
    "acceptance_contract_hash",
    "effect_authorization_hash",
    "tool_projection_hash",
    "workspace",
    "checkpoint_namespace",
)

MODEL_INVOCATION_RECEIPT_FIELDS = (
    "schema",
    "authority_kind",
    "operation_id",
    "provider_id",
    "model_id",
    "provider_model_revision",
    "backend_id",
    "backend_identity",
    "worker_identity_sha256",
    "core_binding_hash",
    "acceptance_contract_hash",
    "effect_authorization_hash",
    "tool_projection_hash",
    "transport_config_sha256",
    "runtime_identity_sha256",
    "workspace",
    "checkpoint_namespace",
    "turn_id",
    "attempt_ordinal",
    "new_conversation",
    "prior_conversation_id_sha256",
    "conversation_id_sha256",
    "started_at",
    "started_monotonic",
    "finished_monotonic",
    "setup_latency_seconds",
    "active_latency_seconds",
    "readback_latency_seconds",
    "total_latency_seconds",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "request_bytes",
    "response_bytes",
    "request_sha256",
    "response_sha256",
    "content_hash",
    "transport_outcome",
    "provider_status",
    "error_class",
    "timeout_class",
    "response_valid",
    "external_effect_outcome_known",
    "blind_replay_prohibited",
    "recovered_after_timeout",
    "resume_required",
    "invocation_id",
    "receipt_sha256",
)

_TIME_CONVERSION_CODES = frozenset({
    "OPENCLI_WEB_TIMEOUT",
    "OPENCLI_WEB_TIMEOUT_RECONCILE_UNKNOWN",
    "OPENCLI_WEB_RESUME_REQUIRED",
    "OPENCLI_WEB_REPAIR_RESUME_EXHAUSTED",
})


class ModelInvocationError(ValueError):
    """Bounded model-invocation evidence failure."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: Any) -> str:
    if isinstance(value, str):
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _fsync_replace(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink() or path.parent.is_symlink():
        raise ModelInvocationError("MODEL_INVOCATION_STATE_SYMLINK")
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


def invocation_binding_from_identity(identity: Any) -> dict[str, str]:
    """Normalize a recovery identity or mapping into the 12-key binding."""
    binding: dict[str, str] = {}
    for key in MODEL_INVOCATION_BINDING_KEYS:
        if isinstance(identity, Mapping):
            value = identity.get(key)
        else:
            value = getattr(identity, key, None)
        if value is None:
            value = ""
        if isinstance(value, (list, tuple)):
            value = "|".join(str(item) for item in value)
        else:
            value = str(value)
        binding[key] = value
    return binding


def classify_transport_outcome(error_code: str) -> str:
    """Map a bounded transport error code to the invocation outcome taxonomy."""
    invalid_response = frozenset({
        "OPENCLI_WEB_RESPONSE_INVALID",
        "OPENCLI_WEB_CONVERSATION_ID_INVALID",
        "OPENCLI_WEB_CONVERSATION_ID_MISMATCH",
        "OPENCLI_WEB_RECONCILE_INVALID",
        "OPENCLI_WEB_TURN_IDENTITY_UNKNOWN",
        "OPENCLI_WEB_RECONCILE_INCOMPLETE",
    })
    if error_code in {"OPENCLI_NOT_FOUND", "OPENCLI_WEB_PROCESS_FAILURE", "OPENCLI_WEB_BUSY"}:
        return TRANSPORT_OUTCOME_TRANSPORT_FAILURE
    if error_code == "OPENCLI_WEB_HARD_BLOCK":
        return TRANSPORT_OUTCOME_PROVIDER_STARTUP_FAILURE
    if error_code == "OPENCLI_WEB_TIMEOUT":
        return TRANSPORT_OUTCOME_PROVIDER_TIMEOUT
    if error_code in {
        "OPENCLI_WEB_TIMEOUT_RECONCILE_UNKNOWN",
        "OPENCLI_WEB_RESUME_REQUIRED",
        "OPENCLI_WEB_REPAIR_RESUME_EXHAUSTED",
    }:
        return TRANSPORT_OUTCOME_OUTCOME_UNKNOWN
    if error_code in invalid_response:
        return TRANSPORT_OUTCOME_INVALID_RESPONSE
    return TRANSPORT_OUTCOME_OUTCOME_UNKNOWN


def provider_status_for_error(error_code: str) -> str:
    """The provider was reachable and answered when its response was rejected."""
    if classify_transport_outcome(error_code) == TRANSPORT_OUTCOME_INVALID_RESPONSE:
        return "ok"
    return "failed"


def external_outcome_known_for_error(error_code: str) -> bool:
    """Known-effect codes: the transport reports a definitive provider state."""
    known_effect_codes = frozenset({
        "OPENCLI_NOT_FOUND",
        "OPENCLI_WEB_BUSY",
        "OPENCLI_WEB_HARD_BLOCK",
    })
    return (
        classify_transport_outcome(error_code) == TRANSPORT_OUTCOME_INVALID_RESPONSE
        or error_code in known_effect_codes
    )


def _required_text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelInvocationError(code)
    return value


def build_model_invocation_receipt(
    binding: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one self-hashed receipt for a physical transport attempt."""
    operation_id = _required_text(
        binding.get("operation_id"), "MODEL_INVOCATION_RECEIPT_FIELDS_INVALID"
    )
    provider_id = _required_text(
        binding.get("provider_id"), "MODEL_INVOCATION_RECEIPT_FIELDS_INVALID"
    )
    model_id = _required_text(
        binding.get("model_id"), "MODEL_INVOCATION_RECEIPT_FIELDS_INVALID"
    )
    backend_id = _required_text(
        attempt.get("backend_id"), "MODEL_INVOCATION_RECEIPT_FIELDS_INVALID"
    )
    conversation_id_sha256 = attempt.get("conversation_id_sha256", NOT_MEASURED)
    response_sha256 = attempt.get("response_sha256", NOT_MEASURED)
    subject = {
        "request_sha256": attempt.get("request_sha256", ""),
        "response_sha256": response_sha256,
        "conversation_id_sha256": conversation_id_sha256,
        "attempt_ordinal": attempt.get("attempt_ordinal", 0),
    }
    content_hash = _sha256(_canonical_json(subject))
    invocation_identity = _sha256(_canonical_json({
        "operation_id": operation_id,
        "provider_id": provider_id,
        "model_id": model_id,
        "backend_id": backend_id,
        "turn_id": attempt.get("turn_id", ""),
        "attempt_ordinal": attempt.get("attempt_ordinal", 0),
        "request_sha256": attempt.get("request_sha256", ""),
        "new_conversation": attempt.get("new_conversation", False),
    }))
    invocation_id = f"inv_{invocation_identity}"
    backend_identity = dict(attempt.get("backend_identity") or {})
    backend_identity["configured_identity"] = {
        "provider_id": provider_id,
        "model_id": model_id,
    }
    backend_identity["observed_execution_identity"] = {
        "provider_id": attempt.get("observed_provider_id", NOT_MEASURED),
        "model_id": attempt.get("observed_model_id", NOT_MEASURED),
        "model_revision": attempt.get(
            "observed_model_revision",
            attempt.get("provider_model_revision", NOT_MEASURED),
        ),
        "observation_source": attempt.get("identity_observation_source", NOT_MEASURED),
    }
    receipt: dict[str, Any] = {
        "schema": MODEL_INVOCATION_RECEIPT_SCHEMA,
        "authority_kind": MODEL_INVOCATION_AUTHORITY_KIND,
        "operation_id": operation_id,
        "provider_id": provider_id,
        "model_id": model_id,
        "provider_model_revision": attempt.get("provider_model_revision", NOT_MEASURED),
        "backend_id": backend_id,
        "backend_identity": backend_identity,
        "worker_identity_sha256": str(binding.get("worker_identity_sha256") or ""),
        "core_binding_hash": str(binding.get("core_binding_hash") or ""),
        "acceptance_contract_hash": str(binding.get("acceptance_contract_hash") or ""),
        "effect_authorization_hash": str(binding.get("effect_authorization_hash") or ""),
        "tool_projection_hash": str(binding.get("tool_projection_hash") or ""),
        "transport_config_sha256": str(binding.get("transport_config_sha256") or ""),
        "runtime_identity_sha256": str(binding.get("runtime_identity_sha256") or ""),
        "workspace": str(binding.get("workspace") or ""),
        "checkpoint_namespace": str(binding.get("checkpoint_namespace") or ""),
        "turn_id": attempt.get("turn_id", ""),
        "attempt_ordinal": attempt.get("attempt_ordinal", 1),
        "new_conversation": bool(attempt.get("new_conversation")),
        "prior_conversation_id_sha256": attempt.get(
            "prior_conversation_id_sha256", NOT_MEASURED
        ),
        "conversation_id_sha256": conversation_id_sha256,
        "started_at": attempt.get("started_at", ""),
        "started_monotonic": attempt.get("started_monotonic", NOT_MEASURED),
        "finished_monotonic": attempt.get("finished_monotonic", NOT_MEASURED),
        "setup_latency_seconds": attempt.get("setup_latency_seconds", NOT_MEASURED),
        "active_latency_seconds": attempt.get("active_latency_seconds", NOT_MEASURED),
        "readback_latency_seconds": attempt.get("readback_latency_seconds", NOT_MEASURED),
        "total_latency_seconds": attempt.get("total_latency_seconds", NOT_MEASURED),
        "input_tokens": attempt.get("input_tokens", NOT_MEASURED),
        "output_tokens": attempt.get("output_tokens", NOT_MEASURED),
        "cached_tokens": attempt.get("cached_tokens", NOT_MEASURED),
        "request_bytes": attempt.get("request_bytes", NOT_MEASURED),
        "response_bytes": attempt.get("response_bytes", NOT_MEASURED),
        "request_sha256": attempt.get("request_sha256", ""),
        "response_sha256": response_sha256,
        "content_hash": content_hash,
        "transport_outcome": attempt.get(
            "transport_outcome", TRANSPORT_OUTCOME_OUTCOME_UNKNOWN
        ),
        "provider_status": attempt.get("provider_status", NOT_MEASURED),
        "error_class": attempt.get("error_class", NOT_MEASURED),
        "timeout_class": attempt.get("timeout_class", NOT_MEASURED),
        "response_valid": attempt.get("response_valid", NOT_MEASURED),
        "external_effect_outcome_known": bool(
            attempt.get("external_effect_outcome_known")
        ),
        "blind_replay_prohibited": bool(attempt.get("blind_replay_prohibited")),
        "recovered_after_timeout": bool(attempt.get("recovered_after_timeout")),
        "resume_required": bool(attempt.get("resume_required")),
        "invocation_id": invocation_id,
    }
    receipt["receipt_sha256"] = _sha256(
        _canonical_json({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    )
    return receipt


def _validated_receipt(
    receipt: Any,
    binding: Mapping[str, str] | None,
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise ModelInvocationError("MODEL_INVOCATION_RECEIPT_FIELDS_INVALID")
    if set(receipt) != set(MODEL_INVOCATION_RECEIPT_FIELDS):
        raise ModelInvocationError("MODEL_INVOCATION_RECEIPT_FIELDS_INVALID")
    if receipt.get("schema") != MODEL_INVOCATION_RECEIPT_SCHEMA:
        raise ModelInvocationError("MODEL_INVOCATION_RECEIPT_FIELDS_INVALID")
    for key in ("operation_id", "provider_id", "model_id", "backend_id", "invocation_id"):
        if not isinstance(receipt.get(key), str) or not receipt[key]:
            raise ModelInvocationError("MODEL_INVOCATION_RECEIPT_FIELDS_INVALID")
    ordered = {key: receipt[key] for key in MODEL_INVOCATION_RECEIPT_FIELDS}
    hashed = {key: value for key, value in ordered.items() if key != "receipt_sha256"}
    computed = _sha256(_canonical_json(hashed))
    if ordered.get("receipt_sha256") != computed:
        raise ModelInvocationError("MODEL_INVOCATION_RECEIPT_HASH_INVALID")
    if binding is not None:
        for key in MODEL_INVOCATION_BINDING_KEYS:
            if binding.get(key, "") != ordered.get(key, ""):
                raise ModelInvocationError("MODEL_INVOCATION_IDENTITY_MISMATCH")
    return ordered




def observed_execution_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return physical observation only; never fall back to configured identity."""
    validated = _validated_receipt(receipt, None)
    backend_identity = validated.get("backend_identity")
    if not isinstance(backend_identity, Mapping):
        backend_identity = {}
    observed = backend_identity.get("observed_execution_identity")
    if isinstance(observed, Mapping):
        return {
            "provider_id": observed.get("provider_id", NOT_MEASURED),
            "model_id": observed.get("model_id", NOT_MEASURED),
            "model_revision": observed.get("model_revision", NOT_MEASURED),
            "observation_source": observed.get("observation_source", NOT_MEASURED),
        }
    return {
        "provider_id": NOT_MEASURED,
        "model_id": NOT_MEASURED,
        "model_revision": NOT_MEASURED,
        "observation_source": "LEGACY_RECEIPT_NO_OBSERVED_IDENTITY",
    }


def _message_material(message: BaseMessage) -> Any:
    if hasattr(message, "model_dump"):
        try:
            return message.model_dump(mode="json")
        except (TypeError, ValueError):
            pass
    return {
        "type": getattr(message, "type", message.__class__.__name__),
        "content": getattr(message, "content", ""),
    }


def _observed_identity_from_message(message: BaseMessage) -> dict[str, Any]:
    """Extract only response-owned identity metadata; configured values are forbidden."""
    metadata = getattr(message, "response_metadata", None)
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}

    def first_text(*keys: str) -> Any:
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return NOT_MEASURED

    provider = first_text("model_provider", "provider", "provider_name")
    model = first_text("model_name", "model", "model_id")
    revision = first_text("model_revision", "model_version", "system_fingerprint")
    observed_keys = [
        key
        for key in (
            "model_provider",
            "provider",
            "provider_name",
            "model_name",
            "model",
            "model_id",
            "model_revision",
            "model_version",
            "system_fingerprint",
        )
        if key in metadata
    ]
    source = (
        "langchain.response_metadata:" + ",".join(observed_keys)
        if observed_keys
        else NOT_MEASURED
    )
    return {
        "provider_id": provider,
        "model_id": model,
        "model_revision": revision,
        "observation_source": source,
    }


def _usage_from_message(message: BaseMessage) -> dict[str, Any]:
    usage = getattr(message, "usage_metadata", None)
    usage = dict(usage) if isinstance(usage, Mapping) else {}

    def integer(name: str) -> Any:
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return NOT_MEASURED

    return {
        "input_tokens": integer("input_tokens"),
        "output_tokens": integer("output_tokens"),
        "cached_tokens": NOT_MEASURED,
    }


@dataclass
class _InvocationEvidenceState:
    binding: dict[str, str] = field(default_factory=dict)
    journal: Any = None
    attempt_ordinal: int = 0
    receipts: list[dict[str, Any]] = field(default_factory=list)


class InvocationEvidenceChatModel(BaseChatModel):
    """Generic LangChain transport wrapper with fail-closed invocation evidence."""

    configured_provider_id: str
    configured_model_id: str
    _delegate: Any = PrivateAttr(default=None)
    _state: _InvocationEvidenceState = PrivateAttr(default_factory=_InvocationEvidenceState)

    def __init__(
        self,
        *,
        delegate: Any,
        configured_provider_id: str,
        configured_model_id: str,
        state: _InvocationEvidenceState | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            configured_provider_id=configured_provider_id,
            configured_model_id=configured_model_id,
            **kwargs,
        )
        self._delegate = delegate
        if state is not None:
            self._state = state

    @property
    def _llm_type(self) -> str:
        return "nexus-invocation-evidence-wrapper"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "configured_provider_id": self.configured_provider_id,
            "configured_model_id": self.configured_model_id,
            "delegate_type": (
                f"{self._delegate.__class__.__module__}."
                f"{self._delegate.__class__.__qualname__}"
            ),
        }

    def configure_invocation_identity(self, identity: Mapping[str, Any]) -> None:
        self._state.binding = invocation_binding_from_identity(identity)

    def configure_invocation_journal(self, journal: Any) -> None:
        binding = getattr(journal, "binding", None)
        if isinstance(binding, Mapping):
            self._state.binding = invocation_binding_from_identity(binding)
        self._state.journal = journal

    def invocations(self) -> list[dict[str, Any]]:
        return list(self._state.receipts)

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        if not hasattr(self._delegate, "bind_tools"):
            raise ModelInvocationError("MODEL_INVOCATION_BIND_TOOLS_UNSUPPORTED")
        bound = self._delegate.bind_tools(tools, tool_choice=tool_choice, **kwargs)
        return InvocationEvidenceChatModel(
            delegate=bound,
            configured_provider_id=self.configured_provider_id,
            configured_model_id=self.configured_model_id,
            state=self._state,
        )

    def _record_attempt(self, attempt: Mapping[str, Any]) -> dict[str, Any]:
        if not self._state.binding:
            raise ModelInvocationError("MODEL_INVOCATION_BINDING_MISSING")
        receipt = build_model_invocation_receipt(self._state.binding, attempt)
        journal = self._state.journal
        if journal is not None:
            journal.record(receipt)
        self._state.receipts.append(receipt)
        return receipt

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        self._state.attempt_ordinal += 1
        ordinal = self._state.attempt_ordinal
        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        request_material = [_message_material(message) for message in messages]
        request_json = _canonical_json(request_material)
        attempt: dict[str, Any] = {
            "backend_id": "langchain-chat-model",
            "backend_identity": {
                "transport_class": "LANGCHAIN_PROVIDER",
                "delegate_module": self._delegate.__class__.__module__,
                "delegate_class": self._delegate.__class__.__qualname__,
            },
            "turn_id": f"langchain:{ordinal}",
            "attempt_ordinal": ordinal,
            "new_conversation": False,
            "started_at": started_at,
            "started_monotonic": round(started, 6),
            "finished_monotonic": NOT_MEASURED,
            "setup_latency_seconds": NOT_MEASURED,
            "active_latency_seconds": NOT_MEASURED,
            "readback_latency_seconds": NOT_MEASURED,
            "total_latency_seconds": NOT_MEASURED,
            "request_bytes": len(request_json.encode("utf-8")),
            "request_sha256": _sha256(request_json),
            "observed_provider_id": NOT_MEASURED,
            "observed_model_id": NOT_MEASURED,
            "observed_model_revision": NOT_MEASURED,
            "identity_observation_source": NOT_MEASURED,
        }
        try:
            message = self._delegate.invoke(messages, stop=stop, **kwargs)
        except Exception as exc:
            finished = time.monotonic()
            attempt.update(
                finished_monotonic=round(finished, 6),
                active_latency_seconds=round(finished - started, 6),
                total_latency_seconds=round(finished - started, 6),
                transport_outcome=TRANSPORT_OUTCOME_OUTCOME_UNKNOWN,
                provider_status="failed",
                error_class=type(exc).__name__,
                timeout_class=NOT_MEASURED,
                response_valid=NOT_MEASURED,
                external_effect_outcome_known=False,
                blind_replay_prohibited=True,
            )
            self._record_attempt(attempt)
            raise

        if not isinstance(message, BaseMessage):
            attempt.update(
                transport_outcome=TRANSPORT_OUTCOME_INVALID_RESPONSE,
                provider_status="ok",
                error_class="MODEL_INVOCATION_RESPONSE_NOT_MESSAGE",
                timeout_class=NOT_MEASURED,
                response_valid=False,
                external_effect_outcome_known=True,
                blind_replay_prohibited=True,
            )
            self._record_attempt(attempt)
            raise ModelInvocationError("MODEL_INVOCATION_RESPONSE_INVALID")

        finished = time.monotonic()
        observed = _observed_identity_from_message(message)
        usage = _usage_from_message(message)
        response_json = _canonical_json(_message_material(message))
        attempt.update(
            finished_monotonic=round(finished, 6),
            active_latency_seconds=round(finished - started, 6),
            total_latency_seconds=round(finished - started, 6),
            response_bytes=len(response_json.encode("utf-8")),
            response_sha256=_sha256(response_json),
            provider_model_revision=observed["model_revision"],
            observed_provider_id=observed["provider_id"],
            observed_model_id=observed["model_id"],
            observed_model_revision=observed["model_revision"],
            identity_observation_source=observed["observation_source"],
            transport_outcome=TRANSPORT_OUTCOME_OBSERVED_OK,
            provider_status="ok",
            error_class=NOT_MEASURED,
            timeout_class=NOT_MEASURED,
            response_valid=True,
            external_effect_outcome_known=True,
            blind_replay_prohibited=False,
            **usage,
        )
        self._record_attempt(attempt)
        return ChatResult(generations=[ChatGeneration(message=message)])


class DurableInvocationJournal:
    """Fail-closed per-invocation evidence store rooted under ``recovery/``."""

    def __init__(self, state_root: str | Path, identity: Any) -> None:
        self.identity = identity
        self.binding = invocation_binding_from_identity(identity)
        self.journal_root = Path(state_root).expanduser() / "recovery" / "invocations"

    def _ensure_root(self) -> None:
        if self.journal_root.is_symlink() or self.journal_root.parent.is_symlink():
            raise ModelInvocationError("MODEL_INVOCATION_STATE_SYMLINK")
        self.journal_root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def record(self, receipt: Mapping[str, Any]) -> None:
        receipt = _validated_receipt(receipt, self.binding)
        self._ensure_root()
        _fsync_replace(
            self.journal_root / f"{receipt['invocation_id']}.json",
            receipt,
        )

    def result_invocations(self, *, operation_id: str | None = None) -> list[dict[str, Any]]:
        if not self.journal_root.is_dir():
            return []
        if self.journal_root.is_symlink() or self.journal_root.parent.is_symlink():
            raise ModelInvocationError("MODEL_INVOCATION_STATE_SYMLINK")
        receipts: list[dict[str, Any]] = []
        for path in sorted(self.journal_root.glob("inv_*.json")):
            if path.is_symlink():
                raise ModelInvocationError("MODEL_INVOCATION_STATE_SYMLINK")
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ModelInvocationError("MODEL_INVOCATION_RECORD_CORRUPT") from exc
            try:
                receipt = _validated_receipt(raw, self.binding)
            except ModelInvocationError as exc:
                if str(exc) == "MODEL_INVOCATION_IDENTITY_MISMATCH":
                    raise
                raise ModelInvocationError("MODEL_INVOCATION_RECORD_CORRUPT") from exc
            if operation_id is not None and receipt["operation_id"] != operation_id:
                continue
            receipts.append(receipt)
        return receipts


__all__ = [
    "MODEL_INVOCATION_AUTHORITY_KIND",
    "MODEL_INVOCATION_BACKEND_ID",
    "MODEL_INVOCATION_RECEIPT_SCHEMA",
    "NOT_MEASURED",
    "TRANSPORT_OUTCOME_OBSERVED_OK",
    "TRANSPORT_OUTCOME_PROVIDER_STARTUP_FAILURE",
    "TRANSPORT_OUTCOME_PROVIDER_TIMEOUT",
    "TRANSPORT_OUTCOME_TRANSPORT_FAILURE",
    "TRANSPORT_OUTCOME_INVALID_RESPONSE",
    "TRANSPORT_OUTCOME_OUTCOME_UNKNOWN",
    "DurableInvocationJournal",
    "InvocationEvidenceChatModel",
    "ModelInvocationError",
    "build_model_invocation_receipt",
    "classify_transport_outcome",
    "external_outcome_known_for_error",
    "invocation_binding_from_identity",
    "observed_execution_identity",
    "provider_status_for_error",
]