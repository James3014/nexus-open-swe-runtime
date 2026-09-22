from __future__ import annotations

import os
import pathlib
import shutil
import tempfile

import pytest

from nexus_open_swe_runtime.model_invocation import (
    MODEL_INVOCATION_RECEIPT_FIELDS,
    MODEL_INVOCATION_RECEIPT_SCHEMA,
    DurableInvocationJournal,
    ModelInvocationError,
    build_model_invocation_receipt,
    invocation_binding_from_identity,
)


def _identity():
    return {
        "operation_id": "op-test",
        "provider_id": "opencli-chatgpt-web",
        "model_id": "chatgpt-4o",
        "worker_identity_sha256": "w",
        "transport_config_sha256": "t",
        "runtime_identity_sha256": "r",
        "core_binding_hash": "c",
        "acceptance_contract_hash": "a",
        "effect_authorization_hash": "e",
        "tool_projection_hash": "p",
        "workspace": "/ws",
        "checkpoint_namespace": "cn",
    }


def _attempt(**overrides):
    attempt = {
        "operation_id": "op-test",
        "provider_id": "opencli-chatgpt-web",
        "model_id": "chatgpt-4o",
        "backend_id": "opencli-chatgpt-web",
        "attempt_ordinal": 1,
        "turn_id": "t1",
        "new_conversation": True,
        "request_sha256": "req",
        "response_sha256": "resp",
        "conversation_id_sha256": "conv",
        "started_at": "2026-01-01T00:00:00Z",
    }
    attempt.update(overrides)
    return attempt


def _receipt(identity=None, attempt=None):
    identity = _identity() if identity is None else identity
    binding = invocation_binding_from_identity(identity)
    return build_model_invocation_receipt(
        binding, _attempt() if attempt is None else attempt
    )


def _fresh_journal():
    """Return (state_root, journal) with the true on-disk layout."""
    state_root = tempfile.mkdtemp()
    journal = DurableInvocationJournal(state_root, _identity())
    return pathlib.Path(state_root), journal


class TestInvocationBinding:
    def test_binding_has_all_twelve_keys(self):
        binding = invocation_binding_from_identity(_identity())
        assert len(binding) == 12
        assert set(binding) == {
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
        }

    def test_binding_changes_when_identity_effect_authorization_changes(self):
        binding_a = invocation_binding_from_identity(_identity())
        binding_b = invocation_binding_from_identity(
            {**_identity(), "effect_authorization_hash": "e2"}
        )
        assert binding_a != binding_b
        assert (
            binding_a["effect_authorization_hash"]
            != binding_b["effect_authorization_hash"]
        )


class TestReceiptBuilder:
    def test_receipt_is_self_hashed_and_stable_across_builds(self):
        r1 = _receipt()
        r2 = _receipt()
        assert r1 == r2
        assert r1["schema"] == MODEL_INVOCATION_RECEIPT_SCHEMA
        assert r1["receipt_sha256"]
        assert set(r1) == set(MODEL_INVOCATION_RECEIPT_FIELDS)

    def test_receipt_hashes_its_own_fields(self):
        r = _receipt()
        assert r["receipt_sha256"]

    def test_transport_classification_is_fail_closed_on_unknown(self):
        from nexus_open_swe_runtime.model_invocation import (
            classify_transport_outcome,
        )

        assert classify_transport_outcome("OPENCLI_WEB_UNKNOWN_THING") == "OUTCOME_UNKNOWN"


class TestJournalRecordFailClosed:
    def test_rejects_field_tamper_on_record(self):
        state_root, journal = _fresh_journal()
        receipt = _receipt()
        tampered = dict(receipt)
        tampered["provider_status"] = "ALSO_NEW_STATUS"
        with pytest.raises(ModelInvocationError) as exc:
            journal.record(tampered)
        assert "MODEL_INVOCATION_RECEIPT" in str(exc.value)

    def test_rejects_receipt_hash_tamper_on_record(self):
        state_root, journal = _fresh_journal()
        receipt = _receipt()
        tampered = dict(receipt)
        tampered["receipt_sha256"] = "0" * 32
        with pytest.raises(ModelInvocationError) as exc:
            journal.record(tampered)
        assert "MODEL_INVOCATION_RECEIPT" in str(exc.value)

    def test_rejects_identity_mismatch_on_record(self):
        state_root, journal = _fresh_journal()
        foreign_identity = {**_identity(), "effect_authorization_hash": "FOREIGN"}
        foreign_receipt = _receipt(identity=foreign_identity)
        with pytest.raises(ModelInvocationError) as exc:
            journal.record(foreign_receipt)
        assert "MODEL_INVOCATION_" in str(exc.value)

    def test_accepts_well_formed_receipt(self):
        state_root, journal = _fresh_journal()
        receipt = _receipt()
        journal.record(receipt)
        assert journal.result_invocations() == [receipt]


class TestJournalReadFailClosed:
    def test_module_uses_symlink_tamper_refusing_read(self):
        """Reading the on-disk journal must refuse symlinked state."""
        state_root, journal = _fresh_journal()
        journal.record(_receipt())
        # point the journal root's parent through a symlink
        inv_root = state_root / "recovery" / "invocations"
        real_dir = tempfile.mkdtemp()
        backup = state_root / "recovery_invocations_real"
        shutil.move(str(inv_root), str(backup))
        os.symlink(real_dir, str(inv_root))
        with pytest.raises(ModelInvocationError, match="MODEL_INVOCATION_STATE_SYMLINK"):
            journal.result_invocations()
        os.unlink(str(inv_root))
        shutil.move(str(backup), str(inv_root))

    def test_module_refuses_symlinked_receipt_file(self):
        state_root, journal = _fresh_journal()
        journal.record(_receipt())
        inv_root = state_root / "recovery" / "invocations"
        target = next(inv_root.iterdir())
        target.unlink()
        target.symlink_to(tempfile.mkdtemp())
        with pytest.raises(ModelInvocationError, match="MODEL_INVOCATION_STATE_SYMLINK"):
            journal.result_invocations()

    def test_module_refuses_corrupt_record(self):
        state_root, journal = _fresh_journal()
        journal.record(_receipt())
        inv_root = state_root / "recovery" / "invocations"
        target = next(inv_root.iterdir())
        target.write_text("{not-json", encoding="utf-8")
        with pytest.raises(ModelInvocationError) as exc:
            journal.result_invocations()
        assert "MODEL_INVOCATION_RECORD_CORRUPT" in str(exc.value)


    def test_cli_receipt_projection_preserves_fail_closed_corrupt_journal(self):
        from nexus_open_swe_runtime.cli import _invocation_receipts

        state_root, journal = _fresh_journal()
        journal.record(_receipt())
        inv_root = state_root / "recovery" / "invocations"
        target = next(inv_root.iterdir())
        target.write_text("{not-json", encoding="utf-8")
        with pytest.raises(ModelInvocationError, match="MODEL_INVOCATION_RECORD_CORRUPT"):
            _invocation_receipts(journal)


class TestJournalRoundTrip:
    def test_round_trip_is_idempotent_and_filters_by_binding_operation(self):
        state_root, journal = _fresh_journal()
        a = _receipt()
        journal.record(a)
        journal.record(a)
        assert journal.result_invocations() == [a]
        assert (
            journal.result_invocations(operation_id=_receipt()["operation_id"])
            == [a]
        )
        assert journal.result_invocations(operation_id="op-other") == []
        assert journal.result_invocations(operation_id="op-none") == []

    def test_receipt_operation_derives_from_binding_not_attempt(self):
        a = _receipt()
        b = _receipt(attempt=_attempt(operation_id="op-other"))
        assert a == b
        assert a["operation_id"] == "op-test"

    def test_result_invocations_empty_before_any_record(self):
        state_root, journal = _fresh_journal()
        assert journal.result_invocations() == []
