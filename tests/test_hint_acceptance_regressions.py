"""Malformed hints preserve execution recovery instead of raising."""
import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("hint_acceptance", Path(__file__).parents[1] / "nexus_open_swe_runtime/hint_execution.py")
assert _SPEC is not None and _SPEC.loader is not None
H = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(H)


def test_malformed_hint_fallback_keeps_authorized_tools():
    for value in (42, "bad", ["bad"], {"repository": 42}):
        result = H.bind_execution_hints(expected_repository="o/r", expected_revision="h", hint_input=value, authorized_tools=["search", "read"])
        assert not result["bound"]
        assert result["authorized_tools"] == ["read", "search"]


def test_discovered_files_do_not_inflate_hinted_total():
    result = H.build_hint_execution_receipt(bound=True, hinted_reads=["a.py"], discovered_outside=["b.py"])
    assert result["hinted_total"] == 1
    assert result["discovered_outside_hints"] == ["b.py"]


def test_invalid_tool_call_counts_rejected():
    for value in (-1, 1.5, True, None):
        with unittest.TestCase().assertRaises(ValueError):
            H.build_hint_execution_receipt(bound=True, tool_calls=[{"tool": "read", "count": value}])
