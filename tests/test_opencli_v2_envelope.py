from __future__ import annotations

import json

import pytest

from nexus_open_swe_runtime import opencli_web_model as wm
from nexus_open_swe_runtime.opencli_web_model import OpenCLIWebModelError

ASK = [{"conversationId": "c1", "response": " hi "}]
HIST = [{"Id": "a"}, {"Id": "b"}]


def wrap(d):
    return json.dumps({"ok": True, "data": d})


def test_unwrap_accepts_bare_and_envelope():
    assert wm._unwrap_opencli_json(json.dumps(ASK)) == ASK
    assert wm._unwrap_opencli_json(wrap(ASK)) == ASK


@pytest.mark.parametrize("raw", [
    json.dumps({"ok": False, "error": {"code": "X"}}),
    json.dumps({"ok": True, "data": {"a": 1}}),
    json.dumps({"ok": True, "data": ASK, "extra": 1}),
    json.dumps({"ok": 1, "data": ASK}),
    json.dumps({"data": ASK}),
])
def test_unwrap_other_shapes_not_list(raw):
    assert not isinstance(wm._unwrap_opencli_json(raw), list)


@pytest.mark.parametrize("fmt", [json.dumps, wrap])
def test_extractors_accept_both_shapes(fmt):
    model = wm.OpenCLIWebChatModel
    assert model._extract_ask_result(fmt(ASK)) == ("c1", "hi")
    assert model._extract_history_ids(fmt(HIST)) == ["a", "b"]
    rows = [{"Role": "User", "Text": "x"}, {"Role": "Assistant", "Text": "y", "Generating": False}]
    assert model._extract_detail_response(fmt(rows)) == "y"


def test_extractors_reject_ok_false():
    bad = json.dumps({"ok": False, "error": {"code": "E"}})
    model = wm.OpenCLIWebChatModel
    with pytest.raises(OpenCLIWebModelError):
        model._extract_ask_result(bad)
    with pytest.raises(OpenCLIWebModelError):
        model._extract_history_ids(bad)
    with pytest.raises(OpenCLIWebModelError):
        model._extract_detail_response(bad)
