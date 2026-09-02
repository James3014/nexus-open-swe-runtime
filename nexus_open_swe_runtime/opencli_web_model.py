from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any, Callable, Mapping, Sequence

from langchain_core.language_models.chat_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

OPENCLI_WEB_PROTOCOL = "nexus.opencli_web_chat.v1"
_ALLOWED_LEVELS = frozenset({"fast", "balanced", "advanced", "very-high", "pro"})


class OpenCLIWebModelError(RuntimeError):
    """Bounded failure from the ChatGPT Web transport."""


def _message_role(message: BaseMessage) -> str:
    if message.type == "human":
        return "user"
    if message.type == "ai":
        return "assistant"
    return str(message.type)


def _jsonable_content(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_jsonable_content(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable_content(item) for key, item in value.items()}
    return str(value)


def _message_payload(message: BaseMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": _message_role(message),
        "content": _jsonable_content(message.content),
    }
    if isinstance(message, AIMessage) and message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": str(call.get("id") or ""),
                "name": str(call.get("name") or ""),
                "arguments": dict(call.get("args") or {}),
            }
            for call in message.tool_calls
        ]
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = str(message.tool_call_id)
    return payload


def _tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    if not isinstance(function, Mapping):
        return ""
    name = function.get("name")
    return str(name) if isinstance(name, str) else ""


def _tool_call_id(name: str, arguments: Mapping[str, Any], raw: str) -> str:
    canonical = json.dumps(
        {"name": name, "arguments": dict(arguments), "raw": raw},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "opencli_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


class OpenCLIWebChatModel(BaseChatModel):
    """LangChain chat-model bridge backed by ChatGPT Web through OpenCLI.

    The bridge deliberately owns no repository tools and no Nexus authority. It
    only translates LangChain messages/tool declarations into one ChatGPT Web
    model turn and converts an explicit web response back into an ``AIMessage``.
    Tool execution remains inside Deep Agents / the external Open SWE runtime.

    Retained-conversation recovery and stricter response-schema enforcement are
    intentionally separate follow-up gates; this W3 bridge uses a fresh ChatGPT
    conversation for each model turn.
    """

    executable: str = "opencli"
    intelligence_level: str = "very-high"
    profile: str = ""
    timeout_seconds: int = 120
    site_session: str = "ephemeral"
    disable_streaming: bool = True
    _conversation_id: str | None = PrivateAttr(default=None)

    @property
    def model_name(self) -> str:
        """Provider-native identifier used by the Deep Agents harness registry."""
        return self.intelligence_level

    @property
    def _llm_type(self) -> str:
        return "opencli-chatgpt-web"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        params = dict(super()._get_ls_params(stop=stop, **kwargs))
        params["ls_provider"] = "opencli_chatgpt"
        params["ls_model_name"] = self.intelligence_level
        return params

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "executable": self.executable,
            "intelligence_level": self.intelligence_level,
            "site_session": self.site_session,
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=formatted, **kwargs)

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.profile:
            env["OPENCLI_PROFILE"] = self.profile
        return env

    def _run(self, argv: Sequence[str]) -> str:
        try:
            result = subprocess.run(
                list(argv),
                check=False,
                capture_output=True,
                text=True,
                timeout=max(self.timeout_seconds + 5, 35),
                shell=False,
                env=self._environment(),
            )
        except FileNotFoundError as exc:
            raise OpenCLIWebModelError("OPENCLI_NOT_FOUND") from exc
        except subprocess.TimeoutExpired as exc:
            raise OpenCLIWebModelError("OPENCLI_WEB_TIMEOUT") from exc
        if result.returncode != 0:
            diagnostic = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
            if "timed out" in diagnostic and "may still complete" in diagnostic:
                raise OpenCLIWebModelError("OPENCLI_WEB_TIMEOUT")
            raise OpenCLIWebModelError("OPENCLI_WEB_PROCESS_FAILURE")
        return result.stdout or ""

    def _select_intelligence_level(self) -> None:
        if self.intelligence_level not in _ALLOWED_LEVELS:
            raise OpenCLIWebModelError("OPENCLI_WEB_MODEL_LEVEL_INVALID")
        argv = [
            self.executable,
            "chatgpt",
            "model",
            self.intelligence_level,
            "--site-session",
            self.site_session,
            "-f",
            "json",
        ]
        try:
            self._run(argv)
        except OpenCLIWebModelError as exc:
            if str(exc) != "OPENCLI_WEB_PROCESS_FAILURE":
                raise
            self._run(argv)

    def _render_prompt(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Mapping[str, Any]],
        tool_choice: str | None,
    ) -> str:
        message_payloads = [_message_payload(message) for message in messages]
        tool_payloads = [dict(tool) for tool in tools]
        turn_material = json.dumps(
            {
                "messages": message_payloads,
                "tools": tool_payloads,
                "tool_choice": tool_choice or "auto",
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        envelope = {
            "protocol": OPENCLI_WEB_PROTOCOL,
            "turn_id": "turn_" + hashlib.sha256(turn_material.encode("utf-8")).hexdigest()[:24],
            "role": "model_transport_only",
            "rules": [
                "Repository tools are executed by the external Open SWE runtime, not by ChatGPT Web.",
                "If a tool is needed, return one JSON object with type=tool_call, name, and arguments.",
                "If no tool is needed, return one JSON object with type=final and content.",
                "Do not claim that a tool ran unless a later tool message reports its result.",
            ],
            "messages": message_payloads,
            "tools": tool_payloads,
            "tool_choice": tool_choice or "auto",
        }
        return json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _extract_ask_result(stdout: str) -> tuple[str, str]:
        try:
            rows = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise OpenCLIWebModelError("OPENCLI_WEB_RESPONSE_INVALID") from exc
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
            raise OpenCLIWebModelError("OPENCLI_WEB_RESPONSE_INVALID")
        conversation_id = rows[0].get("conversationId")
        response = rows[0].get("response")
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise OpenCLIWebModelError("OPENCLI_WEB_CONVERSATION_ID_INVALID")
        if not isinstance(response, str):
            raise OpenCLIWebModelError("OPENCLI_WEB_RESPONSE_INVALID")
        return conversation_id.strip(), response.strip()

    @staticmethod
    def _extract_detail_response(stdout: str, expected_turn_id: str = "") -> str:
        try:
            rows = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise OpenCLIWebModelError("OPENCLI_WEB_RECONCILE_INVALID") from exc
        if not isinstance(rows, list) or not rows:
            raise OpenCLIWebModelError("OPENCLI_WEB_RECONCILE_INVALID")
        start = 0
        if expected_turn_id:
            matching_user_indexes = [
                index
                for index, row in enumerate(rows)
                if isinstance(row, Mapping)
                and row.get("Role") == "User"
                and expected_turn_id in str(row.get("Text") or "")
            ]
            if len(matching_user_indexes) != 1:
                raise OpenCLIWebModelError("OPENCLI_WEB_TURN_IDENTITY_UNKNOWN")
            start = matching_user_indexes[0] + 1
        for row in rows[start:]:
            if not isinstance(row, Mapping) or row.get("Role") != "Assistant":
                continue
            text = row.get("Text")
            generating = row.get("Generating")
            if isinstance(text, str) and text.strip() and generating is False:
                return text.strip()
        raise OpenCLIWebModelError("OPENCLI_WEB_RECONCILE_INCOMPLETE")

    @staticmethod
    def _extract_history_ids(stdout: str) -> list[str]:
        try:
            rows = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise OpenCLIWebModelError("OPENCLI_WEB_HISTORY_INVALID") from exc
        if not isinstance(rows, list):
            raise OpenCLIWebModelError("OPENCLI_WEB_HISTORY_INVALID")
        result: list[str] = []
        for row in rows[:12]:
            if isinstance(row, Mapping) and isinstance(row.get("Id"), str):
                result.append(str(row["Id"]))
        return result

    def _detail_response(self, conversation_id: str, *, wait: bool, turn_id: str = "") -> str:
        readback_timeout = max(self.timeout_seconds, 30) if wait else self.timeout_seconds
        detail = self._run([
            self.executable,
            "chatgpt",
            "detail",
            conversation_id,
            "--wait",
            "true" if wait else "false",
            "--timeout",
            str(readback_timeout),
            "--stable",
            "6",
            "--site-session",
            self.site_session,
            "-f",
            "json",
        ])
        return self._extract_detail_response(detail, turn_id)

    def _reconcile_timeout(self, turn_id: str) -> str:
        if self._conversation_id:
            return self._detail_response(self._conversation_id, wait=True, turn_id=turn_id)
        history = self._run([
            self.executable,
            "chatgpt",
            "history",
            "--site-session",
            self.site_session,
            "-f",
            "json",
        ])
        matches: list[str] = []
        for conversation_id in self._extract_history_ids(history):
            detail = self._run([
                self.executable,
                "chatgpt",
                "detail",
                conversation_id,
                "--wait",
                "false",
                "--site-session",
                self.site_session,
                "-f",
                "json",
            ])
            if turn_id in detail:
                matches.append(conversation_id)
        if len(matches) != 1:
            raise OpenCLIWebModelError("OPENCLI_WEB_TIMEOUT_RECONCILE_UNKNOWN")
        self._conversation_id = matches[0]
        return self._detail_response(matches[0], wait=True, turn_id=turn_id)

    def _send_and_reconcile(self, prompt: str) -> str:
        try:
            prompt_envelope = json.loads(prompt)
        except json.JSONDecodeError:
            prompt_envelope = {}
        turn_id = (
            str(prompt_envelope.get("turn_id") or "")
            if isinstance(prompt_envelope, Mapping)
            else ""
        )
        argv = [self.executable, "chatgpt", "ask", prompt]
        if self._conversation_id:
            argv.extend(["--conversation", self._conversation_id])
        else:
            argv.append("--new")
        argv.extend([
            "--wait",
            "true",
            "--timeout",
            str(self.timeout_seconds),
            "--site-session",
            self.site_session,
            "-f",
            "json",
        ])
        try:
            stdout = self._run(argv)
        except OpenCLIWebModelError as exc:
            if str(exc) != "OPENCLI_WEB_TIMEOUT" or not turn_id:
                raise
            return self._reconcile_timeout(turn_id)
        conversation_id, _immediate_response = self._extract_ask_result(stdout)
        if self._conversation_id and conversation_id != self._conversation_id:
            raise OpenCLIWebModelError("OPENCLI_WEB_CONVERSATION_ID_MISMATCH")
        self._conversation_id = conversation_id
        return self._detail_response(conversation_id, wait=True, turn_id=turn_id)

    @staticmethod
    def _is_complete_protocol_response(response: str) -> bool:
        try:
            envelope = json.loads(response)
        except json.JSONDecodeError:
            return False
        if not isinstance(envelope, Mapping):
            return False
        kind = envelope.get("type")
        if kind == "final":
            return isinstance(envelope.get("content"), str)
        if kind == "tool_call":
            return isinstance(envelope.get("name"), str) and isinstance(
                envelope.get("arguments"), Mapping
            )
        return False

    def _refresh_protocol_response(self, response: str, *, turn_id: str) -> str:
        if self._is_complete_protocol_response(response) or not self._conversation_id:
            return response
        latest = response
        for _ in range(2):
            latest = self._detail_response(
                self._conversation_id,
                wait=True,
                turn_id=turn_id,
            )
            if self._is_complete_protocol_response(latest):
                return latest
        return latest

    def _repair_protocol_response(self, invalid_response: str) -> str:
        if not self._conversation_id:
            raise OpenCLIWebModelError("OPENCLI_WEB_CONVERSATION_ID_INVALID")
        repair_turn_id = (
            "turn_repair_"
            + hashlib.sha256(
                f"{self._conversation_id}\0{invalid_response}".encode("utf-8")
            ).hexdigest()[:20]
        )
        repair_prompt = json.dumps(
            {
                "protocol": OPENCLI_WEB_PROTOCOL,
                "turn_id": repair_turn_id,
                "instruction": (
                    "Your immediately previous response was incomplete or invalid for the declared JSON "
                    "protocol. Repeat the same intended response as exactly one complete JSON object only. "
                    "Do not execute any tool, do not change the intended tool name or arguments, and do not "
                    "add prose or markdown."
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        response = self._send_and_reconcile(repair_prompt)
        if not self._is_complete_protocol_response(response):
            raise OpenCLIWebModelError("OPENCLI_WEB_PROTOCOL_RESPONSE_INVALID")
        return response

    @staticmethod
    def _response_message(response: str, tools: Sequence[Mapping[str, Any]]) -> AIMessage:
        allowed = {name for tool in tools if (name := _tool_name(tool))}
        try:
            envelope = json.loads(response)
        except json.JSONDecodeError:
            return AIMessage(content=response)
        if not isinstance(envelope, Mapping):
            return AIMessage(content=response)
        kind = envelope.get("type")
        if kind == "final" and isinstance(envelope.get("content"), str):
            return AIMessage(content=str(envelope["content"]))
        if kind != "tool_call":
            return AIMessage(content=response)
        name = envelope.get("name")
        arguments = envelope.get("arguments")
        if not isinstance(name, str) or name not in allowed or not isinstance(arguments, Mapping):
            raise OpenCLIWebModelError("OPENCLI_WEB_TOOL_CALL_INVALID")
        args = dict(arguments)
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": _tool_call_id(name, args, response),
                    "type": "tool_call",
                }
            ],
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
        tools = kwargs.get("tools") or []
        if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
            raise OpenCLIWebModelError("OPENCLI_WEB_TOOLS_INVALID")
        normalized_tools = [dict(tool) for tool in tools if isinstance(tool, Mapping)]
        if len(normalized_tools) != len(tools):
            raise OpenCLIWebModelError("OPENCLI_WEB_TOOLS_INVALID")
        tool_choice = kwargs.get("tool_choice")
        if tool_choice is not None and not isinstance(tool_choice, str):
            raise OpenCLIWebModelError("OPENCLI_WEB_TOOL_CHOICE_INVALID")

        self._select_intelligence_level()
        prompt = self._render_prompt(messages, normalized_tools, tool_choice)
        turn_id = str(json.loads(prompt)["turn_id"])
        response = self._send_and_reconcile(prompt)
        response = self._refresh_protocol_response(response, turn_id=turn_id)
        if not self._is_complete_protocol_response(response):
            response = self._repair_protocol_response(response)
        message = self._response_message(response, normalized_tools)
        return ChatResult(generations=[ChatGeneration(message=message)])
