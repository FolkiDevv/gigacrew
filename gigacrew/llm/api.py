"""HTTP-провайдер модели (API v2).

Реализует авторизацию OAuth (Basic access-key -> Bearer access_token) с
автоматическим обновлением токена по истечении срока (или mTLS), и вызов
chat/completions с поддержкой function calling. Адреса API и выдачи токена
берутся только из конфигурации — значений по умолчанию нет. Повторы, лимит
запросов, поток и замер окна — общие для провайдеров (:mod:`gigacrew.llm.transport`).

Адреса: ``GIGACREW_API_URL`` — база API вида ``…/v1``. Чат идёт на ``…/v2/chat/completions``
(хвост ``/vN`` заменяется на ``/v2``), список моделей — на ``…/v1/models``.

Формат v2 (function calling — несколько вызовов за ответ):
  * сообщение — ``role`` и ``content``: список частей ``{"text": …}``,
    ``{"function_call": {"id", "name", "arguments"}}``, ``{"function_result": …}``;
  * инструменты — ``tools: [{"functions": {"specifications": [...]}}]``, параметры
    генерации — в ``model_options`` (там же ``parallel_tool_calls``);
  * ответ — ``messages``: ``role: "reasoning"`` (рассуждения, не сохраняются) и
    ``role: "assistant"`` с текстом, вызовами и ``tool_state_id``; ``finish_reason`` и
    ``usage`` (``input_tokens``/``output_tokens``) — на верхнем уровне;
  * результаты вызовов возвращаются одним сообщением ``role: "tool"`` с тем же
    ``tools_state_id``: по части ``function_result`` (``id`` вызова, ``name``,
    ``result`` — JSON-объект) на каждый вызов, иначе API отвечает 422;
  * поток — SSE-события ``response.message.delta`` (куски текста) и
    ``response.message.done`` (вызовы, состояние, finish_reason, usage).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

from gigacrew.config import Config
from gigacrew.llm.transport import (
    ERROR_BODY_CHARS,
    ApiError,
    HttpProvider,
    LeakedTagFilter,
    error_body,
    iter_sse_events,
)
from gigacrew.llm.types import AssistantTurn, FunctionCall, Message, ToolChoice, ToolSpec, Usage

__all__ = ["ApiError", "ApiProvider", "chat_url"]


_VERSION_SUFFIX = re.compile(r"/v\d+$")


def chat_url(base_url: str) -> str:
    """Адрес чата v2: хвост ``/vN`` базы заменяется на ``/v2`` (без хвоста — база как есть)."""
    return _VERSION_SUFFIX.sub("/v2", base_url.rstrip("/")) + "/chat/completions"


def _strict_json(value: Any) -> bool:
    """Сериализуется ли значение строгим JSON. ``json.loads`` (и ``resp.json()``) принимают
    NaN и Infinity, а тело запроса httpx собирает без них — ValueError до отправки."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _build_function_call(raw: Any) -> FunctionCall | None:
    if not isinstance(raw, Mapping) or not raw.get("name"):
        return None
    args = raw.get("arguments", {})
    # Обычно arguments — уже объект; подстраховка на случай строки.
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {}
    # Вызов уходит обратно в истории: NaN в аргументах ломал бы каждый следующий запрос.
    if not isinstance(args, dict) or not _strict_json(args):
        args = {}
    call_id = raw.get("id")
    return FunctionCall(name=raw["name"], arguments=args, id=str(call_id) if call_id else None)


def _has_reasoning(payload: Mapping[str, Any]) -> bool:
    """Есть ли в теле запроса рассуждения модели."""
    return any(m.get("role") == "reasoning" for m in payload.get("messages") or ())


class _TurnBuilder:
    """Собирает ход ассистента из ответа v2 — целого или по кускам потока.

    Рассуждения (``role: "reasoning"``) пропускаются. Вызов с уже встречавшимся
    ``id`` заменяет прежний (поток может прислать его повторно).
    """

    def __init__(self) -> None:
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._filter = LeakedTagFilter()
        self._calls: dict[str, FunctionCall] = {}
        self._state_id: str | None = None
        self._finish_reason: str | None = None
        self._usage: Any = None

    @property
    def finished(self) -> bool:
        """Пришло ли окончание ответа: причина завершения или расход токенов."""
        return self._finish_reason is not None or self._usage is not None

    def add(self, data: Mapping[str, Any]) -> str:
        """Учесть ответ или кусок потока; возвращает пришедший в нём текст."""
        pieces: list[str] = []
        for msg in data.get("messages") or []:
            if not isinstance(msg, Mapping):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                content = [{"text": content}]
            if msg.get("role") == "reasoning":
                self._reasoning += [
                    part["text"]
                    for part in content or []
                    if isinstance(part, Mapping) and isinstance(part.get("text"), str)
                ]
                continue
            if msg.get("role") != "assistant":
                continue
            state_id = msg.get("tools_state_id") or msg.get("tool_state_id")
            if state_id:
                self._state_id = str(state_id)
            for part in content or []:
                if not isinstance(part, Mapping):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text:
                    pieces.append(text)
                call = _build_function_call(part.get("function_call"))
                if call is not None:
                    self._calls[call.id or f"#{len(self._calls)}"] = call
        if data.get("finish_reason"):
            self._finish_reason = str(data["finish_reason"])
        if data.get("usage"):
            self._usage = data["usage"]
        text = self._filter.feed("".join(pieces))
        self._text.append(text)
        return text

    def turn(self, *, complete: bool = True) -> AssistantTurn:
        """Ход ассистента. ``complete=False`` — поток кончился без окончания ответа:
        без причины завершения ответ без вызовов — ``"incomplete"``, а не ``"stop"``
        (вызовы приходят только в окончании и могли потеряться)."""
        calls = list(self._calls.values())
        message = Message(
            role="assistant",
            content="".join(self._text) + self._filter.pending,  # недособранное начало
            function_calls=calls,
            # Состояние нужно только результатам вызовов.
            state_id=self._state_id if calls else None,
            reasoning="".join(self._reasoning),
        )
        finish = self._finish_reason or (
            "function_call" if calls else "stop" if complete else "incomplete"
        )
        return AssistantTurn(
            message=message, finish_reason=finish, usage=Usage.from_raw(self._usage)
        )


def _stream_error(event: str | None, data: Mapping[str, Any]) -> str | None:
    """Текст ошибки, если событие потока — ошибка API (иначе None)."""
    if event and "error" in event.lower():
        return json.dumps(data, ensure_ascii=False)[:ERROR_BODY_CHARS]
    if "messages" in data:
        return None
    status = data.get("status")
    if data.get("error") or (isinstance(status, int) and status >= 400):
        return json.dumps(data, ensure_ascii=False)[:ERROR_BODY_CHARS]
    return None


class ApiProvider(HttpProvider):
    def __init__(self, config: Config, *, transport: httpx.BaseTransport | None = None):
        """``transport`` — подмена HTTP-транспорта (``httpx.MockTransport`` в тестах)."""
        if config.provider != "api":
            raise RuntimeError(
                f"GIGACREW_PROVIDER={config.provider}: этот провайдер работает только с API v2"
            )
        config.require_credentials()
        super().__init__(config, transport=transport)
        self._mtls = config.auth_mode == "mtls"
        self._token: str | None = None
        self._token_exp: float = 0.0  # unix-время истечения токена

    def _chat_url(self) -> str:
        return chat_url(self._cfg.base_url)

    # ------------------------------------------------------------------ #
    # Авторизация
    # ------------------------------------------------------------------ #
    def _ensure_token(self, force: bool = False) -> str:
        # Обновляем заранее (за 30 с до истечения), чтобы не словить 401 на лету.
        if not force and self._token and time.time() < self._token_exp - 30:
            return self._token

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {self._cfg.access_key}",
        }
        resp = self._request_with_retry(
            lambda: self._client.post(
                self._cfg.auth_url or "",
                headers=headers,
                data={"scope": self._cfg.scope} if self._cfg.scope else {},
            )
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ApiError(
                f"Не удалось получить токен API ({resp.status_code}): {error_body(resp)}"
            ) from e

        data = self._json(resp)
        token = data.get("access_token")
        if not token:
            raise ApiError(f"В ответе OAuth нет access_token: {str(data)[:500]}")
        self._token = token
        # expires_at приходит в миллисекундах unix-времени; если нет — живём 25 мин.
        exp_ms = data.get("expires_at")
        try:
            self._token_exp = float(exp_ms) / 1000.0 if exp_ms else time.time() + 25 * 60
        except (TypeError, ValueError):
            self._token_exp = time.time() + 25 * 60
        return token

    def _auth_headers(self, *, force: bool = False) -> dict[str, str]:
        """Заголовок авторизации для текущей схемы.

        В режиме mTLS — пустой словарь (клиента авторизует сертификат на
        TLS-уровне). В режиме OAuth — ``Authorization: Bearer <token>``.
        """
        if self._mtls:
            return {}
        return {"Authorization": f"Bearer {self._ensure_token(force=force)}"}

    @property
    def _refreshable(self) -> bool:
        """Протухший токен (401) обновляется; клиентский сертификат обновлять нечего."""
        return not self._mtls

    # ------------------------------------------------------------------ #
    # Сериализация / разбор
    # ------------------------------------------------------------------ #
    @staticmethod
    def _result_object(content: str) -> dict[str, Any]:
        """API требует, чтобы ``result`` части ``function_result`` был JSON-объектом.

        Объект передаётся как есть, всё остальное — текст, но и голые JSON-значения
        (``42``, ``true``, ``[]`` — например, листинг из одного файла ``1``) —
        заворачивается в ``{"result": ...}``: отвергнутое API сообщение осталось бы
        в истории и ломало бы каждый следующий запрос этого чата. Объект с NaN или
        Infinity — тоже текст: строгим JSON тело запроса с ним не собрать."""
        text = content or ""
        try:
            value = json.loads(text)
        except ValueError:
            value = None
        return value if isinstance(value, dict) and _strict_json(value) else {"result": text}

    @staticmethod
    def _assistant_payload(msg: Message) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []
        if msg.content:
            parts.append({"text": msg.content})
        for call in msg.function_calls:
            fc: dict[str, Any] = {"name": call.name, "arguments": call.arguments}
            if call.id:
                fc["id"] = call.id
            parts.append({"function_call": fc})
        out: dict[str, Any] = {"role": "assistant", "content": parts or [{"text": ""}]}
        if msg.function_calls and msg.state_id:
            out["tools_state_id"] = msg.state_id
        return out

    @classmethod
    def _messages_payload(
        cls, messages: Sequence[Message], *, reasoning: bool = True
    ) -> list[dict[str, Any]]:
        """История для запроса. Результаты вызовов одного assistant-сообщения (идут
        подряд) склеиваются в одно сообщение ``role: "tool"`` с его ``tools_state_id``.
        Рассуждения ответа — сообщение ``role: "reasoning"`` перед ним, как в ответе API."""
        out: list[dict[str, Any]] = []
        state_id: str | None = None  # состояние assistant-сообщения перед результатами
        for msg in messages:
            if msg.role == "function":
                result: dict[str, Any] = {
                    "name": msg.name or "",
                    "result": cls._result_object(msg.content),
                }
                if msg.call_id:
                    result = {"id": msg.call_id, **result}
                part = {"function_result": result}
                if out and out[-1]["role"] == "tool":
                    out[-1]["content"].append(part)
                else:
                    tool: dict[str, Any] = {"role": "tool", "content": [part]}
                    if state_id:
                        tool["tools_state_id"] = state_id
                    out.append(tool)
                continue
            if msg.role == "assistant":
                if reasoning and msg.reasoning:
                    out.append({"role": "reasoning", "content": [{"text": msg.reasoning}]})
                out.append(cls._assistant_payload(msg))
                state_id = msg.state_id if msg.function_calls else None
                continue
            state_id = None
            out.append({"role": msg.role, "content": [{"text": msg.content or ""}]})
        return out

    @staticmethod
    def _tool_to_payload(tool: ToolSpec) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }

    _has_reasoning = staticmethod(_has_reasoning)

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> AssistantTurn:
        if not isinstance(data.get("messages"), list):
            raise ApiError(f"Неожиданный формат ответа API: {str(data)[:ERROR_BODY_CHARS]}")
        builder = _TurnBuilder()
        builder.add(data)
        return builder.turn()

    def _payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None,
        *,
        model: str | None,
        temperature: float,
        stream: bool = False,
        tool_choice: ToolChoice = "auto",
        reasoning: bool = True,
    ) -> dict[str, Any]:
        """Тело запроса чата; ``reasoning=False`` — без рассуждений модели."""
        options: dict[str, Any] = {"temperature": temperature}
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": self._messages_payload(messages, reasoning=reasoning),
            "model_options": options,
        }
        if stream:
            payload["stream"] = True
        if tools:
            specs = [self._tool_to_payload(t) for t in tools]
            payload["tools"] = [{"functions": {"specifications": specs}}]
            if tool_choice == "none":
                payload["tool_config"] = {"mode": "none"}
            else:
                options["parallel_tool_calls"] = True
        return payload

    def _probe_payload(self, text: str, model: str) -> dict[str, Any]:
        return {
            "model": model,
            "messages": [{"role": "user", "content": [{"text": text}]}],
            "model_options": {"max_tokens": 1, "temperature": 0},
        }

    # ------------------------------------------------------------------ #
    # Поток (SSE)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _consume_sse(resp: httpx.Response, emit: Callable[[str], None]) -> AssistantTurn:
        """Разбирает SSE-поток, отдавая текстовые дельты в ``emit``.

        Формат: события ``response.message.delta`` — куски текста в тех же ``messages``,
        что и у обычного ответа; ``response.message.done`` — вызовы функций,
        ``tool_state_id``, ``finish_reason`` и ``usage``. Поток, где не пришло окончание
        ответа (событие ``response.message.done``, маркер ``[DONE]``, причина завершения
        или расход токенов), — незавершённый: шлюз закрыл соединение посреди ответа.
        """
        builder = _TurnBuilder()
        done = False
        for event, data in iter_sse_events(resp.iter_lines()):
            if data.strip() == "[DONE]":
                done = True
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            error = _stream_error(event, chunk)
            if error is not None:
                raise ApiError(f"API вернул ошибку в потоке: {error}")
            done = done or event == "response.message.done"
            piece = builder.add(chunk)
            if piece:
                emit(piece)
        return builder.turn(complete=done or builder.finished)

    def list_models(self) -> list[str]:
        """Чат-модели (``type == "chat"``): эмбеддеры и прочие в выборе не нужны."""
        resp = self._request_with_retry(
            lambda: self._client.get(
                f"{self._cfg.base_url}/models",
                headers=self._auth_headers(),
            )
        )
        if resp.status_code >= 400:
            raise ApiError(
                f"Не удалось получить список моделей ({resp.status_code}): {error_body(resp)}"
            )
        models = self._json(resp).get("data") or []
        return [
            m["id"] for m in models if isinstance(m, dict) and "id" in m and m.get("type") == "chat"
        ]
