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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, override

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
# Ключ и токен — значение заголовка: только видимые символы ASCII. Иначе httpx/h11
# отвергнет заголовок, и в тексте ошибки окажется сам секрет.
_CREDENTIAL = re.compile(r"[!-~]+")
# Токен обновляется заранее — за столько секунд до истечения.
_REFRESH_MARGIN = 30
# Срок жизни токена, если ответ OAuth не называет время истечения: 25 минут, в секундах.
_DEFAULT_TOKEN_TTL = 1500
# Статус HTTP, начиная с которого ответ — ошибка.
_ERROR_STATUS = 400


def chat_url(base_url: str) -> str:
    """Адрес чата v2: хвост ``/vN`` базы заменяется на ``/v2`` (без хвоста — база как есть)."""
    base = _VERSION_SUFFIX.sub("/v2", base_url.rstrip("/"))
    return f"{base}/chat/completions"


def _strict_json(candidate: Any) -> bool:
    """Сериализуется ли значение строгим JSON.

    ``json.loads`` (и ``resp.json()``) принимают NaN и Infinity, а тело запроса httpx
    собирает без них — ValueError до отправки.
    """
    try:
        json.dumps(candidate, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _text_part(text: str) -> dict[str, Any]:
    """Текстовая часть сообщения v2."""
    return {"text": text}


def _wire_message(role: str, parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Сообщение v2: роль и список частей."""
    return {"role": role, "content": parts}


def _chat_body(
    model: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
) -> dict[str, Any]:
    """Тело запроса чата v2: модель, история и параметры генерации."""
    return {"model": model, "messages": messages, "model_options": options}


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


def _assistant_payload(msg: Message) -> dict[str, Any]:
    """Сообщение ассистента v2: текст, вызовы функций и состояние вызовов."""
    parts: list[dict[str, Any]] = []
    if msg.content:
        parts.append(_text_part(msg.content))
    for call in msg.function_calls:
        fc: dict[str, Any] = {"name": call.name, "arguments": call.arguments}
        if call.id:
            fc["id"] = call.id
        parts.append({"function_call": fc})
    out = _wire_message("assistant", parts or [_text_part("")])
    if msg.function_calls and msg.state_id:
        out["tools_state_id"] = msg.state_id
    return out


def _append_function_result(
    out: list[dict[str, Any]],
    msg: Message,
    result_object: dict[str, Any],
    state_id: str | None,
) -> None:
    """Добавить в историю ``out`` результат вызова ``msg`` — часть ``function_result``.

    Результаты, идущие подряд, — одно сообщение ``role: "tool"`` с ``tools_state_id``
    их assistant-сообщения.
    """
    function_result: dict[str, Any] = {"name": msg.name or "", "result": result_object}
    if msg.call_id:
        function_result = {"id": msg.call_id, **function_result}
    part = {"function_result": function_result}
    if out and out[-1]["role"] == "tool":
        out[-1]["content"].append(part)
        return
    tool = _wire_message("tool", [part])
    if state_id:
        tool["tools_state_id"] = state_id
    out.append(tool)


def _tool_to_payload(tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }


def _access_token(token_response: Mapping[str, Any]) -> str:
    """``access_token`` из ответа OAuth; его нет или он негоден — :class:`ApiError`."""
    token = token_response.get("access_token")
    # Значения ответа не показываются: токен мог прийти под другим именем поля.
    if not token:
        fields = ", ".join(sorted(str(name) for name in token_response))[:ERROR_BODY_CHARS]
        shown = fields or "нет"
        raise ApiError(f"В ответе OAuth нет access_token (поля ответа: {shown})")
    if not isinstance(token, str) or not _CREDENTIAL.fullmatch(token):
        raise ApiError("access_token в ответе OAuth — не строка из видимых символов ASCII")
    return token


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

    def add(self, reply: Mapping[str, Any]) -> str:
        """Учесть ответ или кусок потока; возвращает пришедший в нём текст."""
        pieces: list[str] = []
        for msg in reply.get("messages") or []:
            if isinstance(msg, Mapping):
                self._add_message(msg, pieces)
        if reply.get("finish_reason"):
            self._finish_reason = str(reply["finish_reason"])
        if reply.get("usage"):
            self._usage = reply["usage"]
        text = self._filter.feed("".join(pieces))
        self._text.append(text)
        return text

    def turn(self, *, complete: bool = True) -> AssistantTurn:
        """Ход ассистента.

        ``complete=False`` — поток кончился без окончания ответа: без причины завершения
        ответ без вызовов — ``"incomplete"``, а не ``"stop"`` (вызовы приходят только в
        окончании и могли потеряться).
        """
        calls = list(self._calls.values())
        message = Message(
            role="assistant",
            content="".join(self._text) + self._filter.pending,  # недособранное начало
            function_calls=calls,
            # Состояние нужно только результатам вызовов.
            state_id=self._state_id if calls else None,
            reasoning="".join(self._reasoning),
        )
        if self._finish_reason:
            finish = self._finish_reason
        elif calls:
            finish = "function_call"
        elif complete:
            finish = "stop"
        else:
            finish = "incomplete"
        return AssistantTurn(
            message=message,
            finish_reason=finish,
            usage=Usage.from_raw(self._usage),
        )

    def _add_message(self, msg: Mapping[str, Any], pieces: list[str]) -> None:
        """Учесть одно сообщение ответа; куски его текста — в ``pieces``."""
        content = msg.get("content")
        if isinstance(content, str):
            content = [_text_part(content)]
        role = msg.get("role")
        if role == "reasoning":
            self._add_reasoning(content)
            return
        if role != "assistant":
            return
        state_id = msg.get("tools_state_id") or msg.get("tool_state_id")
        if state_id:
            self._state_id = str(state_id)
        self._add_parts(content, pieces)

    def _add_reasoning(self, content: Any) -> None:
        """Текст рассуждений из частей сообщения ``role: "reasoning"``."""
        self._reasoning += [
            part["text"]
            for part in content or []
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        ]

    def _add_parts(self, content: Any, pieces: list[str]) -> None:
        """Текст (в ``pieces``) и вызовы функций из частей сообщения ассистента."""
        for part in content or []:
            if not isinstance(part, Mapping):
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                pieces.append(text)
            call = _build_function_call(part.get("function_call"))
            if call is not None:
                self._calls[call.id or f"#{len(self._calls)}"] = call


def _stream_error(event: str | None, chunk: Mapping[str, Any]) -> str | None:
    """Текст ошибки, если событие потока — ошибка API (иначе None)."""
    if event and "error" in event.lower():
        return json.dumps(chunk, ensure_ascii=False)[:ERROR_BODY_CHARS]
    if "messages" in chunk:
        return None
    status = chunk.get("status")
    if chunk.get("error") or (isinstance(status, int) and status >= _ERROR_STATUS):
        return json.dumps(chunk, ensure_ascii=False)[:ERROR_BODY_CHARS]
    return None


def _stream_chunks(lines: Iterable[str]) -> Iterator[tuple[str | None, dict[str, Any] | None]]:
    """События SSE-потока с куском ответа — JSON-объектом; прочие данные пропускаются.

    Маркер ``[DONE]`` — пара ``(событие, None)``, после неё событий нет. Ошибка API в
    потоке — :class:`ApiError`.
    """
    for event, body in iter_sse_events(lines):
        if body.strip() == "[DONE]":
            yield event, None
            return
        try:
            chunk = json.loads(body)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue
        error = _stream_error(event, chunk)
        if error is not None:
            raise ApiError(f"API вернул ошибку в потоке: {error}")
        yield event, chunk


class ApiProvider(HttpProvider):
    """Провайдер API v2: авторизация OAuth или mTLS, чат с function calling и поток SSE."""

    def __init__(self, config: Config, *, transport: httpx.BaseTransport | None = None):
        """``transport`` — подмена HTTP-транспорта (``httpx.MockTransport`` в тестах)."""
        if config.provider != "api":
            raise RuntimeError(
                f"GIGACREW_PROVIDER={config.provider}: этот провайдер работает только с API v2",
            )
        config.require_credentials()
        super().__init__(config, transport=transport)
        self._mtls = config.auth_mode == "mtls"
        self._token: str | None = None
        self._token_exp: float = 0  # unix-время истечения токена

    @override
    def list_models(self) -> list[str]:
        """Чат-модели (``type == "chat"``): эмбеддеры и прочие в выборе не нужны."""
        base_url = self._cfg.base_url
        resp = self._request_with_retry(
            lambda: self._client.get(
                f"{base_url}/models",
                headers=self._auth_headers(),
            ),
        )
        if resp.status_code >= _ERROR_STATUS:
            raise ApiError(
                f"Не удалось получить список моделей ({resp.status_code}): {error_body(resp)}",
            )
        models = self._json(resp).get("data") or []
        return [
            entry["id"]
            for entry in models
            if isinstance(entry, dict) and "id" in entry and entry.get("type") == "chat"
        ]

    @override
    def _chat_url(self) -> str:
        return chat_url(self._cfg.base_url)

    # ------------------------------------------------------------------ #
    # Авторизация
    # ------------------------------------------------------------------ #
    def _ensure_token(self, *, force: bool = False) -> str:
        # Обновляем заранее (за 30 с до истечения), чтобы не словить 401 на лету.
        if not force and self._token and time.time() < self._token_exp - _REFRESH_MARGIN:
            return self._token

        # Хвост ключа из файла с CRLF (``$(cat key.txt)``) или пробел в .env — не часть ключа.
        key = (self._cfg.access_key or "").strip()
        if not _CREDENTIAL.fullmatch(key):
            raise ApiError(
                "GIGACREW_ACCESS_KEY содержит пробелы, управляющие или не-ASCII символы — "
                "проверьте ключ (значение не показывается)",
            )
        token_response = self._json(self._request_token(key))
        token = _access_token(token_response)
        self._token = token
        # expires_at приходит в миллисекундах unix-времени; если нет — живём 25 мин.
        exp_ms = token_response.get("expires_at")
        try:
            self._token_exp = float(exp_ms) / 1000.0 if exp_ms else time.time() + _DEFAULT_TOKEN_TTL
        except (TypeError, ValueError):
            self._token_exp = time.time() + _DEFAULT_TOKEN_TTL
        return token

    def _request_token(self, key: str) -> httpx.Response:
        """Обмен ключа ``key`` на токен OAuth; ответ с ошибкой — :class:`ApiError`."""
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {key}",
        }
        resp = self._request_with_retry(
            lambda: self._client.post(
                self._cfg.auth_url or "",
                headers=headers,
                data={"scope": self._cfg.scope} if self._cfg.scope else {},
            ),
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ApiError(
                f"Не удалось получить токен API ({resp.status_code}): {error_body(resp)}",
            ) from exc
        return resp

    @override
    def _auth_headers(self, *, force: bool = False) -> dict[str, str]:
        """Заголовок авторизации для текущей схемы.

        В режиме mTLS — пустой словарь (клиента авторизует сертификат на
        TLS-уровне). В режиме OAuth — ``Authorization: Bearer <token>``.
        """
        if self._mtls:
            return {}
        return {"Authorization": f"Bearer {self._ensure_token(force=force)}"}

    @property
    @override
    def _refreshable(self) -> bool:
        """Протухший токен (401) обновляется; клиентский сертификат обновлять нечего."""
        return not self._mtls

    # ------------------------------------------------------------------ #
    # Сериализация / разбор
    # ------------------------------------------------------------------ #
    @classmethod
    def _result_object(cls, content: str) -> dict[str, Any]:
        """API требует, чтобы ``result`` части ``function_result`` был JSON-объектом.

        Объект передаётся как есть, всё остальное — текст, но и голые JSON-значения
        (``42``, ``true``, ``[]`` — например, листинг из одного файла ``1``) —
        заворачивается в ``{"result": ...}``: отвергнутое API сообщение осталось бы
        в истории и ломало бы каждый следующий запрос этого чата. Объект с NaN или
        Infinity — тоже текст: строгим JSON тело запроса с ним не собрать.
        """
        text = content or ""
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        return parsed if isinstance(parsed, dict) and _strict_json(parsed) else {"result": text}

    @classmethod
    def _messages_payload(
        cls,
        messages: Sequence[Message],
        *,
        reasoning: bool = True,
    ) -> list[dict[str, Any]]:
        """История для запроса.

        Результаты вызовов одного assistant-сообщения (идут подряд) склеиваются в одно
        сообщение ``role: "tool"`` с его ``tools_state_id``. Рассуждения ответа — сообщение
        ``role: "reasoning"`` перед ним, как в ответе API.
        """
        out: list[dict[str, Any]] = []
        state_id: str | None = None  # состояние assistant-сообщения перед результатами
        for msg in messages:
            if msg.role == "function":
                _append_function_result(out, msg, cls._result_object(msg.content), state_id)
                continue
            if msg.role == "assistant":
                if reasoning and msg.reasoning:
                    out.append(_wire_message("reasoning", [_text_part(msg.reasoning)]))
                out.append(_assistant_payload(msg))
                state_id = msg.state_id if msg.function_calls else None
                continue
            state_id = None
            out.append(_wire_message(msg.role, [_text_part(msg.content or "")]))
        return out

    @override
    def _has_reasoning(self, payload: Mapping[str, Any]) -> bool:
        """Есть ли в теле запроса рассуждения модели."""
        return any(message.get("role") == "reasoning" for message in payload.get("messages") or ())

    @classmethod
    @override
    def _parse_response(cls, body: dict[str, Any]) -> AssistantTurn:
        """Ход ассистента из целого ответа v2; без списка ``messages`` — :class:`ApiError`."""
        if not isinstance(body.get("messages"), list):
            shown = str(body)[:ERROR_BODY_CHARS]
            raise ApiError(f"Неожиданный формат ответа API: {shown}")
        builder = _TurnBuilder()
        builder.add(body)
        return builder.turn()

    @override
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
        payload = _chat_body(
            model or self.model,
            self._messages_payload(messages, reasoning=reasoning),
            options,
        )
        if stream:
            payload["stream"] = True
        if tools:
            specs = [_tool_to_payload(tool) for tool in tools]
            payload["tools"] = [{"functions": {"specifications": specs}}]
            if tool_choice == "none":
                payload["tool_config"] = {"mode": "none"}
            else:
                options["parallel_tool_calls"] = True
        return payload

    @override
    def _probe_payload(self, text: str, model: str) -> dict[str, Any]:
        return _chat_body(
            model,
            [_wire_message("user", [_text_part(text)])],
            {"max_tokens": 1, "temperature": 0},
        )

    # ------------------------------------------------------------------ #
    # Поток SSE
    # ------------------------------------------------------------------ #
    @classmethod
    @override
    def _consume_sse(cls, resp: httpx.Response, emit: Callable[[str], None]) -> AssistantTurn:
        """Разбирает SSE-поток, отдавая текстовые дельты в ``emit``.

        Формат: события ``response.message.delta`` — куски текста в тех же ``messages``,
        что и у обычного ответа; ``response.message.done`` — вызовы функций,
        ``tool_state_id``, ``finish_reason`` и ``usage``. Поток, где не пришло окончание
        ответа (событие ``response.message.done``, маркер ``[DONE]``, причина завершения
        или расход токенов), — незавершённый: шлюз закрыл соединение посреди ответа.
        Обрыв соединения уже после окончания ответа — не ошибка: ответ получен целиком.
        """
        builder = _TurnBuilder()
        done = False
        try:
            for event, chunk in _stream_chunks(resp.iter_lines()):
                if chunk is None:  # маркер [DONE]
                    done = True
                    break
                done = done or event == "response.message.done"
                piece = builder.add(chunk)
                if piece:
                    emit(piece)
        except httpx.TransportError:
            if not (done or builder.finished):
                raise
        return builder.turn(complete=done or builder.finished)
