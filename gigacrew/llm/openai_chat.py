"""Кодек формата OpenAI Chat Completions: тело запроса и разбор ответа (целого и потока).

Запрос:
  * ``messages`` — строки ``content``; ход ассистента с вызовами — ``tool_calls``
    (``arguments`` — JSON-строка) и ``content: null``, рассуждения — ``reasoning_content``;
    результат вызова — ``role: "tool"`` с ``tool_call_id`` вызова;
  * ``tools`` — ``[{"type": "function", "function": {...}}]``, ``tool_choice`` — только
    вместе с ними; ``stream_options.include_usage`` — только в потоке.

Ответ — ``choices[0]`` (``delta`` в потоке, ``message`` в целом ответе): ``content`` (текст),
``reasoning_content``/``reasoning``/``thinking`` (рассуждения), ``tool_calls`` (вызовы,
в потоке — по кускам), ``finish_reason``; ``usage`` — отдельным последним куском. Поток —
SSE (``data: …``, ``data: [DONE]``), NDJSON или цельный JSON.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import httpx

from gigacrew.llm.transport import ERROR_BODY_CHARS, ApiError, LeakedTagFilter, iter_sse_events
from gigacrew.llm.types import AssistantTurn, FunctionCall, Message, ToolChoice, ToolSpec, Usage

# Результат вызова, которого нет в истории: API не принимает вызовы без результатов.
MISSING_RESULT = "No result: the call was not executed."
_REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")
_SSE_PREFIXES = ("data:", "event:", "id:", "retry:", ":")
_NDJSON_TYPES = ("application/x-ndjson", "application/ndjson", "application/stream+json")
# Ключи и значения формата, общие для запроса и ответа.
_ROLE = "role"
_CONTENT = "content"
_FUNCTION = "function"
# Длина случайной или хешированной части id вызова (шестнадцатеричные символы).
_CALL_ID_HEX = 24
# Причины завершения: ответ упёрся в лимит длины; модель вызывает инструменты.
_LENGTH_REASONS = frozenset(("length", "max_tokens"))
_CALL_REASONS = frozenset(("tool_calls", "function_call"))


# ------------------------------------------------------------------ #
# Запрос
# ------------------------------------------------------------------ #
def _call_id(call: FunctionCall, message: int, position: int) -> str:
    """Id вызова для запроса: свой у вызова, иначе стабильный по месту и содержимому.

    Стабильный id (чаты, сохранённые без id) — тот же у вызова и его результата.
    """
    if call.id:
        return call.id
    raw = json.dumps([message, position, call.name, call.arguments], sort_keys=True, default=str)
    # Хеш — только стабильный id, не защита.
    digest = hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()[:_CALL_ID_HEX]
    return f"call_{digest}"


def _assistant(msg: Message, ids: list[str], *, reasoning: bool) -> dict[str, Any]:
    """Ход ассистента для запроса: текст, вызовы с id ``ids`` и рассуждения."""
    out: dict[str, Any] = {_ROLE: "assistant", _CONTENT: msg.content or None}
    if msg.function_calls:
        out["tool_calls"] = [
            {
                "id": call_id,
                "type": _FUNCTION,
                _FUNCTION: {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call, call_id in zip(msg.function_calls, ids, strict=True)
        ]
    elif out[_CONTENT] is None:
        out[_CONTENT] = ""
    if reasoning and msg.reasoning:
        out["reasoning_content"] = msg.reasoning
    return out


class _History:
    """История для запроса, собираемая по сообщениям (:func:`messages_payload`)."""

    def __init__(self, *, reasoning: bool) -> None:
        self.payload: list[dict[str, Any]] = []
        self._reasoning = reasoning
        self._pending: list[tuple[str, str]] = []  # (id, имя) вызовов без результата

    def add(self, index: int, msg: Message) -> None:
        """Добавляет сообщение ``msg``; ``index`` — его номер в истории."""
        if msg.role == "function":
            self._add_result(msg)
            return
        self.close_pending()
        if msg.role == "assistant":
            self._add_assistant(index, msg)
        elif msg.role == "user":
            self._add_user(msg.content or "")
        else:
            self.payload.append({_ROLE: msg.role, _CONTENT: msg.content or ""})

    def close_pending(self) -> None:
        """Закрывает заглушкой вызовы, оставшиеся без результата."""
        self.payload.extend(
            {_ROLE: "tool", "tool_call_id": call_id, _CONTENT: MISSING_RESULT}
            for call_id, _ in self._pending
        )
        self._pending.clear()

    def _add_assistant(self, index: int, msg: Message) -> None:
        calls = msg.function_calls
        ids = [_call_id(call, index, position) for position, call in enumerate(calls)]
        self.payload.append(_assistant(msg, ids, reasoning=self._reasoning))
        self._pending = [(call_id, call.name) for call_id, call in zip(ids, calls, strict=True)]

    def _add_result(self, msg: Message) -> None:
        position = next(
            (
                pending_position
                for pending_position, (pending_id, _) in enumerate(self._pending)
                if pending_id == msg.call_id
            ),
            None,
        )
        if position is None and self._pending:
            position = 0
        if position is None:
            name = msg.name or "a tool call"
            self._add_user(f"Result of {name}:\n{msg.content}")
            return
        call_id, _ = self._pending.pop(position)
        self.payload.append({_ROLE: "tool", "tool_call_id": call_id, _CONTENT: msg.content or ""})

    def _add_user(self, text: str) -> None:
        if self.payload and self.payload[-1][_ROLE] == "user":
            previous = self.payload[-1][_CONTENT]
            self.payload[-1][_CONTENT] = f"{previous}\n\n{text}"
        else:
            self.payload.append({_ROLE: "user", _CONTENT: text})


def messages_payload(
    messages: Sequence[Message],
    *,
    reasoning: bool = True,
) -> list[dict[str, Any]]:
    """История для запроса.

    Результат вызова получает ``tool_call_id`` своего вызова: по ``call_id``, иначе по
    порядку (результаты идут за вызовами по одному и по порядку). Вызовы без результата
    закрываются заглушкой, результат без вызова уходит текстом пользователя, подряд идущие
    сообщения пользователя склеиваются — иначе запрос не примет строгий шаблон чата.
    ``reasoning=False`` — без рассуждений модели.
    """
    history = _History(reasoning=reasoning)
    for index, msg in enumerate(messages):
        history.add(index, msg)
    history.close_pending()
    return history.payload


def tools_payload(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    """Схемы инструментов для запроса: ``[{"type": "function", "function": {...}}]``."""
    return [
        {
            "type": _FUNCTION,
            _FUNCTION: {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


# Параметры — поля тела запроса, их передают по имени: объединять их в объект незачем.
def chat_payload(  # noqa: WPS211
    messages: Sequence[Message],
    tools: Sequence[ToolSpec] | None,
    *,
    model: str,
    temperature: float,
    stream: bool,
    tool_choice: ToolChoice = "auto",
    reasoning: bool = True,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Тело запроса chat/completions.

    ``options`` — параметры генерации модели (из каталога): поля gigacrew (модель,
    температура, поток) важнее.
    """
    payload: dict[str, Any] = dict(options or {})
    payload.update(
        model=model,
        messages=messages_payload(messages, reasoning=reasoning),
        temperature=temperature,
        stream=stream,
    )
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if tools:
        payload["tools"] = tools_payload(tools)
        payload["tool_choice"] = tool_choice
    return payload


def probe_payload(text: str, model: str, *, stream: bool) -> dict[str, Any]:
    """Проба замера окна: ``text`` одним сообщением, ответ в 1 токен."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{_ROLE: "user", _CONTENT: text}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def has_reasoning(payload: Mapping[str, Any]) -> bool:
    """Есть ли в теле запроса рассуждения модели."""
    return any(
        isinstance(message, Mapping) and bool(message.get("reasoning_content"))
        for message in payload.get("messages") or ()
    )


# ------------------------------------------------------------------ #
# Ответ
# ------------------------------------------------------------------ #
def parse_arguments(text: str) -> dict[str, Any]:
    """Аргументы вызова из JSON-строки; не объект или мусор — ``{}``.

    Переводы строк внутри строк допускаются; если сервер повторил аргументы целиком
    (``{…}{…}``), берётся первый объект.
    """
    text = text.strip()
    if not text:
        return {}
    decoder = json.JSONDecoder(strict=False)
    try:
        decoded = decoder.decode(text)
    except ValueError:
        decoded = _leading_value(decoder, text)
    return decoded if isinstance(decoded, dict) else {}


def _leading_value(decoder: json.JSONDecoder, text: str) -> Any:
    """Первое JSON-значение в начале ``text`` (за ним — лишнее); мусор — None."""
    try:
        decoded, _end = decoder.raw_decode(text)
    except ValueError:
        return None
    return decoded


def _text_of(content: Any) -> str:
    """Текст ``content``: строка или список частей ``{"type": "text", "text": …}``."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"]
            for part in content
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        )
    return ""


def _error_text(chunk: Mapping[str, Any], event: str | None = None) -> str | None:
    """Текст ошибки, если кусок ответа — ошибка API (иначе None)."""
    if (event and "error" in event.lower()) or chunk.get("error") or chunk.get("object") == "error":
        return json.dumps(chunk, ensure_ascii=False)[:ERROR_BODY_CHARS]
    return None


def _first_choice(chunk: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Вариант ответа с ``index`` 0 из ``choices`` куска; нет — None."""
    choices = chunk.get("choices")
    for choice in choices if isinstance(choices, list) else ():
        if isinstance(choice, Mapping) and choice.get("index", 0) == 0:
            return choice
    return None


def _finish(
    reason: str | None,
    calls: list[FunctionCall],
    *,
    ended: bool,
) -> tuple[str, list[FunctionCall]]:
    """Причина завершения хода и вызовы, которые можно выполнять.

    ``ended`` — ответ дошёл до конца: без причины завершения он закончен, а не оборван.
    Оборванный ответ (лимит длины, поток без окончания, фильтр) теряет вызовы.
    """
    if reason in _LENGTH_REASONS:
        return "length", []
    if reason is None:
        if ended:
            return ("function_call" if calls else "stop"), calls
        return "incomplete", []
    if reason in _CALL_REASONS:
        return ("function_call" if calls else "incomplete"), calls
    if reason == "content_filter":
        return reason, []
    # stop и нестандартные причины
    return ("function_call" if calls else reason), calls


class _CallParts:
    """Вызов, собираемый по кускам потока."""

    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.arguments: list[str] = []
        self.parsed: dict[str, Any] | None = None  # аргументы пришли объектом

    def call(self) -> FunctionCall | None:
        """Собранный вызов; без имени — None."""
        if not self.name:
            return None
        arguments = parse_arguments("".join(self.arguments)) if self.parsed is None else self.parsed
        call_id = self.id
        if not call_id:
            suffix = uuid.uuid4().hex[:_CALL_ID_HEX]
            call_id = f"call_{suffix}"
        return FunctionCall(name=self.name, arguments=arguments, id=call_id)

    def add_function(self, function: Mapping[str, Any]) -> None:
        """Учитывает поле ``function`` куска: имя и часть аргументов (или их объект)."""
        name = function.get("name")
        if isinstance(name, str) and name and not self.name:
            self.name = name  # имя не склеивается: сервер может повторять его в кусках
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            self.arguments.append(arguments)
        elif isinstance(arguments, Mapping):
            self.parsed = dict(arguments)


class ChunkBuilder:
    """Собирает ход ассистента из кусков потока (или из целого ответа)."""

    def __init__(self) -> None:
        """Пустой ход: ни текста, ни вызовов, ни причины завершения."""
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._filter = LeakedTagFilter()
        self._calls: list[_CallParts] = []
        self._by_index: dict[int, _CallParts] = {}
        self._finish_reason: str | None = None
        self._usage: Mapping[str, Any] | None = None

    @property
    def finished(self) -> bool:
        """Пришло ли окончание ответа: причина завершения или расход токенов."""
        return self._finish_reason is not None or self._usage is not None

    def add(self, chunk: Mapping[str, Any]) -> str:
        """Учесть кусок потока или целый ответ; возвращает текст, который можно показать."""
        usage = chunk.get("usage")
        if isinstance(usage, Mapping) and usage:
            self._usage = usage
        choice = _first_choice(chunk)
        if choice is None:
            return ""
        part = choice.get("delta")
        if not isinstance(part, Mapping):
            part = choice.get("message")
        text = self._add_part(part) if isinstance(part, Mapping) else ""
        reason = choice.get("finish_reason")
        if isinstance(reason, str) and reason:
            self._finish_reason = reason
        return text

    def turn(self, *, done: bool = True) -> AssistantTurn:
        """Ход ассистента.

        ``done`` — ответ дошёл до конца (``[DONE]``, конец целого ответа): без причины
        завершения он закончен, а не оборван.

        Оборванный ответ (лимит длины, поток без окончания, фильтр) теряет вызовы: их
        аргументы могли не дойти, а агент выполняет вызовы при любой причине завершения.
        """
        assembled = (parts.call() for parts in self._calls)
        calls = [call for call in assembled if call is not None]
        finish, calls = _finish(self._finish_reason, calls, ended=done or self.finished)
        message = Message(
            role="assistant",
            content="".join(self._text) + self._filter.pending,  # недособранное начало
            function_calls=calls,
            reasoning="".join(self._reasoning),
        )
        return AssistantTurn(
            message=message,
            finish_reason=finish,
            usage=Usage.from_raw(self._usage),
        )

    def _add_part(self, part: Mapping[str, Any]) -> str:
        """Рассуждения, вызовы и текст из ``delta``/``message``; возвращает текст для показа."""
        for key in _REASONING_KEYS:
            reasoning = part.get(key)
            if isinstance(reasoning, str) and reasoning:
                self._reasoning.append(reasoning)
                break
        calls = part.get("tool_calls")
        for raw in calls if isinstance(calls, list) else ():
            if isinstance(raw, Mapping):
                self._add_call(raw)
        text = self._filter.feed(_text_of(part.get(_CONTENT)))
        self._text.append(text)
        return text

    def _add_call(self, raw: Mapping[str, Any]) -> None:
        """Кусок вызова: по ``index``; без него — новый вызов при новом id, иначе к последнему.

        Тот же ``index`` с другим id и именем — тоже новый вызов: некоторые серверы дают
        index 0 каждому параллельному вызову, и второй иначе приклеился бы к первому.
        """
        index = raw.get("index")
        call_id = str(raw["id"]) if raw.get("id") else ""
        function = raw.get(_FUNCTION)
        named = isinstance(function, Mapping) and bool(function.get("name"))
        parts = self._parts_for(index, call_id, named=named)
        if call_id and not parts.id:
            parts.id = call_id
        if isinstance(function, Mapping):
            parts.add_function(function)

    def _parts_for(self, index: object, call_id: str, *, named: bool) -> _CallParts:
        """Вызов, к которому относится кусок: уже собираемый или новый."""
        parts: _CallParts | None = None
        if isinstance(index, int) and not isinstance(index, bool):
            parts = self._by_index.get(index)
            if parts is None or (named and call_id and parts.id and parts.id != call_id):
                parts = self._new_call()
                self._by_index[index] = parts
        elif call_id:
            parts = next((known for known in self._calls if known.id == call_id), None)
            if parts is None and self._calls and not self._calls[-1].id:
                parts = self._calls[-1]  # id пришёл не в первом куске
        elif self._calls:
            parts = self._calls[-1]
        if parts is None:
            parts = self._new_call()
        return parts

    def _new_call(self) -> _CallParts:
        parts = _CallParts()
        self._calls.append(parts)
        return parts


def parse_response(body: Mapping[str, Any]) -> AssistantTurn:
    """Ход ассистента из целого ответа."""
    error = _error_text(body)
    if error is not None:
        raise ApiError(f"API вернул ошибку: {error}")
    if not isinstance(body.get("choices"), list):
        shown = str(body)[:ERROR_BODY_CHARS]
        raise ApiError(f"Неожиданный формат ответа API: {shown}")
    builder = ChunkBuilder()
    builder.add(body)
    return builder.turn(done=True)


def _is_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text, strict=False), dict)
    except ValueError:
        return False


def _stream_kind(content_type: str, first: str) -> str:
    """Формат тела: ``sse``, ``ndjson`` или ``json`` (цельный ответ).

    Определяется по content-type и первой непустой строке.
    """
    kind = content_type.split(";", 1)[0].strip().lower()
    if kind == "text/event-stream" or first.lstrip().startswith(_SSE_PREFIXES):
        return "sse"
    if kind in _NDJSON_TYPES:
        return "ndjson"
    if kind == "application/json" or not _is_object(first.strip()):
        return "json"
    return "ndjson"


def _chunks(kind: str, lines: Iterable[str]) -> Iterator[tuple[str | None, str]]:
    """Куски тела: (имя события SSE или None, данные)."""
    if kind == "sse":
        yield from iter_sse_events(lines)
    elif kind == "ndjson":
        for line in lines:
            text = line.strip().removeprefix("data:").strip()
            if text:
                yield None, text
    else:
        yield None, "\n".join(lines)


_DONE = object()  # маркер конца потока ``[DONE]``


def _decode(body: str, kind: str) -> list[Any]:
    """Объекты из данных куска (``_DONE`` — конец потока).

    Строки ``data:`` без пустой строки между событиями (так шлют некоторые серверы) SSE
    склеивает в одно событие — тогда каждая строка разбирается отдельно. Не-JSON в целом
    ответе — ошибка, в потоке — пропускается.
    """
    text = body.strip()
    if text == "[DONE]":
        return [_DONE]
    try:
        return [json.loads(text, strict=False)]
    except json.JSONDecodeError as exc:
        if kind == "json":
            shown = text[:ERROR_BODY_CHARS]
            raise ApiError(f"API вернул не-JSON ответ: {shown}") from exc
        if "\n" not in text:
            return []
        return [decoded for line in text.splitlines() for decoded in _decode(line, "ndjson")]


def _read_head(lines: Iterator[str]) -> list[str]:
    """Строки до первой непустой включительно: по ней виден формат тела."""
    head: list[str] = []
    for line in lines:
        head.append(line)
        if line.strip():
            break
    return head


def _check_chunk(chunk: Mapping[str, Any], event: str | None, kind: str) -> None:
    """Кусок — ошибка API или (в целом ответе) не ответ чата: :class:`ApiError`."""
    error = _error_text(chunk, event)
    if error is not None:
        raise ApiError(f"API вернул ошибку в потоке: {error}")
    if kind == "json" and not isinstance(chunk.get("choices"), list):
        shown = str(chunk)[:ERROR_BODY_CHARS]
        raise ApiError(f"Неожиданный формат ответа API: {shown}")


def _feed(
    builder: ChunkBuilder,
    event: str | None,
    body: str,
    *,
    kind: str,
    emit: Callable[[str], None],
) -> bool:
    """Учитывает одно событие тела, отдавая текст в ``emit``; True — пришёл ``[DONE]``."""
    for chunk in _decode(body, kind):
        if chunk is _DONE:
            return True
        if not isinstance(chunk, Mapping):
            continue
        _check_chunk(chunk, event, kind)
        piece = builder.add(chunk)
        if piece:
            emit(piece)
    return False


def consume_stream(
    lines: Iterable[str],
    emit: Callable[[str], None],
    *,
    content_type: str = "",
) -> AssistantTurn:
    """Разбирает тело ответа (поток или целый ответ), отдавая куски текста в ``emit``.

    Поток без окончания (``[DONE]``, причина завершения или расход токенов) — оборван:
    сервер закрыл соединение посреди ответа. Обрыв соединения уже после окончания
    ответа — не ошибка: ответ получен целиком (не хватать может разве что расхода токенов).
    """
    remaining = iter(lines)
    head = _read_head(remaining)  # до первой непустой строки: по ней виден формат
    kind = _stream_kind(content_type, head[-1] if head else "")
    builder = ChunkBuilder()
    done = kind == "json"
    try:
        for event, body in _chunks(kind, itertools.chain(head, remaining)):
            if _feed(builder, event, body, kind=kind, emit=emit):
                done = True
            if done and kind != "json":
                break
    except httpx.TransportError:
        if not builder.finished:
            raise
    return builder.turn(done=done)
