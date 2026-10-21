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

from gigacrew.llm.transport import ERROR_BODY_CHARS, ApiError, LeakedTagFilter, iter_sse_events
from gigacrew.llm.types import AssistantTurn, FunctionCall, Message, ToolChoice, ToolSpec, Usage

# Результат вызова, которого нет в истории: API не принимает вызовы без результатов.
MISSING_RESULT = "No result: the call was not executed."
_REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")
_SSE_PREFIXES = ("data:", "event:", "id:", "retry:", ":")
_NDJSON_TYPES = ("application/x-ndjson", "application/ndjson", "application/stream+json")


# ------------------------------------------------------------------ #
# Запрос
# ------------------------------------------------------------------ #
def _call_id(call: FunctionCall, message: int, position: int) -> str:
    """Id вызова для запроса: свой у вызова, иначе стабильный по месту и содержимому
    (чаты, сохранённые без id) — тот же у вызова и его результата."""
    if call.id:
        return call.id
    raw = json.dumps([message, position, call.name, call.arguments], sort_keys=True, default=str)
    return f"call_{hashlib.sha1(raw.encode()).hexdigest()[:24]}"


def _assistant(msg: Message, ids: list[str], *, reasoning: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": msg.content or None}
    if msg.function_calls:
        out["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call, call_id in zip(msg.function_calls, ids, strict=True)
        ]
    elif out["content"] is None:
        out["content"] = ""
    if reasoning and msg.reasoning:
        out["reasoning_content"] = msg.reasoning
    return out


def messages_payload(
    messages: Sequence[Message], *, reasoning: bool = True
) -> list[dict[str, Any]]:
    """История для запроса.

    Результат вызова получает ``tool_call_id`` своего вызова: по ``call_id``, иначе по
    порядку (результаты идут за вызовами по одному и по порядку). Вызовы без результата
    закрываются заглушкой, результат без вызова уходит текстом пользователя, подряд идущие
    сообщения пользователя склеиваются — иначе запрос не примет строгий шаблон чата.
    ``reasoning=False`` — без рассуждений модели.
    """
    out: list[dict[str, Any]] = []
    pending: list[tuple[str, str]] = []  # (id, имя) вызовов без результата

    def close_pending() -> None:
        out.extend(
            {"role": "tool", "tool_call_id": call_id, "content": MISSING_RESULT}
            for call_id, _name in pending
        )
        pending.clear()

    def add_user(text: str) -> None:
        if out and out[-1]["role"] == "user":
            out[-1]["content"] = f"{out[-1]['content']}\n\n{text}"
        else:
            out.append({"role": "user", "content": text})

    for index, msg in enumerate(messages):
        if msg.role == "function":
            match = next((i for i, (cid, _) in enumerate(pending) if cid == msg.call_id), None)
            if match is None and pending:
                match = 0
            if match is None:
                add_user(f"Result of {msg.name or 'a tool call'}:\n{msg.content}")
                continue
            call_id, _name = pending.pop(match)
            out.append({"role": "tool", "tool_call_id": call_id, "content": msg.content or ""})
            continue
        close_pending()
        if msg.role == "assistant":
            ids = [_call_id(call, index, k) for k, call in enumerate(msg.function_calls)]
            out.append(_assistant(msg, ids, reasoning=reasoning))
            pending = [(cid, call.name) for cid, call in zip(ids, msg.function_calls, strict=True)]
        elif msg.role == "user":
            add_user(msg.content or "")
        else:
            out.append({"role": msg.role, "content": msg.content or ""})
    close_pending()
    return out


def tools_payload(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def chat_payload(
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
    """Тело запроса chat/completions. ``options`` — параметры генерации модели (из
    каталога): поля gigacrew (модель, температура, поток) важнее."""
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
        "messages": [{"role": "user", "content": text}],
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
        isinstance(m, Mapping) and bool(m.get("reasoning_content"))
        for m in payload.get("messages") or ()
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
        value = decoder.decode(text)
    except ValueError:
        try:
            value, _end = decoder.raw_decode(text)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


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


class _CallParts:
    """Вызов, собираемый по кускам потока."""

    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.arguments: list[str] = []
        self.parsed: dict[str, Any] | None = None  # аргументы пришли объектом

    def call(self) -> FunctionCall | None:
        if not self.name:
            return None
        arguments = (
            self.parsed if self.parsed is not None else parse_arguments("".join(self.arguments))
        )
        return FunctionCall(
            name=self.name, arguments=arguments, id=self.id or f"call_{uuid.uuid4().hex[:24]}"
        )


class ChunkBuilder:
    """Собирает ход ассистента из кусков потока (или из целого ответа)."""

    def __init__(self) -> None:
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
        choice = self._first_choice(chunk)
        if choice is None:
            return ""
        part = choice.get("delta")
        if not isinstance(part, Mapping):
            part = choice.get("message")
        text = ""
        if isinstance(part, Mapping):
            for key in _REASONING_KEYS:
                value = part.get(key)
                if isinstance(value, str) and value:
                    self._reasoning.append(value)
                    break
            calls = part.get("tool_calls")
            for raw in calls if isinstance(calls, list) else ():
                if isinstance(raw, Mapping):
                    self._add_call(raw)
            text = self._filter.feed(_text_of(part.get("content")))
            self._text.append(text)
        reason = choice.get("finish_reason")
        if isinstance(reason, str) and reason:
            self._finish_reason = reason
        return text

    @staticmethod
    def _first_choice(chunk: Mapping[str, Any]) -> Mapping[str, Any] | None:
        choices = chunk.get("choices")
        for choice in choices if isinstance(choices, list) else ():
            if isinstance(choice, Mapping) and choice.get("index", 0) == 0:
                return choice
        return None

    def _add_call(self, raw: Mapping[str, Any]) -> None:
        """Кусок вызова: по ``index``; без него — новый вызов при новом id, иначе к последнему."""
        index = raw.get("index")
        call_id = str(raw["id"]) if raw.get("id") else ""
        parts: _CallParts | None = None
        if isinstance(index, int) and not isinstance(index, bool):
            parts = self._by_index.get(index)
            if parts is None:
                parts = self._by_index[index] = self._new_call()
        elif call_id:
            parts = next((p for p in self._calls if p.id == call_id), None)
            if parts is None and self._calls and not self._calls[-1].id:
                parts = self._calls[-1]  # id пришёл не в первом куске
        elif self._calls:
            parts = self._calls[-1]
        if parts is None:
            parts = self._new_call()
        if call_id and not parts.id:
            parts.id = call_id
        function = raw.get("function")
        if not isinstance(function, Mapping):
            return
        name = function.get("name")
        if isinstance(name, str) and name and not parts.name:
            parts.name = name  # имя не склеивается: сервер может повторять его в кусках
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            parts.arguments.append(arguments)
        elif isinstance(arguments, Mapping):
            parts.parsed = dict(arguments)

    def _new_call(self) -> _CallParts:
        parts = _CallParts()
        self._calls.append(parts)
        return parts

    def turn(self, *, done: bool = True) -> AssistantTurn:
        """Ход ассистента. ``done`` — ответ дошёл до конца (``[DONE]``, конец целого
        ответа): без причины завершения он закончен, а не оборван.

        Оборванный ответ (лимит длины, поток без окончания, фильтр) теряет вызовы: их
        аргументы могли не дойти, а агент выполняет вызовы при любой причине завершения.
        """
        calls = [call for call in (parts.call() for parts in self._calls) if call is not None]
        reason = self._finish_reason
        if reason in ("length", "max_tokens"):
            finish, calls = "length", []
        elif reason is None:
            if done or self.finished:
                finish = "function_call" if calls else "stop"
            else:
                finish, calls = "incomplete", []
        elif reason in ("tool_calls", "function_call"):
            finish = "function_call" if calls else "incomplete"
        elif reason == "content_filter":
            finish, calls = reason, []
        else:  # stop и нестандартные причины
            finish = "function_call" if calls else reason
        message = Message(
            role="assistant",
            content="".join(self._text) + self._filter.pending,  # недособранное начало
            function_calls=calls,
            reasoning="".join(self._reasoning),
        )
        return AssistantTurn(
            message=message, finish_reason=finish, usage=Usage.from_raw(self._usage)
        )


def parse_response(data: Mapping[str, Any]) -> AssistantTurn:
    """Ход ассистента из целого ответа."""
    error = _error_text(data)
    if error is not None:
        raise ApiError(f"API вернул ошибку: {error}")
    if not isinstance(data.get("choices"), list):
        raise ApiError(f"Неожиданный формат ответа API: {str(data)[:ERROR_BODY_CHARS]}")
    builder = ChunkBuilder()
    builder.add(data)
    return builder.turn(done=True)


def _is_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text, strict=False), dict)
    except ValueError:
        return False


def _stream_kind(content_type: str, first: str) -> str:
    """Формат тела: ``sse``, ``ndjson`` или ``json`` (цельный ответ) — по content-type и
    первой непустой строке."""
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


def _decode(data: str, kind: str) -> list[Any]:
    """Объекты из данных куска (``_DONE`` — конец потока).

    Строки ``data:`` без пустой строки между событиями (так шлют некоторые серверы) SSE
    склеивает в одно событие — тогда каждая строка разбирается отдельно. Не-JSON в целом
    ответе — ошибка, в потоке — пропускается.
    """
    text = data.strip()
    if text == "[DONE]":
        return [_DONE]
    try:
        return [json.loads(text, strict=False)]
    except json.JSONDecodeError as e:
        if kind == "json":
            raise ApiError(f"API вернул не-JSON ответ: {text[:ERROR_BODY_CHARS]}") from e
        if "\n" not in text:
            return []
        return [item for line in text.splitlines() for item in _decode(line, "ndjson")]


def consume_stream(
    lines: Iterable[str], emit: Callable[[str], None], *, content_type: str = ""
) -> AssistantTurn:
    """Разбирает тело ответа (поток или целый ответ), отдавая куски текста в ``emit``.

    Поток без окончания (``[DONE]``, причина завершения или расход токенов) — оборван:
    сервер закрыл соединение посреди ответа.
    """
    lines = iter(lines)
    head: list[str] = []
    for line in lines:  # до первой непустой строки: по ней виден формат
        head.append(line)
        if line.strip():
            break
    kind = _stream_kind(content_type, head[-1] if head else "")
    builder = ChunkBuilder()
    done = kind == "json"
    for event, data in _chunks(kind, itertools.chain(head, lines)):
        for chunk in _decode(data, kind):
            if chunk is _DONE:
                done = True
                break
            if not isinstance(chunk, Mapping):
                continue
            error = _error_text(chunk, event)
            if error is not None:
                raise ApiError(f"API вернул ошибку в потоке: {error}")
            if kind == "json" and not isinstance(chunk.get("choices"), list):
                raise ApiError(f"Неожиданный формат ответа API: {str(chunk)[:ERROR_BODY_CHARS]}")
            piece = builder.add(chunk)
            if piece:
                emit(piece)
        if done and kind != "json":
            break
    return builder.turn(done=done)
