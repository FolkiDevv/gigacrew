"""HTTP-провайдер модели (API v2).

Реализует авторизацию OAuth (Basic access-key -> Bearer access_token) с
автоматическим обновлением токена по истечении срока (или mTLS), и вызов
chat/completions с поддержкой function calling. Адреса API и выдачи токена
берутся только из конфигурации — значений по умолчанию нет.

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
from typing import Any

import httpx

from gigacrew.config import Config
from gigacrew.llm.base import LLMError, LLMProvider, PromptTooLong
from gigacrew.llm.types import AssistantTurn, FunctionCall, Message, ToolChoice, ToolSpec, Usage


class ApiError(LLMError):
    """Ошибка взаимодействия с API модели."""


# Транзиентные сетевые ошибки, которые имеет смысл повторять. Для потокового
# запроса повтор допустим, только пока пользователю не отдано ни одного куска.
# Все прочие httpx.HTTPError (ProxyError, LocalProtocolError, DecodingError...)
# не повторяются и превращаются в ApiError.
_RETRYABLE_EXC = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)

# HTTP-статусы, которые имеет смысл повторить (лимит/временная недоступность).
_RETRY_STATUS = {429, 500, 502, 503, 504}

# Замер окна: 4xx с этими статусами — не про размер запроса (авторизация, нет
# модели, лимит частоты); остальные 4xx и 5xx считаются «запрос не помещается».
_NOT_SIZE_STATUS = {401, 403, 404, 429}
# …кроме временной недоступности шлюза: счесть её отказом по размеру — занизить
# окно, и заниженное значение сохранилось бы в ~/.gigacrew/models.json.
_UNAVAILABLE_STATUS = {502, 503}
# Сколько тела ответа показывать в сообщении об ошибке (HTML-страница прокси и т.п.).
_ERROR_BODY_CHARS = 500


def _body(resp: httpx.Response) -> str:
    """Тело ответа для сообщения об ошибке — не длиннее :data:`_ERROR_BODY_CHARS`."""
    text = resp.text.strip()
    return text if len(text) <= _ERROR_BODY_CHARS else text[:_ERROR_BODY_CHARS] + "…"


_PROBE_ATTEMPTS = 2
# Обрыв или таймаут уже отправляемого запроса — для огромного запроса это тоже
# «не помещается»; ошибки установки соединения к размеру отношения не имеют.
_OVERSIZE_EXC = (
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)


_VERSION_SUFFIX = re.compile(r"/v\d+$")


def chat_url(base_url: str) -> str:
    """Адрес чата v2: хвост ``/vN`` базы заменяется на ``/v2`` (без хвоста — база как есть)."""
    return _VERSION_SUFFIX.sub("/v2", base_url.rstrip("/")) + "/chat/completions"


def _iter_sse_events(lines: Iterable[str]) -> Iterator[tuple[str | None, str]]:
    """События SSE: (имя из ``event:`` или None, склеенные строки ``data:``)."""
    event: str | None = None
    data: list[str] = []
    for line in lines:
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = None, []
            continue
        if line.startswith(":"):  # комментарий
            continue
        name, sep, value = line.partition(":")
        if not sep:
            continue
        value = value.removeprefix(" ")
        if name == "event":
            if data:  # событие без пустой строки-разделителя
                yield event, "\n".join(data)
                data = []
            event = value
        elif name == "data":
            data.append(value)
    if data:
        yield event, "\n".join(data)


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
    call_id = raw.get("id")
    return FunctionCall(
        name=raw["name"],
        arguments=args if isinstance(args, dict) else {},
        id=str(call_id) if call_id else None,
    )


class _TurnBuilder:
    """Собирает ход ассистента из ответа v2 — целого или по кускам потока.

    Рассуждения (``role: "reasoning"``) пропускаются. Вызов с уже встречавшимся
    ``id`` заменяет прежний (поток может прислать его повторно).
    """

    def __init__(self) -> None:
        self._text: list[str] = []
        self._calls: dict[str, FunctionCall] = {}
        self._state_id: str | None = None
        self._finish_reason: str | None = None
        self._usage: Any = None

    def add(self, data: Mapping[str, Any]) -> str:
        """Учесть ответ или кусок потока; возвращает пришедший в нём текст."""
        pieces: list[str] = []
        for msg in data.get("messages") or []:
            if not isinstance(msg, Mapping) or msg.get("role") != "assistant":
                continue
            state_id = msg.get("tools_state_id") or msg.get("tool_state_id")
            if state_id:
                self._state_id = str(state_id)
            content = msg.get("content")
            if isinstance(content, str):
                content = [{"text": content}]
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
        text = "".join(pieces)
        self._text.append(text)
        return text

    def turn(self) -> AssistantTurn:
        calls = list(self._calls.values())
        message = Message(
            role="assistant",
            content="".join(self._text),
            function_calls=calls,
            # Состояние нужно только результатам вызовов.
            state_id=self._state_id if calls else None,
        )
        finish = self._finish_reason or ("function_call" if calls else "stop")
        return AssistantTurn(
            message=message, finish_reason=finish, usage=Usage.from_raw(self._usage)
        )


def _stream_error(event: str | None, data: Mapping[str, Any]) -> str | None:
    """Текст ошибки, если событие потока — ошибка API (иначе None)."""
    if event and "error" in event.lower():
        return json.dumps(data, ensure_ascii=False)[:_ERROR_BODY_CHARS]
    if "messages" in data:
        return None
    status = data.get("status")
    if data.get("error") or (isinstance(status, int) and status >= 400):
        return json.dumps(data, ensure_ascii=False)[:_ERROR_BODY_CHARS]
    return None


class ApiProvider(LLMProvider):
    supports_measure = True

    def __init__(self, config: Config, *, transport: httpx.BaseTransport | None = None):
        """``transport`` — подмена HTTP-транспорта (``httpx.MockTransport`` в тестах)."""
        config.require_credentials()
        self._cfg = config
        # В режиме mTLS verify — это SSL-контекст с клиентским сертификатом;
        # в режиме OAuth — булев флаг проверки серверного сертификата.
        self._client = httpx.Client(
            verify=config.build_ssl_verify(),
            timeout=config.timeout,
            transport=transport,
        )
        self._mtls = config.auth_mode == "mtls"
        self._token: str | None = None
        self._token_exp: float = 0.0  # unix-время истечения токена
        self._max_retries = 5  # повторы при транзиентных сбоях (сеть/429/5xx)

    @property
    def model(self) -> str:
        return self._cfg.model

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ApiProvider:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Сетевой слой с повторами
    # ------------------------------------------------------------------ #
    def _request_with_retry(
        self, do_request: Callable[[], httpx.Response], *, attempts: int | None = None
    ) -> httpx.Response:
        """Выполняет HTTP-запрос, повторяя транзиентные сбои.

        Повторяем (с экспоненциальной задержкой):
          * сетевые ошибки/таймауты — сеть до API Сбера бывает нестабильна;
          * 429 Too Many Requests и 5xx — временная недоступность/лимит.
        Прочие HTTP-ответы (включая 4xx) возвращаем как есть — их разбирает
        вызывающая сторона. Неповторяемые ошибки транспорта → ApiError.
        ``attempts`` — число попыток вместо ``_max_retries``.
        """
        attempts = attempts or self._max_retries
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = do_request()
            except _RETRYABLE_EXC as e:
                last_exc = e
                if attempt < attempts - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 8.0)
                continue
            except httpx.HTTPError as e:
                raise ApiError(f"Ошибка соединения с API: {e}") from e

            if resp.status_code in _RETRY_STATUS and attempt < attempts - 1:
                time.sleep(self._retry_after(resp, delay))
                delay = min(delay * 2, 8.0)
                continue
            return resp

        raise ApiError(
            f"Сетевая ошибка при обращении к API (после {attempts} попыток): {last_exc}"
        ) from last_exc

    @staticmethod
    def _json(resp: httpx.Response) -> dict[str, Any]:
        """Тело ответа как JSON-объект; иначе ApiError."""
        try:
            data = resp.json()
        except ValueError as e:
            raise ApiError(f"API вернул не-JSON ответ ({resp.status_code}): {_body(resp)}") from e
        if not isinstance(data, dict):
            raise ApiError(f"Неожиданный формат ответа API: {str(data)[:500]}")
        return data

    @staticmethod
    def _retry_after(resp: httpx.Response, default: float) -> float:
        """Задержка перед повтором: уважает заголовок Retry-After, если он есть."""
        ra = resp.headers.get("Retry-After")
        if ra:
            try:
                return max(min(float(ra), 30.0), 0.0)  # отрицательное — ValueError в sleep
            except ValueError:
                pass
        return default

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
                f"Не удалось получить токен API ({resp.status_code}): {_body(resp)}"
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

    # ------------------------------------------------------------------ #
    # Сериализация / разбор
    # ------------------------------------------------------------------ #
    @staticmethod
    def _result_object(content: str) -> dict[str, Any]:
        """API требует, чтобы ``result`` части ``function_result`` был JSON-объектом.

        Объект передаётся как есть, всё остальное — текст, но и голые JSON-значения
        (``42``, ``true``, ``[]`` — например, листинг из одного файла ``1``) —
        заворачивается в ``{"result": ...}``: отвергнутое API сообщение осталось бы
        в истории и ломало бы каждый следующий запрос этого чата."""
        text = content or ""
        try:
            value = json.loads(text)
        except ValueError:
            value = None
        return value if isinstance(value, dict) else {"result": text}

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
    def _messages_payload(cls, messages: Sequence[Message]) -> list[dict[str, Any]]:
        """История для запроса. Результаты вызовов одного assistant-сообщения (идут
        подряд) склеиваются в одно сообщение ``role: "tool"`` с его ``tools_state_id``."""
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

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> AssistantTurn:
        if not isinstance(data.get("messages"), list):
            raise ApiError(f"Неожиданный формат ответа API: {str(data)[:_ERROR_BODY_CHARS]}")
        builder = _TurnBuilder()
        builder.add(data)
        return builder.turn()

    # ------------------------------------------------------------------ #
    # Основной вызов
    # ------------------------------------------------------------------ #
    def _payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None,
        *,
        model: str | None,
        temperature: float,
        stream: bool = False,
        tool_choice: ToolChoice = "auto",
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": temperature}
        payload: dict[str, Any] = {
            "model": model or self._cfg.model,
            "messages": self._messages_payload(messages),
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

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        tool_choice: ToolChoice = "auto",
    ) -> AssistantTurn:
        payload = self._payload(
            messages, tools, model=model, temperature=temperature, tool_choice=tool_choice
        )
        resp = self._post_chat(payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ApiError(f"API вернул {resp.status_code}: {_body(resp)}") from e

        return self._parse_response(self._json(resp))

    def _post_chat(
        self,
        payload: dict[str, Any],
        send: Callable[[Callable[[], httpx.Response]], httpx.Response] | None = None,
    ) -> httpx.Response:
        """POST на chat/completions; при протухшем токене (401) — обновить и повторить раз.

        ``send`` — как отправлять сам запрос (по умолчанию ``_request_with_retry``);
        получение токена идёт мимо него, его ошибки — обычные ``ApiError``.
        """
        url = chat_url(self._cfg.base_url)
        headers = {"Content-Type": "application/json", **self._auth_headers()}
        send = send or self._request_with_retry

        resp = send(lambda: self._client.post(url, json=payload, headers=headers))
        if resp.status_code == 401 and not self._mtls:
            # токен протух — обновляем принудительно и повторяем один раз
            headers.update(self._auth_headers(force=True))
            resp = send(lambda: self._client.post(url, json=payload, headers=headers))
        return resp

    # ------------------------------------------------------------------ #
    # Замер окна контекста
    # ------------------------------------------------------------------ #
    def _send_probe(self, do_request: Callable[[], httpx.Response]) -> httpx.Response:
        """Отправка пробы. Огромный запрос может упереться в таймаут шлюза: долгие
        повторы бессмысленны — обрыв уже отправляемого запроса и есть «не помещается»."""
        try:
            return self._request_with_retry(do_request, attempts=_PROBE_ATTEMPTS)
        except ApiError as e:
            if isinstance(e.__cause__, _OVERSIZE_EXC):
                raise PromptTooLong(None, str(e)) from e
            raise

    def measure_prompt(self, text: str, *, model: str | None = None) -> int:
        payload = {
            "model": model or self._cfg.model,
            "messages": [{"role": "user", "content": [{"text": text}]}],
            "model_options": {"max_tokens": 1, "temperature": 0},
        }
        resp = self._post_chat(payload, self._send_probe)
        status = resp.status_code
        if status in _UNAVAILABLE_STATUS:
            raise ApiError(f"API временно недоступен ({status}): {_body(resp)}")
        if status >= 500 or (status >= 400 and status not in _NOT_SIZE_STATUS):
            raise PromptTooLong(status, resp.text)
        if status >= 400:
            raise ApiError(f"API вернул {status}: {_body(resp)}")
        usage = Usage.from_raw(self._json(resp).get("usage"))
        if usage.prompt_tokens <= 0:
            raise ApiError("API не вернул usage.prompt_tokens — замер невозможен")
        return usage.prompt_tokens

    # ------------------------------------------------------------------ #
    # Потоковый вызов (SSE)
    # ------------------------------------------------------------------ #
    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        tool_choice: ToolChoice = "auto",
        on_delta: Callable[[str], None] | None = None,
    ) -> AssistantTurn:
        """Потоковая генерация. Вызывает on_delta(text) на каждый кусок текста и
        возвращает финальный AssistantTurn (идентичный complete()).

        Формат SSE: события ``response.message.delta`` — куски текста в тех же
        ``messages``, что и у обычного ответа; ``response.message.done`` — вызовы
        функций, ``tool_state_id``, ``finish_reason`` и ``usage``.

        Повтор при сетевом сбое выполняется, только пока пользователю не отдан
        ни один кусок текста — иначе он увидел бы ответ дважды.
        """
        sink = on_delta or (lambda _s: None)
        state = {"emitted": False}

        def emit(piece: str) -> None:
            state["emitted"] = True
            sink(piece)

        payload = self._payload(
            messages,
            tools,
            model=model,
            temperature=temperature,
            stream=True,
            tool_choice=tool_choice,
        )
        url = chat_url(self._cfg.base_url)
        delay = 1.0
        last_exc: Exception | None = None
        refreshed = False

        for attempt in range(self._max_retries):
            headers = {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                **self._auth_headers(),
            }
            try:
                with self._client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code == 401 and not self._mtls and not refreshed:
                        resp.read()
                        self._ensure_token(force=True)
                        refreshed = True
                        continue
                    if resp.status_code in _RETRY_STATUS and attempt < self._max_retries - 1:
                        resp.read()
                        time.sleep(self._retry_after(resp, delay))
                        delay = min(delay * 2, 8.0)
                        continue
                    if resp.status_code >= 400:
                        resp.read()
                        raise ApiError(f"API вернул {resp.status_code}: {_body(resp)}")
                    return self._consume_sse(resp, emit)
            except _RETRYABLE_EXC as e:
                last_exc = e
                if state["emitted"]:
                    # часть текста уже отдана пользователю — повтор приведёт к дублю
                    raise ApiError(f"Сетевой сбой во время потоковой передачи: {e}") from e
                if attempt < self._max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 8.0)
            except httpx.HTTPError as e:
                raise ApiError(f"Ошибка соединения с API: {e}") from e

        raise ApiError(
            f"Сетевая ошибка при потоковом обращении к API (после "
            f"{self._max_retries} попыток): {last_exc}"
        ) from last_exc

    @staticmethod
    def _consume_sse(resp: httpx.Response, emit: Callable[[str], None]) -> AssistantTurn:
        """Разбирает SSE-поток, отдавая текстовые дельты в ``emit``."""
        builder = _TurnBuilder()
        for event, data in _iter_sse_events(resp.iter_lines()):
            if data.strip() == "[DONE]":
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
            piece = builder.add(chunk)
            if piece:
                emit(piece)
        return builder.turn()

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
                f"Не удалось получить список моделей ({resp.status_code}): {_body(resp)}"
            )
        models = self._json(resp).get("data") or []
        return [
            m["id"] for m in models if isinstance(m, dict) and "id" in m and m.get("type") == "chat"
        ]
