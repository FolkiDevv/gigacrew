"""HTTP-провайдер модели.

Реализует авторизацию OAuth (Basic access-key -> Bearer access_token) с
автоматическим обновлением токена по истечении срока (или mTLS), и вызов
chat/completions с поддержкой function calling. Адреса API и выдачи токена
берутся только из конфигурации — значений по умолчанию нет.

Формат function calling:
  * запрос содержит массив ``functions`` (а не ``tools``) и
    ``function_call: "auto"``;
  * ответ при вызове инструмента: ``message.function_call`` с полями
    ``name`` и ``arguments`` (arguments — уже распарсенный объект, НЕ строка),
    ``finish_reason == "function_call"`` и непрозрачный ``functions_state_id``;
  * результат инструмента возвращается сообщением ``role: "function"`` с
    полями ``name`` и ``content``;
  * ``functions_state_id`` в историю запроса НЕ возвращается: с ним API сверяет
    вызовы с результатами по своему состоянию и на реальных запросах агента
    отвечает 422 «Function calls count does not match function results count»,
    хотя вызов и результат в истории парные; без него та же история принимается.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from gigacrew.config import Config
from gigacrew.llm.base import LLMError, LLMProvider, PromptTooLong
from gigacrew.llm.types import AssistantTurn, FunctionCall, Message, ToolSpec, Usage


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
    def _as_json_content(content: str) -> str:
        """API требует, чтобы content сообщения role=function был JSON-объектом.

        Объект передаётся как есть, всё остальное — текст, но и голые JSON-значения
        (``42``, ``true``, ``[]`` — например, листинг из одного файла ``1``) —
        заворачивается в ``{"result": ...}``: отвергнутое API сообщение осталось бы
        в истории и ломало бы каждый следующий запрос этого чата."""
        text = content or ""
        try:
            if isinstance(json.loads(text), dict):
                return text
        except ValueError:
            pass
        return json.dumps({"result": text}, ensure_ascii=False)

    @classmethod
    def _message_to_payload(cls, msg: Message) -> dict[str, Any]:
        """Сообщение истории для запроса. ``functions_state_id`` не отправляется —
        см. описание модуля."""
        content = msg.content or ""
        if msg.role == "function":
            content = cls._as_json_content(content)
        out: dict[str, Any] = {"role": msg.role, "content": content}
        if msg.name is not None:
            out["name"] = msg.name
        if msg.function_call is not None:
            out["function_call"] = {
                "name": msg.function_call.name,
                "arguments": msg.function_call.arguments,
            }
        return out

    @staticmethod
    def _tool_to_payload(tool: ToolSpec) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }

    @staticmethod
    def _build_function_call(fc_raw: dict[str, Any] | None) -> FunctionCall | None:
        if not fc_raw:
            return None
        args = fc_raw.get("arguments", {})
        # Обычно arguments — уже объект; подстраховка на случай строки.
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        return FunctionCall(name=fc_raw["name"], arguments=args or {})

    def _parse_response(self, data: dict[str, Any]) -> AssistantTurn:
        try:
            choice = data["choices"][0]
            raw = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise ApiError(f"Неожиданный формат ответа API: {data}") from e

        function_call = self._build_function_call(raw.get("function_call"))

        message = Message(
            role="assistant",
            content=raw.get("content") or "",
            function_call=function_call,
            functions_state_id=raw.get("functions_state_id"),
        )
        return AssistantTurn(
            message=message,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=Usage.from_raw(data.get("usage")),
        )

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
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self._cfg.model,
            "messages": [self._message_to_payload(m) for m in messages],
            "temperature": temperature,
        }
        if stream:
            payload["stream"] = True
        if tools:
            payload["functions"] = [self._tool_to_payload(t) for t in tools]
            payload["function_call"] = "auto"
        return payload

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> AssistantTurn:
        payload = self._payload(messages, tools, model=model, temperature=temperature)
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
        url = f"{self._cfg.base_url}/chat/completions"
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
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1,
            "temperature": 0,
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
        on_delta: Callable[[str], None] | None = None,
    ) -> AssistantTurn:
        """Потоковая генерация. Вызывает on_delta(text) на каждый кусок текста и
        возвращает финальный AssistantTurn (идентичный complete()).

        Формат SSE API: строки ``data: {json}`` с ``choices[0].delta`` и
        терминатор ``data: [DONE]``. Текст приходит в ``delta.content``; вызов
        функции — целиком в одном чанке (``delta.function_call`` + finish_reason).

        Повтор при сетевом сбое выполняется, только пока пользователю не отдан
        ни один кусок текста — иначе он увидел бы ответ дважды.
        """
        sink = on_delta or (lambda _s: None)
        state = {"emitted": False}

        def emit(piece: str) -> None:
            state["emitted"] = True
            sink(piece)

        payload = self._payload(messages, tools, model=model, temperature=temperature, stream=True)
        url = f"{self._cfg.base_url}/chat/completions"
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

    def _consume_sse(self, resp: httpx.Response, emit: Callable[[str], None]) -> AssistantTurn:
        """Разбирает SSE-поток, отдавая текстовые дельты в ``emit``."""
        content_parts: list[str] = []
        function_call = None
        state_id: str | None = None
        finish_reason = "stop"
        usage: dict[str, Any] = {}

        for raw_line in resp.iter_lines():
            if not raw_line or not raw_line.startswith("data:"):
                continue
            data = raw_line[len("data:") :].lstrip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            choices = chunk.get("choices") or []
            if choices:
                choice = choices[0]
                delta = choice.get("delta", {}) or {}
                piece = delta.get("content")
                if piece:
                    content_parts.append(piece)
                    emit(piece)
                if delta.get("function_call") and function_call is None:
                    function_call = self._build_function_call(delta["function_call"])
                if delta.get("functions_state_id"):
                    state_id = delta["functions_state_id"]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
            if chunk.get("usage"):
                usage = chunk["usage"]

        message = Message(
            role="assistant",
            content="".join(content_parts),
            function_call=function_call,
            functions_state_id=state_id,
        )
        return AssistantTurn(
            message=message, finish_reason=finish_reason, usage=Usage.from_raw(usage)
        )

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
