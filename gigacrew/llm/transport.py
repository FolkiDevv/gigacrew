"""Общий HTTP-слой провайдеров модели: повторы, лимит запросов, поток и замер окна.

Провайдер конкретного API наследует :class:`HttpProvider` и задаёт через хуки адрес чата,
авторизацию, тело запроса и разбор ответа (целого и потока). Сетевая часть — одна на всех:
повторы транзиентных сбоев, ожидание при 429, повтор без рассуждений модели, если шлюз их
не принял, долгое ожидание ответа модели и замер окна контекста.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, Self

import httpx

from gigacrew.config import Config
from gigacrew.llm.base import LLMError, LLMProvider, PromptTooLong
from gigacrew.llm.types import AssistantTurn, Message, ToolChoice, ToolSpec, Usage


class ApiError(LLMError):
    """Ошибка взаимодействия с API модели."""


# Транзиентные сетевые ошибки, которые имеет смысл повторять. Для потокового
# запроса повтор допустим, только пока пользователю не отдано ни одного куска.
# Все прочие httpx.HTTPError (ProxyError, LocalProtocolError, DecodingError...)
# не повторяются и превращаются в ApiError.
RETRYABLE_EXC = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)

# HTTP-статусы, которые имеет смысл повторить (лимит/временная недоступность; 529 —
# перегрузка сервера модели).
RETRY_STATUS = {429, 500, 502, 503, 504, 529}
# Лимит частоты запросов (429) шлюз считает за минуту, а обычные повторы (5 попыток с
# паузами 1–8 с) укладываются в ~15 с. На 429 — свой бюджет ожидания на запрос
# (``GIGACREW_RATE_LIMIT_WAIT``) и паузы длиннее; Retry-After шлюза — в приоритете.
RATE_LIMIT_FIRST_DELAY = 5.0
RATE_LIMIT_MAX_DELAY = 30.0
# Ответ шлюза на запрос, который он не принял по формату. Если в запросе были рассуждения
# модели, он повторяется без них; удалось — модель до конца сессии работает без возврата
# рассуждений: так их возврат не может сломать модель.
REJECTED_STATUS = {400, 422}
# Закрывающий тег рассуждений, который шлюз оставляет в начале текста ответа: сами
# рассуждения ушли отдельно, а тег — в ответ (MiniMax: ``</mm:think>``).
LEAKED_REASONING_TAGS = ("</mm:think>", "</think>")

# Замер окна: 4xx с этими статусами — не про размер запроса (авторизация, нет
# модели, лимит частоты); остальные 4xx и 5xx считаются «запрос не помещается».
NOT_SIZE_STATUS = {401, 403, 404, 429}
# …кроме временной недоступности шлюза: счесть её отказом по размеру — занизить
# окно, и заниженное значение сохранилось бы в ~/.gigacode/models.json.
UNAVAILABLE_STATUS = {502, 503, 529}
# Сколько тела ответа показывать в сообщении об ошибке (HTML-страница прокси и т.п.).
ERROR_BODY_CHARS = 500
# Попыток запроса чата, на который модель молчит дольше read_timeout: ожидание долгое,
# и обычные пять попыток растянулись бы почти на час.
SILENT_ATTEMPTS = 2

PROBE_ATTEMPTS = 2
# Обрыв или таймаут уже отправляемого запроса — для огромного запроса это тоже
# «не помещается»; ошибки установки соединения к размеру отношения не имеют.
OVERSIZE_EXC = (
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)


def error_body(resp: httpx.Response) -> str:
    """Тело ответа для сообщения об ошибке — не длиннее :data:`ERROR_BODY_CHARS`."""
    text = resp.text.strip()
    return text if len(text) <= ERROR_BODY_CHARS else text[:ERROR_BODY_CHARS] + "…"


def iter_sse_events(lines: Iterable[str]) -> Iterator[tuple[str | None, str]]:
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


def retry_after(resp: httpx.Response, default: float) -> float:
    """Задержка перед повтором: уважает заголовок Retry-After, если он есть."""
    ra = resp.headers.get("Retry-After")
    if ra:
        try:
            return max(min(float(ra), 30.0), 0.0)  # отрицательное — ValueError в sleep
        except ValueError:
            pass
    return default


class RateLimitWait:
    """Ожидание при 429 в пределах бюджета одного запроса (``budget`` секунд).

    Паузы растут от :data:`RATE_LIMIT_FIRST_DELAY` вдвое до :data:`RATE_LIMIT_MAX_DELAY`;
    Retry-After шлюза важнее, но не меньше секунды — нулевая пауза крутила бы повторы, не
    расходуя бюджет. О каждой паузе сообщает ``on_notice``.
    """

    def __init__(self, budget: float, on_notice: Callable[[str], None] | None):
        self._left = float(budget)
        self._delay = RATE_LIMIT_FIRST_DELAY
        self._on_notice = on_notice

    def wait(self, resp: httpx.Response) -> bool:
        """Подождать перед повтором; False — бюджет исчерпан, 429 — ошибка."""
        if self._left <= 0:
            return False
        pause = max(retry_after(resp, self._delay), 1.0)
        pause = min(pause, RATE_LIMIT_MAX_DELAY, self._left)
        if self._on_notice is not None:
            self._on_notice(f"лимит запросов шлюза (429) — жду {pause:.0f} с и повторяю")
        time.sleep(pause)
        self._left -= pause
        self._delay = min(self._delay * 2, RATE_LIMIT_MAX_DELAY)
        return True


class LeakedTagFilter:
    """Текст ответа без утёкшего закрывающего тега рассуждений в самом начале.

    Пока по началу ответа не ясно, тег ли это, текст копится и не отдаётся
    (:attr:`pending`); после убранного тега пропускаются пустые строки до самого ответа.
    """

    def __init__(self) -> None:
        # Начало ответа, пока неясно, не утёкший ли это тег (None — ясно).
        self._head: str | None = ""
        self._after_tag = False

    @property
    def pending(self) -> str:
        """Недособранное начало: ещё не отдано, но тегом может и не оказаться."""
        return self._head or ""

    def feed(self, text: str) -> str:
        """Очередной кусок текста; возвращает то, что можно показать."""
        if self._head is None:
            if self._after_tag:
                text = text.lstrip()
                self._after_tag = not text
            return text
        head = self._head + text
        stripped = head.lstrip()
        for tag in LEAKED_REASONING_TAGS:
            if stripped.startswith(tag):
                self._head = None
                rest = stripped[len(tag) :].lstrip()
                self._after_tag = not rest
                return rest
        if any(tag.startswith(stripped) for tag in LEAKED_REASONING_TAGS):
            self._head = head  # может оказаться тегом — ждём продолжения
            return ""
        self._head = None
        return head


class HttpProvider(LLMProvider):
    """Провайдер поверх HTTP: сетевая часть общая, формат API и авторизация — хуки.

    Хуки формата: :meth:`_chat_url`, :meth:`_payload`, :meth:`_has_reasoning`,
    :meth:`_parse_response`, :meth:`_consume_sse`, :meth:`_probe_payload`; авторизации —
    :meth:`_auth_headers` и :attr:`_refreshable` (401 на чате — обновить и повторить раз).
    """

    supports_measure = True

    def __init__(
        self,
        config: Config,
        *,
        transport: httpx.BaseTransport | None = None,
        headers: Mapping[str, str] | None = None,
    ):
        """``transport`` — подмена HTTP-транспорта (``httpx.MockTransport`` в тестах);
        ``headers`` — заголовки всех запросов клиента."""
        self._cfg = config
        # В режиме mTLS verify — это SSL-контекст с клиентским сертификатом;
        # иначе — булев флаг проверки серверного сертификата или контекст с CA.
        self._client = httpx.Client(
            verify=config.build_ssl_verify(),
            timeout=config.timeout,
            transport=transport,
            headers=headers,
        )
        # Запросы чата ждут ответа модели дольше (``read_timeout``): вызовы инструментов
        # приходят целиком в конце ответа, и пока модель пишет длинный аргумент (план,
        # файл), байты не идут. Авторизация, список моделей и замер окна — по обычному
        # timeout: долгая проба при замере и означает «не помещается».
        self._chat_timeout = httpx.Timeout(
            config.timeout, read=max(config.timeout, config.read_timeout)
        )
        self._max_retries = 5  # повторы при транзиентных сбоях (сеть/5xx)
        self._rate_limit_wait = config.rate_limit_wait  # ожидание при 429, с на запрос
        # Модели, для которых шлюз не принял рассуждения в запросе: дальше — без них.
        self._no_reasoning: set[str] = set()

    @property
    def model(self) -> str:
        return self._cfg.model

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Хуки формата и авторизации
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _chat_url(self) -> str:
        """Адрес chat/completions."""

    def _auth_headers(self, *, force: bool = False) -> dict[str, str]:
        """Заголовки авторизации; ``force`` — обновить её (сервер ответил 401)."""
        return {}

    @property
    def _refreshable(self) -> bool:
        """Можно ли на 401 обновить авторизацию и повторить запрос (один раз)."""
        return False

    def _chat_headers(self, *, stream: bool) -> dict[str, str]:
        """Заголовки запроса чата, кроме авторизации."""
        headers = {"Content-Type": "application/json"}
        if stream:
            headers["Accept"] = "text/event-stream"
        return headers

    @abstractmethod
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

    @abstractmethod
    def _has_reasoning(self, payload: Mapping[str, Any]) -> bool:
        """Есть ли в теле запроса рассуждения модели."""

    @abstractmethod
    def _parse_response(self, data: dict[str, Any]) -> AssistantTurn:
        """Ход ассистента из целого (не потокового) ответа."""

    @abstractmethod
    def _consume_sse(self, resp: httpx.Response, emit: Callable[[str], None]) -> AssistantTurn:
        """Ход ассистента из потока; куски текста — в ``emit``."""

    @abstractmethod
    def _probe_payload(self, text: str, model: str) -> dict[str, Any]:
        """Тело пробы для замера окна: ``text`` одним сообщением, ответ в 1 токен."""

    def _probe_usage(self, resp: httpx.Response) -> Usage:
        """Расход токенов из успешного ответа на пробу."""
        return Usage.from_raw(self._json(resp).get("usage"))

    # ------------------------------------------------------------------ #
    # Сетевой слой с повторами
    # ------------------------------------------------------------------ #
    def _request_with_retry(
        self,
        do_request: Callable[[], httpx.Response],
        *,
        attempts: int | None = None,
        silent_attempts: int | None = None,
        on_notice: Callable[[str], None] | None = None,
    ) -> httpx.Response:
        """Выполняет HTTP-запрос, повторяя транзиентные сбои.

        Повторяем (с экспоненциальной задержкой):
          * сетевые ошибки/таймауты — сеть до API бывает нестабильна;
          * 5xx — временная недоступность;
          * 429 Too Many Requests — в пределах своего бюджета ожидания (не в счёт
            попыток), о каждой паузе сообщает ``on_notice``.
        Прочие HTTP-ответы (включая 4xx) возвращаем как есть — их разбирает
        вызывающая сторона. Неповторяемые ошибки транспорта → ApiError.
        ``attempts`` — число попыток вместо ``_max_retries``; ``silent_attempts`` —
        сколько раз из них ответ может не прийти за время ожидания чтения.
        """
        attempts = attempts or self._max_retries
        delay = 1.0
        last_exc: Exception | None = None
        silences = 0
        limit = RateLimitWait(self._rate_limit_wait, on_notice)
        attempt = 0
        while attempt < attempts:
            try:
                resp = do_request()
            except RETRYABLE_EXC as e:
                last_exc = e
                if isinstance(e, httpx.ReadTimeout) and silent_attempts is not None:
                    silences += 1
                    if silences >= silent_attempts:
                        raise ApiError(self._silence_message()) from e
                attempt += 1
                if attempt < attempts:
                    time.sleep(delay)
                    delay = min(delay * 2, 8.0)
                continue
            except httpx.HTTPError as e:
                raise ApiError(f"Ошибка соединения с API: {e}") from e

            if resp.status_code == 429:
                if limit.wait(resp):
                    continue
                return resp
            if resp.status_code in RETRY_STATUS and attempt < attempts - 1:
                time.sleep(self._retry_after(resp, delay))
                delay = min(delay * 2, 8.0)
                attempt += 1
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
            raise ApiError(
                f"API вернул не-JSON ответ ({resp.status_code}): {error_body(resp)}"
            ) from e
        if not isinstance(data, dict):
            raise ApiError(f"Неожиданный формат ответа API: {str(data)[:500]}")
        return data

    _retry_after = staticmethod(retry_after)

    # ------------------------------------------------------------------ #
    # Основной вызов
    # ------------------------------------------------------------------ #
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        tool_choice: ToolChoice = "auto",
        on_notice: Callable[[str], None] | None = None,
    ) -> AssistantTurn:
        name = model or self.model
        payload = self._payload(
            messages,
            tools,
            model=model,
            temperature=temperature,
            tool_choice=tool_choice,
            reasoning=name not in self._no_reasoning,
        )

        def send(do_request: Callable[[], httpx.Response]) -> httpx.Response:
            return self._send_chat(do_request, on_notice=on_notice)

        resp = self._post_chat(payload, send, timeout=self._chat_timeout)
        if resp.status_code in REJECTED_STATUS and self._has_reasoning(payload):
            plain = self._payload(
                messages,
                tools,
                model=model,
                temperature=temperature,
                tool_choice=tool_choice,
                reasoning=False,
            )
            retry = self._post_chat(plain, send, timeout=self._chat_timeout)
            if retry.status_code < 400:
                self._reject_reasoning(name, resp.status_code, on_notice)
            resp = retry
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ApiError(f"API вернул {resp.status_code}: {error_body(resp)}") from e

        return self._parse_response(self._json(resp))

    def _post_chat(
        self,
        payload: dict[str, Any],
        send: Callable[[Callable[[], httpx.Response]], httpx.Response] | None = None,
        *,
        timeout: httpx.Timeout | None = None,
    ) -> httpx.Response:
        """POST на chat/completions; при протухшей авторизации (401) — обновить и повторить раз.

        ``send`` — как отправлять сам запрос (по умолчанию ``_request_with_retry``);
        обновление авторизации идёт мимо него, его ошибки — обычные ``ApiError``.
        ``timeout`` — таймауты запроса вместо клиентских (None — клиентские; не «без
        таймаута»).
        """
        url = self._chat_url()
        headers = {**self._chat_headers(stream=False), **self._auth_headers()}
        send = send or self._request_with_retry
        options = {} if timeout is None else {"timeout": timeout}

        resp = send(lambda: self._client.post(url, json=payload, headers=headers, **options))
        if resp.status_code == 401 and self._refreshable:
            # авторизация протухла — обновляем принудительно и повторяем один раз
            headers.update(self._auth_headers(force=True))
            resp = send(lambda: self._client.post(url, json=payload, headers=headers, **options))
        return resp

    def _reject_reasoning(
        self, model: str, status: int, on_notice: Callable[[str], None] | None
    ) -> None:
        """Шлюз не принял запрос с рассуждениями, а без них принял: дальше — без них."""
        self._no_reasoning.add(model)
        if on_notice is not None:
            on_notice(
                f"шлюз не принял рассуждения модели {model} в запросе ({status}) — дальше без них"
            )

    def _send_chat(
        self,
        do_request: Callable[[], httpx.Response],
        *,
        on_notice: Callable[[str], None] | None = None,
    ) -> httpx.Response:
        """Отправка запроса чата: молчание модели дольше ожидания чтения — не больше
        :data:`SILENT_ATTEMPTS` попыток."""
        return self._request_with_retry(
            do_request, silent_attempts=SILENT_ATTEMPTS, on_notice=on_notice
        )

    def _silence_message(self) -> str:
        seconds = int(self._chat_timeout.read or 0)
        return (
            f"Модель не отвечала дольше {seconds} с — ответ оборван. Если она писала длинный "
            "вызов (план, большой файл), увеличьте GIGACREW_READ_TIMEOUT."
        )

    # ------------------------------------------------------------------ #
    # Замер окна контекста
    # ------------------------------------------------------------------ #
    def _send_probe(self, do_request: Callable[[], httpx.Response]) -> httpx.Response:
        """Отправка пробы. Огромный запрос может упереться в таймаут шлюза: долгие
        повторы бессмысленны — обрыв уже отправляемого запроса и есть «не помещается»."""
        try:
            return self._request_with_retry(do_request, attempts=PROBE_ATTEMPTS)
        except ApiError as e:
            if isinstance(e.__cause__, OVERSIZE_EXC):
                raise PromptTooLong(None, str(e)) from e
            raise

    def measure_prompt(self, text: str, *, model: str | None = None) -> int:
        payload = self._probe_payload(text, model or self.model)
        resp = self._post_chat(payload, self._send_probe)
        status = resp.status_code
        if status in UNAVAILABLE_STATUS:
            raise ApiError(f"API временно недоступен ({status}): {error_body(resp)}")
        if status >= 500 or (status >= 400 and status not in NOT_SIZE_STATUS):
            raise PromptTooLong(status, resp.text)
        if status >= 400:
            raise ApiError(f"API вернул {status}: {error_body(resp)}")
        usage = self._probe_usage(resp)
        if usage.prompt_tokens <= 0:
            raise ApiError("API не вернул usage.prompt_tokens — замер невозможен")
        return usage.prompt_tokens

    # ------------------------------------------------------------------ #
    # Потоковый вызов
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
        on_notice: Callable[[str], None] | None = None,
    ) -> AssistantTurn:
        """Потоковая генерация. Вызывает on_delta(text) на каждый кусок текста и
        возвращает финальный AssistantTurn (идентичный complete()).

        Повтор при сетевом сбое выполняется, только пока пользователю не отдан
        ни один кусок текста — иначе он увидел бы ответ дважды. Авторизация после 401
        обновляется перед следующей попыткой, когда ответ уже закрыт.
        """
        sink = on_delta or (lambda _s: None)
        state = {"emitted": False}

        def emit(piece: str) -> None:
            state["emitted"] = True
            sink(piece)

        name = model or self.model
        payload = self._payload(
            messages,
            tools,
            model=model,
            temperature=temperature,
            stream=True,
            tool_choice=tool_choice,
            reasoning=name not in self._no_reasoning,
        )
        url = self._chat_url()
        delay = 1.0
        last_exc: Exception | None = None
        refresh = False  # обновить авторизацию перед этой попыткой (прошлая — 401)
        refreshed = False
        silences = 0  # попыток, на которые модель молчала дольше ожидания чтения
        limit = RateLimitWait(self._rate_limit_wait, on_notice)
        rejected: int | None = None  # статус отказа запросу с рассуждениями

        attempt = 0
        while attempt < self._max_retries:
            headers = {**self._chat_headers(stream=True), **self._auth_headers(force=refresh)}
            refresh = False
            try:
                with self._client.stream(
                    "POST", url, json=payload, headers=headers, timeout=self._chat_timeout
                ) as resp:
                    if resp.status_code == 401 and self._refreshable and not refreshed:
                        resp.read()
                        refresh = refreshed = True
                        attempt += 1
                        continue
                    if resp.status_code in REJECTED_STATUS and self._has_reasoning(payload):
                        resp.read()
                        rejected = resp.status_code
                        payload = self._payload(
                            messages,
                            tools,
                            model=model,
                            temperature=temperature,
                            stream=True,
                            tool_choice=tool_choice,
                            reasoning=False,
                        )
                        attempt += 1
                        continue
                    if resp.status_code == 429:
                        resp.read()
                        if limit.wait(resp):
                            continue
                    elif resp.status_code in RETRY_STATUS and attempt < self._max_retries - 1:
                        resp.read()
                        time.sleep(self._retry_after(resp, delay))
                        delay = min(delay * 2, 8.0)
                        attempt += 1
                        continue
                    if resp.status_code >= 400:
                        resp.read()
                        raise ApiError(f"API вернул {resp.status_code}: {error_body(resp)}")
                    if rejected is not None:
                        self._reject_reasoning(name, rejected, on_notice)
                    return self._consume_sse(resp, emit)
            except RETRYABLE_EXC as e:
                last_exc = e
                silent = isinstance(e, httpx.ReadTimeout)
                if silent:
                    silences += 1
                if state["emitted"]:
                    # часть текста уже отдана пользователю — повтор приведёт к дублю
                    if silent:
                        raise ApiError(self._silence_message()) from e
                    raise ApiError(f"Сетевой сбой во время потоковой передачи: {e}") from e
                if silences >= SILENT_ATTEMPTS:
                    raise ApiError(self._silence_message()) from e
                attempt += 1
                if attempt < self._max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 8.0)
            except httpx.HTTPError as e:
                raise ApiError(f"Ошибка соединения с API: {e}") from e

        raise ApiError(
            f"Сетевая ошибка при потоковом обращении к API (после "
            f"{self._max_retries} попыток): {last_exc}"
        ) from last_exc
