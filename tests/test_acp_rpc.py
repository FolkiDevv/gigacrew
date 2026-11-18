"""Транспорт JSON-RPC 2.0 для ACP: строка JSON на сообщение поверх двоичных потоков."""

from __future__ import annotations

import functools
import io
import json
import queue
import threading
from collections.abc import Buffer, Callable, Iterator, Mapping
from contextlib import closing
from typing import Any, override

import pytest
from acp_client import TIMEOUT, AcpClient
from pydantic import BaseModel

from gigacrew.cli.acp.rpc import (
    INVALID_PARAMS,
    ConnectionClosedError,
    RequestCancelledError,
    RpcConnection,
    RpcError,
)

# Исключение в любом потоке (обработчик, запись ответа) роняет тест, а не теряется.
pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")

type Handlers = Mapping[str, Callable[[Any], Any]]

# Ответ на «ping» — признак того, что до него сервер ничего не написал и цикл жив.
PING_ID = 999


def _echo(params: Any) -> Any:
    return params


def _pong(params: Any) -> str:
    return "pong"


class _Runner:
    """Запускает ``RpcConnection.serve`` в фоновом потоке с обработчиками теста."""

    def __init__(self, connection: RpcConnection) -> None:
        self.connection = connection
        self.threads: list[threading.Thread] = []

    def start(
        self,
        requests: Handlers | None = None,
        notifications: Handlers | None = None,
    ) -> threading.Thread:
        handlers = {"echo": _echo, "ping": _pong, **(requests or {})}
        serve = functools.partial(
            self.connection.serve,
            requests=handlers,
            notifications=notifications or {},
        )
        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        self.threads.append(worker)
        return worker


type _Failure = RpcError | ConnectionClosedError | RequestCancelledError


class _Background[Returned]:
    """Вызов в отдельном потоке: ``wait`` возвращает его итог или бросает его ошибку RPC."""

    def __init__(self, target: Callable[[], Returned]) -> None:
        self._target = target
        self._outcome: queue.Queue[Returned | _Failure] = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def wait(self) -> Returned:
        outcome = self._outcome.get(timeout=TIMEOUT)
        if isinstance(outcome, (RpcError, ConnectionClosedError, RequestCancelledError)):
            raise outcome
        return outcome

    def _run(self) -> None:
        try:
            outcome = self._target()
        except (RpcError, ConnectionClosedError, RequestCancelledError) as exc:
            self._outcome.put(exc)
            return
        self._outcome.put(outcome)


class _BrokenPipeWriter(io.BytesIO):
    """Выход, читатель которого ушёл: любая запись — BrokenPipeError."""

    @override
    def write(self, buffer: Buffer, /) -> int:
        raise BrokenPipeError(32, "Broken pipe")


def _closed_writer() -> io.BytesIO:
    stream = io.BytesIO()
    stream.close()
    return stream


@pytest.fixture
def client() -> Iterator[AcpClient]:
    with closing(AcpClient()) as pipe:
        yield pipe


@pytest.fixture
def connection(client: AcpClient) -> RpcConnection:
    return RpcConnection(client.server_input, client.server_output)


@pytest.fixture
def runner(client: AcpClient, connection: RpcConnection) -> Iterator[_Runner]:
    serving = _Runner(connection)
    yield serving
    client.close_input()
    for worker in serving.threads:
        worker.join(TIMEOUT)
        assert not worker.is_alive(), "serve не завершился после конца ввода"


def _assert_quiet_and_alive(client: AcpClient) -> None:
    """Сервер ничего не написал до сих пор и продолжает читать вход."""
    client.request(PING_ID, "ping")
    assert client.receive() == {"jsonrpc": "2.0", "id": PING_ID, "result": "pong"}


# ------------------------- входящие запросы ------------------------- #
@pytest.mark.parametrize(
    ("msg_id", "params", "expected"),
    [
        (1, {"text": "привет", "items": [1, 2]}, {"text": "привет", "items": [1, 2]}),
        ("abc", None, None),
    ],
)
def test_request_gets_result_with_same_id(
    client: AcpClient,
    runner: _Runner,
    msg_id: int | str,
    params: Any,
    expected: Any,
) -> None:
    runner.start()
    client.request(msg_id, "echo", params)
    assert client.receive() == {"jsonrpc": "2.0", "id": msg_id, "result": expected}


@pytest.mark.parametrize(
    ("line", "code", "msg_id"),
    [
        (b"{not json\n", -32700, None),
        (b"\xff\xfe{}\n", -32700, None),
        (b"[1, 2]\n", -32600, None),
        (b'"text"\n', -32600, None),
        (b'{"id": 4, "method": "echo"}\n', -32600, 4),
        (b'{"jsonrpc": "1.0", "id": 5, "method": "echo"}\n', -32600, 5),
        (b'{"jsonrpc": "2.0", "id": 6, "method": 7}\n', -32600, 6),
        (b'{"jsonrpc": "2.0", "id": {"x": 1}, "method": "echo"}\n', -32600, None),
        (b'{"jsonrpc": "2.0", "id": true, "method": "echo"}\n', -32600, None),
        (b'{"jsonrpc": "2.0", "id": 8, "method": "echo", "params": 5}\n', -32600, 8),
        (b'{"jsonrpc": "2.0", "id": 7, "method": "nope"}\n', -32601, 7),
    ],
    ids=[
        "parse-error",
        "bad-utf8",
        "batch",
        "not-object",
        "no-jsonrpc",
        "wrong-jsonrpc",
        "method-not-string",
        "bad-id",
        "bool-id",
        "scalar-params",
        "unknown-method",
    ],
)
def test_bad_message_gets_error_and_loop_continues(
    client: AcpClient,
    runner: _Runner,
    line: bytes,
    code: int,
    msg_id: int | None,
) -> None:
    runner.start()
    client.send_line(line)
    reply = client.receive()
    assert reply["id"] == msg_id
    assert "result" not in reply
    assert reply["error"]["code"] == code
    assert isinstance(reply["error"]["message"], str)
    assert reply["error"]["message"]
    _assert_quiet_and_alive(client)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RpcError(INVALID_PARAMS, "нет sessionId"), {"code": -32602, "message": "нет sessionId"}),
        (
            RpcError(-32000, "сессия занята", {"sessionId": "s1"}),
            {"code": -32000, "message": "сессия занята", "data": {"sessionId": "s1"}},
        ),
    ],
)
def test_handler_rpc_error_is_sent_as_is(
    client: AcpClient,
    runner: _Runner,
    error: RpcError,
    expected: dict[str, Any],
) -> None:
    def fail(params: Any) -> None:
        raise error

    runner.start({"fail": fail})
    client.request(1, "fail")
    assert client.receive() == {"jsonrpc": "2.0", "id": 1, "error": expected}


def test_validation_error_in_handler_is_invalid_params(
    client: AcpClient,
    runner: _Runner,
) -> None:
    class Point(BaseModel):
        column: int

    runner.start({"point": lambda params: Point.model_validate(params).column})
    client.request(1, "point", {"column": "не число"})
    reply = client.receive()
    assert reply["id"] == 1
    assert reply["error"]["code"] == -32602
    client.request(2, "point", {"column": 3})
    assert client.receive() == {"jsonrpc": "2.0", "id": 2, "result": 3}


def test_unexpected_exception_is_internal_error_without_traceback(
    client: AcpClient,
    runner: _Runner,
) -> None:
    def crash(params: Any) -> None:
        raise RuntimeError("сбой обработчика")

    runner.start({"crash": crash})
    client.request(1, "crash")
    reply = client.receive()
    assert reply["id"] == 1
    assert reply["error"]["code"] == -32603
    assert "Traceback" not in json.dumps(reply)
    assert 'File "' not in json.dumps(reply)
    _assert_quiet_and_alive(client)


# ------------------ ввод, на который ответа нет ------------------ #
def _raise_runtime(params: Any) -> None:
    raise RuntimeError("сбой уведомления")


def _raise_rpc(params: Any) -> None:
    raise RpcError(INVALID_PARAMS, "плохие параметры")


@pytest.mark.parametrize(
    "line",
    [
        b"\n",
        b"   \r\n",
        b'{"jsonrpc": "2.0", "id": 12345, "result": {"ok": true}}\n',
        b'{"jsonrpc": "2.0", "id": "x", "error": {"code": -32000, "message": "nope"}}\n',
        b'{"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": "Parse error"}}\n',
        b'{"jsonrpc": "2.0", "method": "unknown/notification", "params": {}}\n',
        b'{"jsonrpc": "2.0", "method": "crash"}\n',
        b'{"jsonrpc": "2.0", "method": "reject", "params": {}}\n',
    ],
    ids=[
        "empty-line",
        "blank-line",
        "result-unknown-id",
        "error-unknown-id",
        "error-null-id",
        "unknown-notification",
        "notification-crash",
        "notification-rpc-error",
    ],
)
def test_input_without_reply(client: AcpClient, runner: _Runner, line: bytes) -> None:
    runner.start(notifications={"crash": _raise_runtime, "reject": _raise_rpc})
    client.send_line(line)
    _assert_quiet_and_alive(client)


def test_notification_handled_in_reader_thread_before_next_message(
    client: AcpClient,
    runner: _Runner,
) -> None:
    seen: list[tuple[Any, threading.Thread]] = []

    def remember(params: Any) -> None:
        seen.append((params, threading.current_thread()))

    reader = runner.start(
        requests={"recall": lambda params: [noted for noted, _ in seen]},
        notifications={"remember": remember},
    )
    client.notify("remember", {"n": 1})
    client.request(1, "recall")
    assert client.receive() == {"jsonrpc": "2.0", "id": 1, "result": [{"n": 1}]}
    assert len(seen) == 1
    assert seen[0][1] is reader


def test_slow_request_does_not_block_reading(client: AcpClient, runner: _Runner) -> None:
    release = threading.Event()
    handled_in: list[threading.Thread] = []

    def block(params: Any) -> str:
        handled_in.append(threading.current_thread())
        assert release.wait(TIMEOUT)
        return "done"

    reader = runner.start(
        requests={"block": block},
        notifications={"release": lambda params: release.set()},
    )
    client.request(1, "block")
    client.request(2, "echo", ["fast"])
    assert client.receive() == {"jsonrpc": "2.0", "id": 2, "result": ["fast"]}
    client.notify("release")
    assert client.receive() == {"jsonrpc": "2.0", "id": 1, "result": "done"}
    assert handled_in[0] is not reader


def test_notifications_from_handler_precede_its_reply(
    client: AcpClient,
    runner: _Runner,
) -> None:
    def stream(params: Any) -> str:
        for index in range(5):
            runner.connection.notify("session/update", {"index": index})
        return "end"

    runner.start({"stream": stream})
    client.request(1, "stream")
    assert client.response(1) == {"jsonrpc": "2.0", "id": 1, "result": "end"}
    assert client.take_notifications("session/update") == [{"index": index} for index in range(5)]
    assert not client.backlog


# ------------------------- исходящие сообщения ------------------------- #
@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (
            {"sessionId": "s1", "update": {"text": "привет"}},
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"sessionId": "s1", "update": {"text": "привет"}},
            },
        ),
        (None, {"jsonrpc": "2.0", "method": "session/update"}),
    ],
)
def test_outgoing_notification(
    client: AcpClient,
    connection: RpcConnection,
    params: Any,
    expected: dict[str, Any],
) -> None:
    connection.notify("session/update", params)
    assert client.receive() == expected


def test_outgoing_request_from_handler_returns_client_result(
    client: AcpClient,
    runner: _Runner,
) -> None:
    def ask(params: Any) -> Any:
        return runner.connection.request("client/question", {"q": params["q"]})

    runner.start({"ask": ask})
    client.request(1, "ask", {"q": "можно?"})
    outgoing = client.receive()
    assert set(outgoing) == {"jsonrpc", "id", "method", "params"}
    assert outgoing["method"] == "client/question"
    assert outgoing["params"] == {"q": "можно?"}
    client.respond(outgoing["id"], {"answer": "да"})
    assert client.receive() == {"jsonrpc": "2.0", "id": 1, "result": {"answer": "да"}}


def test_outgoing_request_error_raises_rpc_error(
    client: AcpClient,
    runner: _Runner,
) -> None:
    runner.start()
    pending = _Background(lambda: runner.connection.request("client/question"))
    outgoing = client.receive()
    assert "params" not in outgoing
    client.respond_error(outgoing["id"], -32000, "отказано", {"why": "нельзя"})
    with pytest.raises(RpcError) as caught:
        pending.wait()
    assert (caught.value.code, caught.value.message) == (-32000, "отказано")
    assert caught.value.data == {"why": "нельзя"}


def test_outgoing_responses_matched_by_id(client: AcpClient, runner: _Runner) -> None:
    runner.start()
    first = _Background(lambda: runner.connection.request("client/question", {"n": 1}))
    second = _Background(lambda: runner.connection.request("client/question", {"n": 2}))
    sent = [client.receive(), client.receive()]
    assert sent[0]["id"] != sent[1]["id"]
    for outgoing in reversed(sent):
        client.respond(outgoing["id"], {"echo": outgoing["params"]["n"]})
    assert first.wait() == {"echo": 1}
    assert second.wait() == {"echo": 2}


def test_concurrent_writes_keep_messages_whole(
    client: AcpClient,
    connection: RpcConnection,
) -> None:
    writers, per_writer = 8, 20
    # Большое сообщение: без блокировки его запись частями (несколько write) перемешается
    # с записями других потоков.
    payload = "ж" * 20_000
    start = threading.Barrier(writers)

    def write(writer: int) -> None:
        start.wait(TIMEOUT)
        for index in range(per_writer):
            connection.notify("bulk", {"writer": writer, "index": index, "text": payload})

    threads = [threading.Thread(target=write, args=(writer,)) for writer in range(writers)]
    for thread in threads:
        thread.start()
    received = [client.receive()["params"] for _ in range(writers * per_writer)]
    for thread in threads:
        thread.join(TIMEOUT)
        assert not thread.is_alive()
    assert all(params["text"] == payload for params in received)
    for writer in range(writers):
        indexes = [params["index"] for params in received if params["writer"] == writer]
        assert indexes == list(range(per_writer))


@pytest.mark.parametrize("writer", [_BrokenPipeWriter, _closed_writer])
def test_write_failure_is_connection_closed(
    client: AcpClient,
    writer: Callable[[], io.BytesIO],
) -> None:
    connection = RpcConnection(client.server_input, writer())
    with pytest.raises(ConnectionClosedError):
        connection.notify("session/update", {"n": 1})
    with pytest.raises(ConnectionClosedError):
        _Background(lambda: connection.request("client/question")).wait()


def test_reply_write_failure_does_not_stop_serving(client: AcpClient) -> None:
    serving = _Runner(RpcConnection(client.server_input, _BrokenPipeWriter()))
    handled_in: queue.Queue[threading.Thread] = queue.Queue()
    alive = threading.Event()

    def work(params: Any) -> str:
        handled_in.put(threading.current_thread())
        return "done"

    reader = serving.start(
        requests={"work": work},
        notifications={"alive": lambda params: alive.set()},
    )
    client.request(1, "work")
    worker = handled_in.get(timeout=TIMEOUT)
    worker.join(TIMEOUT)
    assert not worker.is_alive()
    client.request(2, "nope")
    client.notify("alive")
    assert alive.wait(TIMEOUT)
    client.close_input()
    reader.join(TIMEOUT)
    assert not reader.is_alive()


# ---------------------- отмена исходящего запроса ---------------------- #
def test_cancel_stops_waiting_and_late_reply_is_ignored(
    client: AcpClient,
    runner: _Runner,
) -> None:
    runner.start()
    cancel = threading.Event()
    pending = _Background(lambda: runner.connection.request("client/question", cancel=cancel))
    cancelled = client.receive()
    cancel.set()
    with pytest.raises(RequestCancelledError):
        pending.wait()
    client.respond(cancelled["id"], {"late": True})
    _assert_quiet_and_alive(client)
    following = _Background(lambda: runner.connection.request("client/question"))
    outgoing = client.receive()
    assert outgoing["id"] != cancelled["id"]
    client.respond(outgoing["id"], {"answer": 1})
    assert following.wait() == {"answer": 1}


def test_cancel_set_before_call_raises_without_sending(
    client: AcpClient,
    runner: _Runner,
) -> None:
    runner.start()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RequestCancelledError):
        _Background(lambda: runner.connection.request("client/question", cancel=cancel)).wait()
    _assert_quiet_and_alive(client)


@pytest.mark.parametrize("failure", [ConnectionClosedError, RequestCancelledError])
def test_transport_failures_are_not_rpc_errors(failure: type[Exception]) -> None:
    # Обработчик, ловящий ответ-ошибку клиента (RpcError), не должен принять за неё
    # обрыв связи или отмену.
    assert not issubclass(failure, RpcError)


# ----------------------------- конец ввода ----------------------------- #
def test_end_of_input_stops_loop_and_fails_pending_requests(
    client: AcpClient,
    runner: _Runner,
) -> None:
    failures: queue.Queue[BaseException] = queue.Queue()

    def ask(params: Any) -> str:
        try:
            runner.connection.request("client/question")
        except ConnectionClosedError as exc:
            failures.put(exc)
        return "closed"

    reader = runner.start({"ask": ask})
    client.request(1, "ask")
    assert client.receive()["method"] == "client/question"
    client.close_input()
    reader.join(TIMEOUT)
    assert not reader.is_alive()
    assert isinstance(failures.get(timeout=TIMEOUT), ConnectionClosedError)
    assert client.response(1)["result"] == "closed"
    with pytest.raises(ConnectionClosedError):
        _Background(lambda: runner.connection.request("client/question")).wait()


def test_end_of_input_does_not_wait_for_running_handlers(
    client: AcpClient,
    runner: _Runner,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def block(params: Any) -> str:
        started.set()
        assert release.wait(TIMEOUT)
        return "done"

    reader = runner.start({"block": block})
    client.request(1, "block")
    assert started.wait(TIMEOUT)
    client.close_input()
    reader.join(TIMEOUT)
    assert not reader.is_alive()
    release.set()
    assert client.response(1)["result"] == "done"
