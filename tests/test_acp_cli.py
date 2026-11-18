"""Подкоманда ``gigacrew acp``: ACP-сервер на stdin/stdout настоящего процесса.

Так gigacrew запускает T3 Code: ``commandPath`` + ``commandArgs: ["acp"]``, у каждого инстанса
своё окружение (``GIGACREW_PROVIDER=api`` или ``gigacode``). stdout процесса — только
сообщения протокола: предупреждения, журнал и случайный вывод (``print``, дочерние процессы)
уходят в stderr. Сеть не нужна: адреса API — несуществующие, к ним никто не обращается.
"""

from __future__ import annotations

import itertools
import json
import os
import queue
import shlex
import signal
import socket
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import closing, suppress
from pathlib import Path
from types import MappingProxyType
from typing import IO, Any, Final

import pytest
from acp_harness import PROTOCOL_VERSION

from gigacrew.cli.acp.rpc import INTERNAL_ERROR
from gigacrew.project.settings import TrustStore, load_settings, project_fingerprint
from gigacrew.project.workspace import Workspace

# Исключение в потоке-читателе роняет тест, а не теряется.
pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")

# Запуск, как у T3: интерпретатор окружения gigacrew и подкоманда acp.
GIGACREW: Final = (sys.executable, "-m", "gigacrew")
ACP: Final = (*GIGACREW, "acp")
# Сколько ждать процесса, с: запуск интерпретатора с импортами gigacrew — не мгновенный.
PROCESS_TIMEOUT: Final = 30.0
# Срок хука, с: заведомо дольше теста — хук завершает только остановка gigacrew.
HOOK_TIMEOUT: Final = 600
# Код выхода argparse при ошибке в аргументах.
USAGE_ERROR: Final = 2

# Инстанс «GigaCrew · API v2»: реквизиты есть, адреса никуда не ведут.
API_INSTANCE: Final = MappingProxyType({
    "GIGACREW_PROVIDER": "api",
    "GIGACREW_ACCESS_KEY": "dummy",
    "GIGACREW_AUTH_URL": "https://auth.test/oauth",
    "GIGACREW_API_URL": "https://api.test/v1",
    "GIGACREW_MODEL": "M-default",
})
# Инстанс «GigaCrew · GigaCode» без токена официального CLI (GIGACODE_HOME — отдельно).
GIGACODE_INSTANCE: Final = MappingProxyType({
    "GIGACREW_PROVIDER": "gigacode",
    "GIGACREW_API_URL": "https://gigacode.test/api",
    "GIGACREW_USER_AGENT": "gigacrew-test",
})

# Посторонний вывод работающего сервера (до ответа на initialize): print, запись в
# sys.stdout, в дескриптор 1 и дочерний процесс, унаследовавший stdout. Всё это — не протокол.
STRAY_MARKERS: Final = ("STRAY-PRINT", "STRAY-WRITE", "STRAY-FD", "STRAY-CHILD")
STRAY_SCRIPT: Final = r"""
import os
import subprocess
import sys

from gigacrew.cli import app
from gigacrew.cli.acp.server import AcpServer

original = AcpServer.serve


def noisy(self, reader, writer):
    print("STRAY-PRINT")
    sys.stdout.write("STRAY-WRITE\n")
    sys.stdout.flush()
    os.write(1, b"STRAY-FD\n")
    subprocess.run([sys.executable, "-c", "print('STRAY-CHILD')"], check=True)
    return original(self, reader, writer)


AcpServer.serve = noisy
sys.exit(app.main(["acp"]))
"""

# Хук UserPromptSubmit: сообщает тесту свой pid через сокет и ждёт, пока его не убьют.
# Пока процесс хука жив, соединение открыто; умер — тест читает конец потока.
HOOK_SCRIPT: Final = """
import os
import socket
import sys

conn = socket.create_connection(("127.0.0.1", int(sys.argv[1])))
conn.sendall(str(os.getpid()).encode())
conn.recv(1)
"""

type Message = dict[str, Any]


class _AcpProcess:
    """``gigacrew acp`` в подпроцессе и клиент ACP к нему.

    Каждая строка stdout обязана быть сообщением JSON-RPC 2.0 (проверяет :meth:`finish`),
    stderr копится отдельно.
    """

    def __init__(self, command: Sequence[str], *, env: Mapping[str, str], cwd: Path) -> None:
        self.proc = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(env),
            cwd=cwd,
        )
        self.output: list[bytes] = []
        self._lines: queue.Queue[bytes | None] = queue.Queue()
        self._stderr: list[bytes] = []
        self._ids = itertools.count(1)
        self._backlog: list[Message] = []
        self._pumps = [
            threading.Thread(target=self._read_stdout, args=(_pipe(self.proc.stdout),)),
            threading.Thread(target=self._read_stderr, args=(_pipe(self.proc.stderr),)),
        ]
        for pump in self._pumps:
            pump.start()

    @property
    def stderr(self) -> str:
        """Всё, что процесс написал в stderr к этому моменту."""
        return b"".join(self._stderr).decode(errors="replace")

    def begin(self, method: str, params: Any) -> int:
        """Отправляет запрос, не дожидаясь ответа; id запроса."""
        msg_id = next(self._ids)
        message = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        stdin = _pipe(self.proc.stdin)
        stdin.write(f"{json.dumps(message, ensure_ascii=False)}\n".encode())
        stdin.flush()
        return msg_id

    def reply(self, msg_id: int) -> Message:
        """Ответ на запрос ``msg_id``; уведомления и запросы сервера по пути — в сторону."""
        while True:
            message = self._receive()
            if message.get("id") == msg_id and "method" not in message:
                return message
            self._backlog.append(message)

    def send(self, method: str, params: Any) -> Message:
        """Запрос и ответ сервера целиком (``result`` или ``error``)."""
        return self.reply(self.begin(method, params))

    def initialize(self) -> Any:
        """``initialize`` как у T3; ошибка JSON-RPC роняет тест."""
        response = self.send(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "clientCapabilities": {},
                "clientInfo": {"name": "t3-test", "version": "0.0.0"},
            },
        )
        assert "result" in response, f"initialize: {response}\nstderr:\n{self.stderr}"
        return response["result"]

    def new_session(self, cwd: Path) -> Message:
        """Ответ на ``session/new`` в каталоге ``cwd`` (результат или ошибка)."""
        return self.send("session/new", {"cwd": str(cwd), "mcpServers": []})

    def close_input(self) -> None:
        """Конец ввода: сервер обязан закрыть сессии и выйти."""
        stdin = _pipe(self.proc.stdin)
        if not stdin.closed:
            stdin.close()

    def finish(self) -> int:
        """Дожидается выхода процесса; код выхода. stdout — только сообщения протокола."""
        try:
            code = self.proc.wait(PROCESS_TIMEOUT)
        except subprocess.TimeoutExpired:
            pytest.fail(f"процесс не завершился за {PROCESS_TIMEOUT} с\nstderr:\n{self.stderr}")
        for pump in self._pumps:
            pump.join(PROCESS_TIMEOUT)
        for line in self.output:
            _assert_protocol(line, self.stderr)
        return code

    def close(self) -> None:
        """Убивает процесс, если он ещё работает, и закрывает каналы."""
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(PROCESS_TIMEOUT)
        for pump in self._pumps:
            pump.join(PROCESS_TIMEOUT)
        self.close_input()

    def _receive(self) -> Message:
        try:
            line = self._lines.get(timeout=PROCESS_TIMEOUT)
        except queue.Empty:
            pytest.fail(f"сервер ничего не прислал за {PROCESS_TIMEOUT} с\nstderr:\n{self.stderr}")
        if line is None:
            pytest.fail(f"сервер закрыл stdout раньше ответа\nstderr:\n{self.stderr}")
        return _assert_protocol(line, self.stderr)

    def _read_stdout(self, stdout: IO[bytes]) -> None:
        for line in iter(stdout.readline, b""):
            self.output.append(line)
            self._lines.put(line)
        self._lines.put(None)
        stdout.close()

    def _read_stderr(self, stderr: IO[bytes]) -> None:
        for chunk in iter(stderr.readline, b""):
            self._stderr.append(chunk)
        stderr.close()


def _pipe(stream: IO[bytes] | None) -> IO[bytes]:
    """Канал процесса, открытый с PIPE."""
    assert stream is not None
    return stream


def _assert_protocol(line: bytes, stderr: str) -> Message:
    """Строка stdout — одно сообщение JSON-RPC 2.0 целиком; иначе тест падает."""
    hint = f"посторонний вывод в stdout: {line!r}\nstderr:\n{stderr}"
    assert line.endswith(b"\n"), hint
    try:
        message = json.loads(line)
    except ValueError:
        pytest.fail(hint)
    assert isinstance(message, dict), hint
    assert message.get("jsonrpc") == "2.0", hint
    return message


def _instance_env(home: Path, settings: Mapping[str, str]) -> dict[str, str]:
    """Окружение инстанса: настройки разработчика (GIGACREW_*/GIGACODE_*) убраны, свой HOME."""
    env = {
        name: setting
        for name, setting in os.environ.items()
        if not name.startswith(("GIGACREW_", "GIGACODE_"))
    }
    env.update({"HOME": str(home), "USERPROFILE": str(home)})
    env.update(settings)
    return env


@pytest.fixture
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Пустой текущий каталог процесса сервера (не проект сессии)."""
    return tmp_path_factory.mktemp("process-cwd")


@pytest.fixture
def root(project: Path) -> Path:
    """Каталог проекта сессии (полный путь)."""
    return project.resolve()


@pytest.fixture
def launch(workdir: Path, user_home: Path) -> Iterator[_Launcher]:
    """``launch(settings)`` — запущенный ``gigacrew acp`` с окружением инстанса ``settings``."""
    launcher = _Launcher(workdir, user_home)
    with closing(launcher):
        yield launcher


class _Launcher:
    """Запускает процессы ``gigacrew acp`` и убивает недоработавшие в конце теста."""

    def __init__(self, workdir: Path, home: Path) -> None:
        self._workdir = workdir
        self._home = home
        self._started: list[_AcpProcess] = []

    def __call__(
        self,
        settings: Mapping[str, str],
        command: Sequence[str] = ACP,
    ) -> _AcpProcess:
        """Процесс ``command`` в окружении инстанса ``settings``."""
        process = _AcpProcess(
            command,
            env=_instance_env(self._home, settings),
            cwd=self._workdir,
        )
        self._started.append(process)
        return process

    def close(self) -> None:
        """Убивает все запущенные процессы."""
        for process in self._started:
            process.close()


def _run_cli(
    args: Sequence[str],
    user_home: Path,
    workdir: Path,
) -> subprocess.CompletedProcess[str]:
    """``gigacrew <args>`` с пустым вводом; вывод — текстом."""
    return subprocess.run(
        [*GIGACREW, *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=_instance_env(user_home, API_INSTANCE),
        cwd=workdir,
        timeout=PROCESS_TIMEOUT,
        check=False,
    )


def test_acp_answers_initialize_and_exits_on_end_of_input(
    launch: _Launcher,
    workdir: Path,
) -> None:
    """Ответ на initialize; конец stdin → выход с кодом 0; в stdout только этот ответ.

    В текущем каталоге процесса ничего не появляется.
    """
    server = launch(API_INSTANCE)

    answer = server.initialize()
    server.close_input()

    assert answer["protocolVersion"] == PROTOCOL_VERSION
    assert answer["agentInfo"]["name"] == "gigacrew"
    assert server.finish() == 0, server.stderr
    assert len(server.output) == 1  # ни баннера, ни приглашения — только ответ
    assert not any(workdir.iterdir())


def test_provider_follows_gigacrew_provider_of_the_instance(
    launch: _Launcher,
    root: Path,
    workdir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Два инстанса, как в T3: API v2 открывает сессию, GigaCode без токена — понятная ошибка.

    Провайдер сессии — по GIGACREW_PROVIDER процесса: API v2 к сети при открытии сессии не
    обращается, а бэкенд GigaCode сразу ищет токен официального CLI в GIGACODE_HOME.
    Сессия пишет только в свой проект и в ~/.gigacode, не в текущий каталог процесса.
    """
    cli_home = tmp_path_factory.mktemp("gigacode-cli")
    api = launch(API_INSTANCE)
    gigacode = launch({**GIGACODE_INSTANCE, "GIGACODE_HOME": str(cli_home)})
    api.initialize()
    gigacode.initialize()

    opened = api.new_session(root)
    refused = gigacode.new_session(root)

    assert "result" in opened, f"{opened}\nstderr:\n{api.stderr}"
    assert opened["result"]["sessionId"]
    assert refused["error"]["code"] == INTERNAL_ERROR
    message = refused["error"]["message"]
    assert "нет токена GigaCode" in message
    assert str(cli_home / "gigacode_token.json") in message
    assert "Traceback" not in message
    for server in (api, gigacode):
        server.close_input()
        assert server.finish() == 0, server.stderr
    assert not any(workdir.iterdir())


def test_session_warnings_go_to_stderr(launch: _Launcher, root: Path) -> None:
    """Предупреждения сборки сессии (TLS, пропущенный адрес API из .env проекта) — в stderr."""
    (root / ".env").write_text("GIGACREW_API_URL=https://elsewhere.test/v1\n", encoding="utf-8")
    server = launch({**API_INSTANCE, "GIGACREW_VERIFY_SSL": "0"})
    server.initialize()

    opened = server.new_session(root)
    server.close_input()

    assert "result" in opened, f"{opened}\nstderr:\n{server.stderr}"
    assert server.finish() == 0
    assert "проверка TLS-сертификата сервера выключена" in server.stderr
    assert "не действуют настройки безопасности: GIGACREW_API_URL" in server.stderr


def test_stray_output_does_not_reach_the_protocol(launch: _Launcher) -> None:
    """print, sys.stdout, дескриптор 1 и дочерний процесс во время работы — в stderr."""
    server = launch(API_INSTANCE, command=(sys.executable, "-c", STRAY_SCRIPT))

    answer = server.initialize()
    server.close_input()

    assert answer["agentInfo"]["name"] == "gigacrew"
    assert server.finish() == 0, server.stderr
    for marker in STRAY_MARKERS:
        assert marker in server.stderr


def test_acp_help(user_home: Path, workdir: Path) -> None:
    """``gigacrew acp --help`` — справка подкоманды; общая справка упоминает acp."""
    sub_help = _run_cli(["acp", "--help"], user_home, workdir)
    main_help = _run_cli(["--help"], user_home, workdir)

    assert sub_help.returncode == 0, sub_help.stderr
    assert "gigacrew acp" in sub_help.stdout
    assert main_help.returncode == 0, main_help.stderr
    assert "acp" in main_help.stdout


def test_acp_rejects_foreign_arguments(user_home: Path, workdir: Path) -> None:
    """Аргументы REPL у подкоманды (``-p``) — ошибка argparse (код 2), сервер не запускается.

    Строка ошибки называет именно ``-p``: аргумент отвергнут как лишний для acp, а не
    принят общим разбором (тогда лишним оказалось бы само ``acp``).
    """
    completed = _run_cli(["acp", "-p", "привет"], user_home, workdir)

    assert completed.returncode == USAGE_ERROR
    assert not completed.stdout
    error_line = completed.stderr.strip().splitlines()[-1]
    assert "-p" in error_line, completed.stderr


@pytest.mark.parametrize(
    "run_options",
    [["--no-save", "--untrusted-project"], ["-y"], ["--mode", "plan"]],
    ids=["no-save-untrusted", "yes", "mode"],
)
def test_acp_rejects_run_options_before_subcommand(
    user_home: Path,
    workdir: Path,
    run_options: list[str],
) -> None:
    """Параметры запуска перед acp сессиям не передаются: молча их не теряем — ошибка (код 2)."""
    completed = _run_cli([*run_options, "acp"], user_home, workdir)

    assert completed.returncode == USAGE_ERROR, completed.stderr
    assert not completed.stdout
    error_line = completed.stderr.strip().splitlines()[-1]
    assert "acp не принимает параметры запуска" in error_line, completed.stderr


def _hooked_project(root: Path, scripts: Path, port: int) -> None:
    """Проект с подтверждённым хуком UserPromptSubmit, который ждёт, пока его не убьют."""
    hook = scripts / "hook.py"
    hook.write_text(HOOK_SCRIPT, encoding="utf-8")
    command = shlex.join([sys.executable, str(hook), str(port)])
    hook_spec = {"type": "command", "command": command, "timeout": HOOK_TIMEOUT}
    hooks = {"UserPromptSubmit": [{"hooks": [hook_spec]}]}
    (root / ".gigacode").mkdir()
    (root / ".gigacode" / "settings.json").write_text(
        json.dumps({"hooks": hooks}),
        encoding="utf-8",
    )
    workspace = Workspace(root)
    settings, _ = load_settings(workspace)
    TrustStore().trust(workspace.root, project_fingerprint(settings))


def _stop_by_eof(server: _AcpProcess) -> None:
    server.close_input()


def _stop_by_sigterm(server: _AcpProcess) -> None:
    server.proc.send_signal(signal.SIGTERM)


def _stop_by_sigint(server: _AcpProcess) -> None:
    server.proc.send_signal(signal.SIGINT)


@pytest.mark.skipif(sys.platform == "win32", reason="сигналы и группы процессов — POSIX")
@pytest.mark.parametrize(
    ("stop", "exit_code"),
    [(_stop_by_eof, 0), (_stop_by_sigterm, None), (_stop_by_sigint, None)],
    ids=["eof", "sigterm", "sigint"],
)
def test_stopping_during_a_turn_kills_its_processes(
    launch: _Launcher,
    root: Path,
    tmp_path_factory: pytest.TempPathFactory,
    stop: Callable[[_AcpProcess], None],
    exit_code: int | None,
) -> None:
    """Конец stdin, SIGTERM или SIGINT посреди хода: процесс выходит, процессы хода убиты.

    Ход висит в хуке UserPromptSubmit — дочернем процессе, который сам не завершится.
    Конец stdin — штатное завершение (T3 закрыл сессию): код 0. Код выхода по сигналу
    (``exit_code=None``) не проверяется: важно, что процесс вышел и не оставил детей.
    """
    with socket.create_server(("127.0.0.1", 0)) as listener:
        listener.settimeout(PROCESS_TIMEOUT)
        _hooked_project(root, tmp_path_factory.mktemp("hook"), listener.getsockname()[1])
        server = launch(API_INSTANCE)
        server.initialize()
        session = server.new_session(root)["result"]["sessionId"]
        server.begin("session/prompt", {"sessionId": session, "prompt": [_text("привет")]})
        conn, _ = listener.accept()  # хук запущен — ход идёт
    with conn:
        conn.settimeout(PROCESS_TIMEOUT)
        hook_pid = int(conn.recv(64))
        try:
            stop(server)
            code = server.finish()
            assert _closed_by_peer(conn), "процесс хука пережил gigacrew acp"
        finally:
            with suppress(ProcessLookupError):
                os.kill(hook_pid, signal.SIGKILL)
    if exit_code is not None:
        assert code == exit_code, server.stderr


def _closed_by_peer(conn: socket.socket) -> bool:
    """Другая сторона соединения закрыла его (процесс умер) за время ожидания."""
    try:
        return conn.recv(1) == b""
    except ConnectionResetError:
        return True
    except TimeoutError:
        return False


def _text(text: str) -> Message:
    return {"type": "text", "text": text}
