"""Сохранённые чаты: ``~/.gigacode/projects/<ключ проекта>/chats/<id>.json``.

Один файл на чат: метаданные (заголовок, время, модель) и диалог в формате
:meth:`Conversation.to_dict`. Запись атомарная (временный файл + ``os.replace``),
файл доступен только владельцу — в диалоге бывают секреты из вывода команд.

Чаты хранятся у пользователя, а не в проекте: продолженный чат — доверенные данные (его
запросы видит рецензент команд как запросы пользователя, его модель становится текущей),
а файл в проекте мог прийти вместе с репозиторием. Чаты с прежнего места
(``.gigacode/chats`` проекта) переносятся при первом обращении (:meth:`ChatStore.migrate`);
закоммиченные в git — нет. Из проекта чаты не читаются никогда.

:class:`ChatStore` — хранилище, :class:`ChatRecorder` — текущий чат сессии
(автосохранение после хода, переключение по ``/resume``, новый чат по ``/clear``).
Модуль не знает о UI: ошибки возвращаются текстом или исключением :class:`ChatStoreError`.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from gigacrew.agent.conversation import Conversation, request_text
from gigacrew.llm.types import Message
from gigacrew.project.workspace import Workspace

TITLE_LIMIT = 60
PREVIEW_LIMIT = 160
# Версия формата файла чата.
FORMAT_VERSION = 1
_ID_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z_-]*$")
# Время создания в начале id чата.
_ID_TIME_FORMAT = "%Y%m%d-%H%M%S"
# Каталог чатов доступен только владельцу.
_PRIVATE_DIR_MODE = 0o700


class ChatStoreError(Exception):
    """Чат не найден, ссылка неоднозначна или файл повреждён."""


@dataclass(frozen=True)
class ChatInfo:
    """Метаданные сохранённого чата (без самого диалога)."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    model: str = ""
    requests: int = 0  # запросов пользователя
    preview: str = ""  # начало последнего ответа модели
    messages: int = 0  # сообщений диалога, с шагами инструментов (0 — неизвестно)

    @classmethod
    def from_record(cls, record: Any, chat_id: str) -> ChatInfo:
        """Метаданные из разобранного JSON файла чата.

        Чужая версия формата — ``ValueError``; повреждённые поля — ``KeyError``,
        ``TypeError`` или ``ValueError``.
        """
        if not isinstance(record, dict) or record.get("version") != FORMAT_VERSION:
            raise ValueError("неподдерживаемая версия формата")
        return cls(
            id=chat_id,
            title=str(record.get("title") or "(без названия)"),
            created_at=_parse_time(record["created_at"]),
            updated_at=_parse_time(record["updated_at"]),
            model=str(record.get("model") or ""),
            requests=int(record.get("requests") or 0),
            preview=str(record.get("preview") or ""),
            messages=_message_count(record),
        )


@dataclass(frozen=True)
class SavedChat:
    """Сохранённый чат: метаданные и диалог."""

    info: ChatInfo
    conversation: Conversation


def _now() -> datetime:
    return datetime.now().astimezone().replace(microsecond=0)


def _parse_time(text: str) -> datetime:
    """Время из файла; без часового пояса (файл правили вручную) — считается местным.

    Иначе наивное и aware-время нельзя было бы сравнить при сортировке списка чатов.
    """
    when = datetime.fromisoformat(text)
    return when.astimezone() if when.tzinfo is None else when


def _one_line(text: str, limit: int) -> str:
    line = " ".join(text.split())
    if len(line) <= limit:
        return line
    head = line[: limit - 1].rstrip()
    return f"{head}…"


def chat_title(messages: Iterable[Message]) -> str:
    """Заголовок чата — первый запрос пользователя в одну строку."""
    for message in messages:
        if message.is_request and message.content.strip():
            return _one_line(request_text(message.content), TITLE_LIMIT)
    return "(без названия)"


def _preview(messages: tuple[Message, ...]) -> str:
    for message in reversed(messages):
        if message.role == "assistant" and not message.function_calls and message.content.strip():
            return _one_line(message.content, PREVIEW_LIMIT)
    return ""


def _message_count(record: dict[str, Any]) -> int:
    """Число сообщений чата; в файлах до появления ключа ``messages`` — по самому диалогу."""
    try:
        count = record["messages"]
    except KeyError:
        raw = record.get("conversation")
        messages = raw.get("messages") if isinstance(raw, dict) else None
        return len(messages) if isinstance(messages, list) else 0
    return int(count or 0)


def _saved_chat(record: Any, chat_id: str) -> SavedChat:
    """Чат из разобранного JSON файла.

    Повреждённый файл — ``KeyError``, ``TypeError`` или ``ValueError``.
    """
    chat_info = ChatInfo.from_record(record, chat_id)
    raw = record["conversation"]
    if not isinstance(raw, dict):
        raise TypeError("поле conversation — не объект")
    return SavedChat(chat_info, Conversation.from_dict(raw))


_GIT_TIMEOUT = 10
# Как у инструмента git: fsmonitor-хук из настроек репозитория — внешняя программа.
_GIT_SAFE_CONFIG = ("-c", "core.fsmonitor=false", "-c", "safe.bareRepository=explicit")


def _git_tracked(workspace: Workspace, directory: Path) -> set[str] | None:
    """Имена файлов ``directory`` (в git-репозитории), которые отслеживает git.

    None — не удалось узнать (git не установлен, ошибка): тогда переносить нельзя.
    """
    try:
        run = subprocess.run(
            [
                "git",
                *_GIT_SAFE_CONFIG,
                "-C",
                str(workspace.root),
                "ls-files",
                "-z",
                "--",
                directory.relative_to(workspace.root).as_posix(),
            ],
            capture_output=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if run.returncode != 0:
        return None
    names = run.stdout.decode("utf-8", "surrogateescape").split("\0")
    return {name.rsplit("/", 1)[-1] for name in names if name}


def _move(src: Path, dst: Path) -> None:
    """Переместить файл; на другую файловую систему — копия и удаление оригинала.

    Сбой до удаления оригинала — ``OSError``, оригинал на месте, неполной копии нет.
    """
    try:
        Path(src).replace(dst)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
    else:
        return
    fd, temp_name = tempfile.mkstemp(prefix=f".{dst.stem}.", suffix=".tmp", dir=dst.parent)
    os.close(fd)
    try:
        shutil.copyfile(src, temp_name)  # mkstemp уже дал права 0600
        Path(temp_name).replace(dst)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    src.unlink()


@dataclass
class _Migration:
    """Итог переноса чатов с прежнего места: сколько перенесено и что — нет."""

    moved: int = 0
    shipped: list[str] = field(default_factory=list)  # отслеживает git
    existing: list[str] = field(default_factory=list)  # чат с тем же id уже есть
    failed: list[str] = field(default_factory=list)  # «id (ошибка)»


class ChatStore:
    """Хранилище сохранённых чатов проекта в каталоге пользователя (:attr:`directory`)."""

    def __init__(self, workspace: Workspace):
        """Конструктор не трогает файловую систему."""
        self._workspace = workspace
        self._directory: Path | None = None
        self._migrated = False

    @property
    def directory(self) -> Path:
        """Каталог чатов проекта."""
        if self._directory is None:  # ключ проекта читает .git — один раз
            self._directory = self._workspace.chats_dir
        return self._directory

    # ------------------------------------------------------------------ #
    def migrate(self) -> list[str]:
        """Перенести чаты с прежнего места (``.gigacode/chats`` проекта) в :attr:`directory`.

        Выполняется один раз на хранилище (повторный вызов — пустой список), остальные
        методы вызывают его сами. Возвращает тексты для пользователя. Переносятся только
        обычные файлы (не симлинки), которые не отслеживает git: закоммиченный чат пришёл
        с репозиторием. Вне git-репозитория этого не узнать (проект мог прийти архивом) —
        там чаты не переносятся, пользователь переносит свои сам. Перенос — перемещение;
        не удалось — оригинал остаётся на месте (но и оттуда не читается). Чат с тем же id
        у пользователя не перезаписывается.
        """
        if self._migrated:
            return []
        self._migrated = True
        legacy = self._workspace.legacy_chats_dir
        try:
            if legacy.parent.is_symlink() or legacy.is_symlink() or not legacy.is_dir():
                return []
            files = sorted(
                candidate
                for candidate in legacy.iterdir()
                if candidate.suffix == ".json" and _ID_RE.match(candidate.stem)
            )
        except OSError as error:
            return [f"не удалось прочитать прежний каталог чатов {legacy}: {error}"]
        if not files:
            return []
        if not self._workspace.in_git_repo():
            return [
                (
                    f"чаты в {legacy} не перенесены: проект не в git-репозитории, и не проверить, "
                    "не пришли ли они вместе с ним; оттуда они не загружаются. Если это ваши "
                    f"чаты, перенесите файлы в {self.directory} вручную"
                ),
            ]
        tracked = _git_tracked(self._workspace, legacy)
        if tracked is None:
            return [
                (
                    f"чаты в {legacy} не перенесены: не удалось проверить через git, не пришли ли "
                    "они с репозиторием; оттуда они не загружаются"
                ),
            ]
        report = _Migration()
        for path in files:
            if path.name in tracked:
                report.shipped.append(path.stem)
                continue
            try:
                self._migrate_file(path, report)
            except OSError as move_error:
                report.failed.append(f"{path.stem} ({move_error})")
        with contextlib.suppress(OSError):
            legacy.rmdir()  # только пустой
        return self._migration_notices(legacy, report)

    # ------------------------------------------------------------------ #
    @staticmethod
    def new_id(now: datetime | None = None) -> str:
        """Id сортируется по времени создания; суффикс — от коллизий чатов одной секунды."""
        stamp = (now or _now()).strftime(_ID_TIME_FORMAT)
        suffix = secrets.token_hex(4)
        return f"{stamp}-{suffix}"

    def exists(self, chat_id: str) -> bool:
        """Есть ли сохранённый чат с этим id."""
        self.migrate()
        return self._path(chat_id).exists()

    def version(self, chat_id: str) -> tuple[int, int, int] | None:
        """Версия файла чата: inode, время изменения, размер (None — файла нет).

        Запись атомарна (новый файл + ``os.replace``), поэтому каждая запись — из
        этой сессии или из другой — меняет inode.
        """
        try:
            st = self._path(chat_id).stat()
        except OSError:
            return None
        return st.st_ino, st.st_mtime_ns, st.st_size

    def save(
        self,
        chat_id: str,
        conversation: Conversation,
        *,
        model: str = "",
        created_at: datetime | None = None,
    ) -> ChatInfo:
        """Записывает чат (атомарно). Ошибки записи — ``OSError``."""
        messages = conversation.messages
        now = _now()
        chat_info = ChatInfo(
            id=chat_id,
            title=chat_title(messages),
            created_at=created_at or now,
            updated_at=now,
            model=model,
            requests=sum(1 for message in messages if message.is_request),
            preview=_preview(messages),
            messages=len(messages),
        )
        record = {
            "version": FORMAT_VERSION,
            "id": chat_info.id,
            "title": chat_info.title,
            "created_at": chat_info.created_at.isoformat(),
            "updated_at": chat_info.updated_at.isoformat(),
            "model": chat_info.model,
            "requests": chat_info.requests,
            "preview": chat_info.preview,
            "messages": chat_info.messages,
            "conversation": conversation.to_dict(),
        }
        path = self._path(chat_id)
        self.migrate()
        self.directory.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIR_MODE)
        # mkstemp создаёт файл с правами 0600 в том же каталоге — os.replace атомарен.
        fd, temp_name = tempfile.mkstemp(prefix=f".{chat_id}.", suffix=".tmp", dir=self.directory)
        try:
            # backslashreplace: одиночный суррогат (имя файла не в UTF-8), если он всё же
            # попал в диалог, пишется экранированным «\udcee» — это корректный JSON.
            with os.fdopen(fd, "w", encoding="utf-8", errors="backslashreplace") as stream:
                json.dump(record, stream, ensure_ascii=False)
            Path(temp_name).replace(path)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise
        return chat_info

    # ------------------------------------------------------------------ #
    def load(self, chat_id: str) -> SavedChat:
        """Читает чат; не найден, не читается или повреждён — :class:`ChatStoreError`."""
        path = self._path(chat_id)
        self.migrate()
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ChatStoreError(f"чат {chat_id} не найден") from None
        except (OSError, ValueError) as read_error:
            raise ChatStoreError(f"не удалось прочитать чат {chat_id}: {read_error}") from None
        try:
            saved = _saved_chat(record, chat_id)
        except (KeyError, TypeError, ValueError) as parse_error:
            raise ChatStoreError(f"файл чата {chat_id} повреждён: {parse_error}") from None
        saved.conversation.repair()  # на случай файла, записанного посреди вызова инструмента
        return saved

    def recent(self, limit: int | None = 20) -> list[ChatInfo]:
        """Последние чаты, свежие первыми. Битые файлы и чужие версии пропускаются."""
        chats: list[ChatInfo] = []
        for path in self._files():
            if limit is not None and len(chats) >= limit:
                break
            with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
                record = json.loads(path.read_text(encoding="utf-8"))
                chats.append(ChatInfo.from_record(record, path.stem))
        chats.sort(key=lambda chat: chat.updated_at, reverse=True)
        return chats

    def latest(self) -> SavedChat | None:
        """Последний чат, который удаётся прочитать (битые файлы пропускаются)."""
        for path in self._files():
            with contextlib.suppress(ChatStoreError):
                return self.load(path.stem)
        return None

    def find(self, ref: str) -> ChatInfo:
        """Чат по точному id или однозначному началу id."""
        ref = ref.strip()
        if not ref:
            raise ChatStoreError("не указан id чата")
        ids = [path.stem for path in self._files()]
        if ref in ids:
            return self.load(ref).info
        matches = [chat_id for chat_id in ids if chat_id.startswith(ref)]
        if not matches:
            raise ChatStoreError(f"чат {ref} не найден")
        if len(matches) > 1:
            shown = ", ".join(matches[:5])
            raise ChatStoreError(f"под «{ref}» подходит несколько чатов: {shown}")
        return self.load(matches[0]).info

    # ------------------------------------------------------------------ #
    def _path(self, chat_id: str) -> Path:
        if not _ID_RE.match(chat_id):
            raise ChatStoreError(f"некорректный id чата: {chat_id!r}")
        return self.directory / f"{chat_id}.json"

    def _files(self) -> list[Path]:
        """Файлы чатов, свежие первыми (по времени изменения)."""
        self.migrate()
        try:
            files = [path for path in self.directory.glob("*.json") if _ID_RE.match(path.stem)]
        except OSError:
            return []
        stamped: list[tuple[float, Path]] = []
        for chat_file in files:
            with contextlib.suppress(OSError):
                stamped.append((chat_file.stat().st_mtime, chat_file))
        stamped.sort(key=lambda stamp: (stamp[0], stamp[1].name), reverse=True)
        return [path for _, path in stamped]

    def _migrate_file(self, path: Path, report: _Migration) -> None:
        """Перенести один чат с прежнего места; сбой — ``OSError``."""
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return  # симлинк или не файл — не наш чат
        target = self._path(path.stem)
        if target.exists():
            report.existing.append(path.stem)
            return
        self.directory.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIR_MODE)
        _move(path, target)
        report.moved += 1

    def _migration_notices(self, legacy: Path, report: _Migration) -> list[str]:
        """Тексты для пользователя об итогах переноса чатов."""
        notices = []
        if report.moved:
            notices.append(
                f"перенесено чатов: {report.moved} — из {legacy} в {self.directory} (чаты теперь "
                "хранятся у пользователя, а не в проекте)",
            )
        if report.shipped:
            shipped = ", ".join(report.shipped)
            notices.append(
                f"чаты {shipped} в {legacy} отслеживает git — они пришли с "
                "репозиторием и не загружаются",
            )
        if report.existing:
            existing = ", ".join(report.existing)
            notices.append(
                f"чаты {existing} уже есть в {self.directory} — копии в {legacy} "
                "оставлены и не загружаются",
            )
        if report.failed:
            failed = "; ".join(report.failed)
            notices.append(
                f"не удалось перенести чаты из {legacy}: {failed} — они остались "
                "на месте и не загружаются",
            )
        return notices


class ChatRecorder:
    """Текущий чат сессии: куда сохранять диалог после каждого хода.

    Если записать не удалось (проект только для чтения), сохранение отключается до
    конца сессии, а :meth:`save` один раз возвращает текст предупреждения.

    Один чат могут продолжить две сессии (два терминала с ``gigacrew -c``): каждая
    пишет файл целиком, и вторая затёрла бы ходы первой. Поэтому запоминается
    версия файла после своей записи (или при продолжении); если к следующей записи
    файл изменила другая сессия, диалог сохраняется новым чатом — с предупреждением.
    """

    def __init__(self, store: ChatStore, *, enabled: bool = True):
        """Новый чат в хранилище ``store``; ``enabled=False`` — без сохранения."""
        self.store = store
        self.enabled = enabled
        self._chat_id = store.new_id()
        self._created_at: datetime | None = None
        self._version: tuple[int, int, int] | None = None  # файла после своей записи
        self.title = ""  # заголовок продолженного чата (пусто — новый)

    @property
    def chat_id(self) -> str:
        """Id текущего чата."""
        return self._chat_id

    def start_new(self) -> None:
        """Дальнейшие сохранения идут в новый чат."""
        self._chat_id = self.store.new_id()
        self._created_at = None
        self._version = None
        self.title = ""

    def switch_to(self, chat: ChatInfo) -> None:
        """Дальнейшие сохранения идут в этот (продолженный) чат."""
        self._chat_id = chat.id
        self._created_at = chat.created_at
        self._version = self.store.version(chat.id)
        self.title = chat.title

    def save(self, conversation: Conversation, *, model: str = "") -> str | None:
        """Сохраняет диалог; возвращает предупреждение или None.

        Предупреждение — если запись не удалась или диалог пришлось сохранить новым чатом.
        """
        if not self.enabled or not any(message.is_request for message in conversation.messages):
            return None
        warning = None
        try:
            if self._created_at is not None and self._version is not None:
                current = self.store.version(self._chat_id)
                if current is not None and current != self._version:
                    # Чат изменила другая сессия: не затирать её ходы — новый чат.
                    old = self._chat_id
                    self._chat_id = self.store.new_id()
                    self._created_at = None
                    warning = (
                        f"чат {old} изменила другая сессия gigacrew — чтобы не затереть её "
                        f"ходы, этот диалог сохранён отдельным чатом {self._chat_id}"
                    )
            if self._created_at is None:
                # Первая запись нового чата: id не должен затереть чужой чат
                # (другое окно gigacrew, начатое в ту же секунду).
                while self.store.exists(self._chat_id):
                    self._chat_id = self.store.new_id()
            saved = self.store.save(
                self._chat_id,
                conversation,
                model=model,
                created_at=self._created_at,
            )
        except (OSError, ValueError) as error:  # ValueError — в том числе UnicodeEncodeError
            self.enabled = False
            return f"чат не сохранён, автосохранение отключено: {error}"
        self._created_at = saved.created_at
        self._version = self.store.version(self._chat_id)
        return warning
