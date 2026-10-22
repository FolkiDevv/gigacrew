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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from gigacrew.agent.conversation import Conversation, request_text
from gigacrew.llm.types import Message
from gigacrew.project.workspace import Workspace

TITLE_LIMIT = 60
PREVIEW_LIMIT = 160
_ID_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z_-]*$")


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


@dataclass(frozen=True)
class SavedChat:
    info: ChatInfo
    conversation: Conversation


def _now() -> datetime:
    return datetime.now().astimezone().replace(microsecond=0)


def _parse_time(value: str) -> datetime:
    """Время из файла; без часового пояса (файл правили вручную) — считается местным.

    Иначе наивное и aware-время нельзя было бы сравнить при сортировке списка чатов.
    """
    when = datetime.fromisoformat(value)
    return when if when.tzinfo is not None else when.astimezone()


def _one_line(text: str, limit: int) -> str:
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


def chat_title(messages: Iterable[Message]) -> str:
    """Заголовок чата — первый запрос пользователя в одну строку."""
    for m in messages:
        if m.is_request and m.content.strip():
            return _one_line(request_text(m.content), TITLE_LIMIT)
    return "(без названия)"


def _preview(messages: tuple[Message, ...]) -> str:
    for m in reversed(messages):
        if m.role == "assistant" and not m.function_calls and m.content.strip():
            return _one_line(m.content, PREVIEW_LIMIT)
    return ""


def _message_count(data: dict[str, Any]) -> int:
    """Число сообщений чата; в файлах до появления ключа ``messages`` — по самому диалогу."""
    if "messages" in data:
        return int(data["messages"] or 0)
    raw = data.get("conversation")
    messages = raw.get("messages") if isinstance(raw, dict) else None
    return len(messages) if isinstance(messages, list) else 0


_GIT_TIMEOUT = 10
# Как у инструмента git: fsmonitor-хук из настроек репозитория — внешняя программа.
_GIT_SAFE_CONFIG = ("-c", "core.fsmonitor=false", "-c", "safe.bareRepository=explicit")


def _git_tracked(workspace: Workspace, directory: Path) -> set[str] | None:
    """Имена файлов ``directory`` (в git-репозитории), которые отслеживает git.

    None — не удалось узнать (git не установлен, ошибка): тогда переносить нельзя.
    """
    try:
        run = subprocess.run(
            ["git", *_GIT_SAFE_CONFIG, "-C", str(workspace.root), "ls-files", "-z", "--"]
            + [directory.relative_to(workspace.root).as_posix()],
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
        os.replace(src, dst)
        return
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
    fd, tmp = tempfile.mkstemp(prefix=f".{dst.stem}.", suffix=".tmp", dir=dst.parent)
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)  # mkstemp уже дал права 0600
        os.replace(tmp, dst)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    src.unlink()


class ChatStore:
    FORMAT_VERSION = 1

    def __init__(self, workspace: Workspace):
        """Конструктор не трогает файловую систему."""
        self._workspace = workspace
        self._directory: Path | None = None
        self._migrated = False

    @property
    def directory(self) -> Path:
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
                p for p in legacy.iterdir() if p.suffix == ".json" and _ID_RE.match(p.stem)
            )
        except OSError as e:
            return [f"не удалось прочитать прежний каталог чатов {legacy}: {e}"]
        if not files:
            return []
        if not self._workspace.in_git_repo():
            return [
                f"чаты в {legacy} не перенесены: проект не в git-репозитории, и не проверить, "
                "не пришли ли они вместе с ним; оттуда они не загружаются. Если это ваши "
                f"чаты, перенесите файлы в {self.directory} вручную"
            ]
        tracked = _git_tracked(self._workspace, legacy)
        if tracked is None:
            return [
                f"чаты в {legacy} не перенесены: не удалось проверить через git, не пришли ли "
                "они с репозиторием; оттуда они не загружаются"
            ]
        moved, shipped, existing, failed = 0, [], [], []
        for path in files:
            if path.name in tracked:
                shipped.append(path.stem)
                continue
            try:
                if not stat.S_ISREG(os.lstat(path).st_mode):
                    continue  # симлинк или не файл — не наш чат
                target = self._path(path.stem)
                if target.exists():
                    existing.append(path.stem)
                    continue
                self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                _move(path, target)
            except OSError as e:
                failed.append(f"{path.stem} ({e})")
                continue
            moved += 1
        with contextlib.suppress(OSError):
            legacy.rmdir()  # только пустой
        notices = []
        if moved:
            notices.append(
                f"перенесено чатов: {moved} — из {legacy} в {self.directory} (чаты теперь "
                "хранятся у пользователя, а не в проекте)"
            )
        if shipped:
            notices.append(
                f"чаты {', '.join(shipped)} в {legacy} отслеживает git — они пришли с "
                "репозиторием и не загружаются"
            )
        if existing:
            notices.append(
                f"чаты {', '.join(existing)} уже есть в {self.directory} — копии в {legacy} "
                "оставлены и не загружаются"
            )
        if failed:
            notices.append(
                f"не удалось перенести чаты из {legacy}: {'; '.join(failed)} — они остались "
                "на месте и не загружаются"
            )
        return notices

    # ------------------------------------------------------------------ #
    @staticmethod
    def new_id(now: datetime | None = None) -> str:
        """Id сортируется по времени создания; суффикс — от коллизий чатов одной секунды."""
        return f"{(now or _now()):%Y%m%d-%H%M%S}-{secrets.token_hex(4)}"

    def exists(self, chat_id: str) -> bool:
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

    def _path(self, chat_id: str) -> Path:
        if not _ID_RE.match(chat_id):
            raise ChatStoreError(f"некорректный id чата: {chat_id!r}")
        return self.directory / f"{chat_id}.json"

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
        info = ChatInfo(
            id=chat_id,
            title=chat_title(messages),
            created_at=created_at or now,
            updated_at=now,
            model=model,
            requests=sum(1 for m in messages if m.is_request),
            preview=_preview(messages),
            messages=len(messages),
        )
        data = {
            "version": self.FORMAT_VERSION,
            "id": info.id,
            "title": info.title,
            "created_at": info.created_at.isoformat(),
            "updated_at": info.updated_at.isoformat(),
            "model": info.model,
            "requests": info.requests,
            "preview": info.preview,
            "messages": info.messages,
            "conversation": conversation.to_dict(),
        }
        path = self._path(chat_id)
        self.migrate()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        # mkstemp создаёт файл с правами 0600 в том же каталоге — os.replace атомарен.
        fd, tmp = tempfile.mkstemp(prefix=f".{chat_id}.", suffix=".tmp", dir=self.directory)
        try:
            # backslashreplace: одиночный суррогат (имя файла не в UTF-8), если он всё же
            # попал в диалог, пишется экранированным «\udcee» — это корректный JSON.
            with os.fdopen(fd, "w", encoding="utf-8", errors="backslashreplace") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return info

    # ------------------------------------------------------------------ #
    def load(self, chat_id: str) -> SavedChat:
        path = self._path(chat_id)
        self.migrate()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ChatStoreError(f"чат {chat_id} не найден") from None
        except (OSError, ValueError) as e:
            raise ChatStoreError(f"не удалось прочитать чат {chat_id}: {e}") from None
        try:
            info = self._info(data, chat_id)
            raw = data["conversation"]
            if not isinstance(raw, dict):
                raise ValueError("поле conversation — не объект")
            conversation = Conversation.from_dict(raw)
        except (KeyError, TypeError, ValueError) as e:
            raise ChatStoreError(f"файл чата {chat_id} повреждён: {e}") from None
        conversation.repair()  # на случай файла, записанного посреди вызова инструмента
        return SavedChat(info, conversation)

    def _info(self, data: Any, chat_id: str) -> ChatInfo:
        if not isinstance(data, dict) or data.get("version") != self.FORMAT_VERSION:
            raise ValueError("неподдерживаемая версия формата")
        return ChatInfo(
            id=chat_id,
            title=str(data.get("title") or "(без названия)"),
            created_at=_parse_time(data["created_at"]),
            updated_at=_parse_time(data["updated_at"]),
            model=str(data.get("model") or ""),
            requests=int(data.get("requests") or 0),
            preview=str(data.get("preview") or ""),
            messages=_message_count(data),
        )

    def _files(self) -> list[Path]:
        """Файлы чатов, свежие первыми (по времени изменения)."""
        self.migrate()
        try:
            files = [p for p in self.directory.glob("*.json") if _ID_RE.match(p.stem)]
        except OSError:
            return []
        stamped = []
        for p in files:
            try:
                stamped.append((p.stat().st_mtime, p))
            except OSError:
                continue
        return [p for _, p in sorted(stamped, key=lambda x: (x[0], x[1].name), reverse=True)]

    def recent(self, limit: int | None = 20) -> list[ChatInfo]:
        """Последние чаты, свежие первыми. Битые файлы и чужие версии пропускаются."""
        result: list[ChatInfo] = []
        for path in self._files():
            if limit is not None and len(result) >= limit:
                break
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                result.append(self._info(data, path.stem))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        result.sort(key=lambda info: info.updated_at, reverse=True)
        return result

    def latest(self) -> SavedChat | None:
        """Последний чат, который удаётся прочитать (битые файлы пропускаются)."""
        for path in self._files():
            try:
                return self.load(path.stem)
            except ChatStoreError:
                continue
        return None

    def find(self, ref: str) -> ChatInfo:
        """Чат по точному id или однозначному началу id."""
        ref = ref.strip()
        if not ref:
            raise ChatStoreError("не указан id чата")
        ids = [p.stem for p in self._files()]
        if ref in ids:
            return self.load(ref).info
        matches = [i for i in ids if i.startswith(ref)]
        if not matches:
            raise ChatStoreError(f"чат {ref} не найден")
        if len(matches) > 1:
            shown = ", ".join(matches[:5])
            raise ChatStoreError(f"под «{ref}» подходит несколько чатов: {shown}")
        return self.load(matches[0]).info


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
        self.store = store
        self.enabled = enabled
        self._chat_id = store.new_id()
        self._created_at: datetime | None = None
        self._version: tuple[int, int, int] | None = None  # файла после своей записи
        self.title = ""  # заголовок продолженного чата (пусто — новый)

    @property
    def chat_id(self) -> str:
        return self._chat_id

    def start_new(self) -> None:
        self._chat_id = self.store.new_id()
        self._created_at = None
        self._version = None
        self.title = ""

    def switch_to(self, info: ChatInfo) -> None:
        """Дальнейшие сохранения идут в этот (продолженный) чат."""
        self._chat_id = info.id
        self._created_at = info.created_at
        self._version = self.store.version(info.id)
        self.title = info.title

    def save(self, conversation: Conversation, *, model: str = "") -> str | None:
        """Сохраняет диалог; возвращает предупреждение, если запись не удалась или
        диалог пришлось сохранить новым чатом."""
        if not self.enabled or not any(m.is_request for m in conversation.messages):
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
            info = self.store.save(
                self._chat_id, conversation, model=model, created_at=self._created_at
            )
        except (OSError, ValueError) as e:  # ValueError — в том числе UnicodeEncodeError
            self.enabled = False
            return f"чат не сохранён, автосохранение отключено: {e}"
        self._created_at = info.created_at
        self._version = self.store.version(self._chat_id)
        return warning
