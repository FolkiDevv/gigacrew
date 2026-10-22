"""Сохранённые чаты: ``.gigacode/chats/<id>.json``.

Один файл на чат: метаданные (заголовок, время, модель) и диалог в формате
:meth:`Conversation.to_dict`. Запись атомарная (временный файл + ``os.replace``),
файл доступен только владельцу — в диалоге бывают секреты из вывода команд.

:class:`ChatStore` — хранилище, :class:`ChatRecorder` — текущий чат сессии
(автосохранение после хода, переключение по ``/resume``, новый чат по ``/clear``).
Модуль не знает о UI: ошибки возвращаются текстом или исключением :class:`ChatStoreError`.
"""

from __future__ import annotations

import json
import os
import re
import secrets
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


class ChatStore:
    FORMAT_VERSION = 1

    def __init__(self, workspace: Workspace):
        """Конструктор не трогает файловую систему."""
        self._workspace = workspace

    @property
    def directory(self) -> Path:
        return self._workspace.chats_dir

    # ------------------------------------------------------------------ #
    @staticmethod
    def new_id(now: datetime | None = None) -> str:
        """Id сортируется по времени создания; суффикс — от коллизий чатов одной секунды."""
        return f"{(now or _now()):%Y%m%d-%H%M%S}-{secrets.token_hex(4)}"

    def exists(self, chat_id: str) -> bool:
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
        self._workspace.ensure_data_dir()
        self.directory.mkdir(exist_ok=True)
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
