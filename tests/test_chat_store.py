"""Тесты хранилища чатов: формат, атомарная запись, список, поиск, автосохранение."""

from __future__ import annotations

import json
import os
import stat
import sys
from datetime import UTC, datetime

import pytest

from gigacrew.agent.chat_store import ChatRecorder, ChatStore, ChatStoreError, chat_title
from gigacrew.agent.conversation import Conversation
from gigacrew.llm.types import FunctionCall, Message, Usage
from gigacrew.project.workspace import Workspace


def _conv(request="почини тесты", answer="готово") -> Conversation:
    conv = Conversation()
    conv.add_user(request)
    conv.add_assistant(
        Message(role="assistant", function_calls=[FunctionCall(name="read_file", arguments={})])
    )
    conv.add_function_result("read_file", "1\tx")
    conv.add_assistant(Message(role="assistant", content=answer), Usage(prompt_tokens=7))
    return conv


@pytest.fixture
def store(tmp_path) -> ChatStore:
    return ChatStore(Workspace(tmp_path))


def _touch(store: ChatStore, chat_id: str, mtime: float) -> None:
    os.utime(store.directory / f"{chat_id}.json", (mtime, mtime))


def test_round_trip(store, tmp_path):
    conv = _conv()
    info = store.save("20260101-120000-abcd", conv, model="Alpha-2-Max")
    assert info.title == "почини тесты" and info.requests == 1 and info.preview == "готово"

    saved = store.load("20260101-120000-abcd")
    assert saved.conversation.messages == conv.messages
    assert saved.conversation.last_usage == Usage(prompt_tokens=7)
    assert saved.info.model == "Alpha-2-Max"
    assert saved.info.created_at == info.created_at
    # служебная папка создана с .gitignore, временных файлов не осталось
    assert "/chats/" in (tmp_path / ".gigacode" / ".gitignore").read_text().splitlines()
    assert [p.name for p in store.directory.iterdir()] == ["20260101-120000-abcd.json"]


@pytest.mark.skipif(sys.platform == "win32", reason="права POSIX")
def test_file_is_private(store):
    store.save("20260101-120000-abcd", _conv())
    mode = stat.S_IMODE((store.directory / "20260101-120000-abcd.json").stat().st_mode)
    assert mode == 0o600


def test_load_repairs_pending_call(store):
    conv = Conversation()
    conv.add_user("x")
    conv.add_assistant(Message(role="assistant", function_calls=[FunctionCall(name="run_shell")]))
    store.save("a1", conv)
    assert store.load("a1").conversation.pending_calls() == []


def test_load_errors(store):
    with pytest.raises(ChatStoreError, match="не найден"):
        store.load("nope")
    with pytest.raises(ChatStoreError, match="некорректный"):
        store.load("../secret")
    store.save("bad", _conv())
    (store.directory / "bad.json").write_text("{не json", encoding="utf-8")
    with pytest.raises(ChatStoreError):
        store.load("bad")


@pytest.mark.parametrize("bad", ["../secret", "a/../../secret", "a/b", "a.json", "a\nb", ""])
def test_ids_that_could_leave_the_chats_dir_are_rejected(store, bad):
    # Первый символ id — буква или цифра, и весь id — без «/» и «.»: ни load, ни save, ни
    # exists не выходят за .gigacode/chats.
    (store.directory / "a").mkdir(parents=True)
    for action in (store.load, store.exists, lambda i: store.save(i, _conv())):
        with pytest.raises(ChatStoreError, match="некорректный"):
            action(bad)


@pytest.mark.parametrize("conversation", [[], "x", None, {"version": 1, "messages": 5}])
def test_load_rejects_malformed_conversation(store, conversation):
    store.save("odd", _conv())
    path = store.directory / "odd.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["conversation"] = conversation
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ChatStoreError, match="повреждён"):
        store.load("odd")
    assert [c.id for c in store.recent()] == ["odd"]  # метаданные целы — чат в списке
    assert store.latest() is None


def test_recent_order_and_skips_broken(store):
    for chat_id in ["c-old", "c-new", "c-mid"]:
        store.save(chat_id, _conv(f"запрос {chat_id}"))
        _touch(store, chat_id, 1_000_000 + {"c-old": 0, "c-mid": 1, "c-new": 2}[chat_id] * 10)
    (store.directory / "broken.json").write_text("[]", encoding="utf-8")
    other = json.loads((store.directory / "c-old.json").read_text())
    other["version"] = 99
    (store.directory / "future.json").write_text(json.dumps(other), encoding="utf-8")
    (store.directory / "notes.txt").write_text("x", encoding="utf-8")

    ids = [c.id for c in store.recent()]
    assert ids == ["c-new", "c-mid", "c-old"]  # updated_at равны — порядок по mtime
    assert [c.id for c in store.recent(limit=2)] == ["c-new", "c-mid"]


def test_recent_accepts_timestamps_without_timezone(store):
    store.save("aware", _conv("обычный"))
    store.save("naive", _conv("правленый вручную"))
    path = store.directory / "naive.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["created_at"] = data["updated_at"] = "2026-01-01T10:00:00"
    path.write_text(json.dumps(data), encoding="utf-8")
    chats = store.recent()
    assert [c.id for c in chats] == ["aware", "naive"]
    assert all(c.updated_at.tzinfo is not None for c in chats)
    assert store.load("naive").info.created_at.tzinfo is not None


def test_message_count_is_saved_and_read_from_older_files(store):
    """В списке видно, сколько в чате сообщений, а не только запросов. В файлах, записанных
    до появления счётчика, он считается по самому диалогу."""
    info = store.save("new", _conv())
    assert info.messages == 4 and info.requests == 1
    store.save("old", _conv())
    path = store.directory / "old.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["messages"]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert {c.id: c.messages for c in store.recent()} == {"new": 4, "old": 4}
    assert store.load("old").info.messages == 4


def test_recent_on_missing_dir(store):
    assert store.recent() == []
    assert store.latest() is None


def test_latest(store):
    store.save("a1", _conv("первый"))
    store.save("a2", _conv("второй"))
    _touch(store, "a1", 1_000)
    _touch(store, "a2", 2_000)
    latest = store.latest()
    assert latest is not None and latest.info.title == "второй"


def test_latest_skips_any_number_of_broken_chats(store):
    store.save("good", _conv("целый"))
    _touch(store, "good", 1_000)
    for i in range(7):  # метаданные целы, диалог испорчен
        store.save(f"bad{i}", _conv())
        path = store.directory / f"bad{i}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["conversation"] = []
        path.write_text(json.dumps(data), encoding="utf-8")
        _touch(store, f"bad{i}", 2_000 + i)
    latest = store.latest()
    assert latest is not None and latest.info.title == "целый"


def test_find_by_id_and_prefix(store):
    store.save("20260101-120000-aaaa", _conv())
    store.save("20260101-130000-bbbb", _conv())
    assert store.find("20260101-120000-aaaa").id == "20260101-120000-aaaa"
    assert store.find("20260101-13").id == "20260101-130000-bbbb"
    with pytest.raises(ChatStoreError, match="несколько"):
        store.find("20260101")
    with pytest.raises(ChatStoreError, match="не найден"):
        store.find("2027")
    with pytest.raises(ChatStoreError):
        store.find("  ")


def test_find_prefers_exact_id_over_longer_ids_with_that_prefix(store):
    store.save("a1", _conv("первый"))
    store.save("a10", _conv("второй"))
    assert store.find("a1").title == "первый"
    with pytest.raises(ChatStoreError, match="несколько"):
        store.find("a")


def test_failed_write_leaves_no_temporary_file(store, monkeypatch):
    import gigacrew.agent.chat_store as chat_store

    def full(*a, **kw):
        raise OSError("диск заполнен")

    monkeypatch.setattr(chat_store.json, "dump", full)
    with pytest.raises(OSError, match="заполнен"):
        store.save("a1", _conv())
    assert list(store.directory.iterdir()) == []


def test_new_ids_are_unique_and_sortable(store):
    ids = {store.new_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(len(i.split("-")) == 3 for i in ids)


def test_title():
    assert chat_title([]) == "(без названия)"
    long = "сделай   что-нибудь\nполезное " + "x" * 100
    title = chat_title([Message(role="user", content=long)])
    assert title.startswith("сделай что-нибудь полезное") and title.endswith("…")
    assert len(title) == 60
    meta = Message(role="user", content="<system-reminder>…</system-reminder>", meta=True)
    assert chat_title([meta, Message(role="user", content="задача")]) == "задача"


def test_meta_messages_are_not_counted_as_requests(store):
    conv = _conv()
    conv.add_user("<system-reminder>план одобрен</system-reminder>", meta=True)
    conv.add_assistant(Message(role="assistant", content="сделано"))
    info = store.save("20260101-120000-abcd", conv)
    assert info.requests == 1 and info.title == "почини тесты"


# ------------------------------ ChatRecorder ------------------------------ #
def test_recorder_skips_empty_and_keeps_created_at(store):
    rec = ChatRecorder(store)
    assert rec.save(Conversation()) is None
    assert store.recent() == []

    conv = _conv()
    assert rec.save(conv, model="m") is None
    first = store.load(rec.chat_id).info
    conv.add_user("ещё")
    rec.save(conv)
    again = store.load(rec.chat_id).info
    assert again.created_at == first.created_at and again.requests == 2
    assert [c.id for c in store.recent()] == [rec.chat_id]  # тот же файл, не новый
    # Продолженный чат хранит время создания, а не время продолжения (метки — в
    # секундах: в одной секунде проверка выше прошла бы и без этого).
    when = datetime(2020, 1, 1, tzinfo=UTC)
    assert store.save("old1", conv, created_at=when).created_at == when
    rec.switch_to(store.load("old1").info)
    conv.add_user("и ещё")
    rec.save(conv)
    info = store.load("old1").info
    assert info.created_at == when and info.updated_at > when and info.requests == 3


def test_recorder_does_not_save_harness_only_conversation(store):
    conv = Conversation()
    conv.add_user("<system-reminder>\nплан одобрен\n</system-reminder>", meta=True)
    assert ChatRecorder(store).save(conv) is None
    assert store.recent() == []


def test_recorder_new_and_switch(store):
    rec = ChatRecorder(store)
    rec.save(_conv())
    old = rec.chat_id
    rec.start_new()
    assert rec.chat_id != old and rec.title == ""

    info = store.load(old).info
    rec.switch_to(info)
    assert rec.chat_id == old and rec.title == info.title
    rec.save(_conv("другое"))
    assert len(store.recent()) == 1


def test_recorder_never_overwrites_another_new_chat(store, monkeypatch):
    first = ChatRecorder(store)
    first.save(_conv("первый"))
    ids = iter([first.chat_id, first.chat_id, "20260101-000000-ffffffff"])
    monkeypatch.setattr(store, "new_id", lambda now=None: next(ids))
    second = ChatRecorder(store)  # тот же id, что у первого
    assert second.chat_id == first.chat_id
    second.save(_conv("второй"))
    assert second.chat_id == "20260101-000000-ffffffff"
    assert store.load(first.chat_id).info.title == "первый"
    # продолженный чат по-прежнему дописывается в свой файл
    second.switch_to(store.load(first.chat_id).info)
    second.save(_conv("первый", answer="дописано"))
    assert second.chat_id == first.chat_id
    assert store.load(first.chat_id).info.preview == "дописано"


def test_recorder_disabled(store):
    rec = ChatRecorder(store, enabled=False)
    rec.save(_conv())
    assert store.recent() == []


def test_recorder_write_error_warns_once(store, monkeypatch):
    rec = ChatRecorder(store)

    def boom(*a, **kw):
        raise PermissionError("только чтение")

    monkeypatch.setattr(store, "save", boom)
    warning = rec.save(_conv())
    assert warning and "только чтение" in warning
    assert rec.enabled is False
    assert rec.save(_conv()) is None


# ------------------- имена файлов не в UTF-8 (одиночные суррогаты) ------------------- #
_CP1251_NAME = b"\xee\xf2\xf7\xb8\xf2.txt".decode("utf-8", "surrogateescape")  # «отчёт.txt»


def test_save_survives_lone_surrogates(store):
    # Диалог, собранный в обход Conversation.add_* (например, из старого файла), с
    # одиночными суррогатами: запись не падает, чат читается.
    conv = Conversation(
        [
            Message(role="user", content=f"открой {_CP1251_NAME}"),
            Message(role="assistant", content="ок \ud800"),
        ]
    )
    info = store.save("a1", conv)
    assert info.title.startswith("открой ")
    loaded = store.load("a1").conversation
    assert [m.role for m in loaded.messages] == ["user", "assistant"]


def test_recorder_serialization_error_warns_once(store, monkeypatch):
    rec = ChatRecorder(store)

    def boom(*a, **kw):
        raise UnicodeEncodeError("utf-8", "\udcee", 0, 1, "surrogates not allowed")

    monkeypatch.setattr(store, "save", boom)
    warning = rec.save(_conv())
    assert warning and "surrogates not allowed" in warning
    assert rec.enabled is False and rec.save(_conv()) is None


def test_two_sessions_continuing_one_chat_do_not_lose_turns(store):
    # Два терминала с `gigacrew -c`: второй сохранивший не затирает ход первого, а
    # продолжает в новом чате и предупреждает.
    ChatRecorder(store).save(_conv("запрос 1"))
    (info,) = store.recent()
    a, b = ChatRecorder(store), ChatRecorder(store)
    a.switch_to(store.load(info.id).info)
    b.switch_to(store.load(info.id).info)
    conv_a, conv_b = store.load(info.id).conversation, store.load(info.id).conversation
    conv_a.add_user("A: допиши docs")
    assert a.save(conv_a) is None and a.chat_id == info.id
    conv_b.add_user("B: допиши тесты")
    warning = b.save(conv_b)
    assert warning is not None and info.id in warning and b.chat_id in warning
    assert b.chat_id != info.id
    requests = [m.content for m in store.load(info.id).conversation.messages if m.is_request]
    assert requests == ["запрос 1", "A: допиши docs"]  # ход A сохранён
    forked = store.load(b.chat_id)
    assert [m.content for m in forked.conversation.messages if m.is_request] == [
        "запрос 1",
        "B: допиши тесты",
    ]
    # дальше каждый пишет в свой чат без предупреждений
    conv_b.add_user("B: ещё")
    assert b.save(conv_b) is None
    conv_a.add_user("A: ещё")
    assert a.save(conv_a) is None and a.chat_id == info.id
    assert len(store.recent()) == 2
