"""Тесты CLI (без сети): разбор команд, реестр, REPL, одноразовый режим."""

from __future__ import annotations

import pytest
from fakes import CatalogueProvider, ScriptedProvider, WindowProvider, text_turn, tool_turn

import gigacrew.cli.app as app
from gigacrew.agent.chat_store import ChatRecorder, ChatStore
from gigacrew.agent.loop import Agent
from gigacrew.cli.commands import (
    CommandContext,
    SlashCommand,
    default_commands,
    is_repl_command,
)
from gigacrew.cli.repl import run_repl
from gigacrew.config import Config
from gigacrew.permissions import PermissionMode
from gigacrew.project.workspace import Workspace
from gigacrew.tools.base import build_default_registry
from gigacrew.ui.console import Console


@pytest.mark.parametrize(
    "line",
    ["/help", "/exit", "/q", "/clear", "/model Alpha-2-Max", "/model"],
)
def test_recognised_commands(line):
    assert is_repl_command(line) is True


@pytest.mark.parametrize(
    "line",
    [
        "/home/user/repos/gigacrew",  # абсолютный путь — НЕ команда
        "/usr/bin/python3 запусти это",
        "/",  # просто слеш
        "проанализируй /home/user/x",  # путь внутри запроса
        "посмотри код",
        "",
    ],
)
def test_non_commands_go_to_agent(line):
    assert is_repl_command(line) is False


# ------------------------------ реестр ------------------------------ #
@pytest.fixture
def cli_env(tmp_path, capsys):
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, auto_approve=True)
    provider = ScriptedProvider()
    ui = Console(no_color=True)
    agent = Agent(provider, build_default_registry(), cfg, ui)
    commands = default_commands()
    ctx = CommandContext(agent=agent, ui=ui, commands=commands)
    return agent, ui, commands, ctx, provider


def _out(capsys) -> str:
    return capsys.readouterr().out


def test_help_lists_all_commands(cli_env, capsys):
    _, _, commands, ctx, _ = cli_env
    assert commands.dispatch("/help", ctx) is True
    out = _out(capsys)
    for cmd in commands:
        assert cmd.name in out
    assert "/quit" in out


def test_unknown_command_suggests_closest(cli_env, capsys):
    _, _, commands, ctx, _ = cli_env
    assert commands.dispatch("/hlep", ctx) is True
    assert "неизвестная команда: /hlep — возможно, /help?" in _out(capsys)
    commands.dispatch("/zzzz", ctx)
    out = _out(capsys)
    assert "неизвестная команда: /zzzz (список — /help)" in out and "возможно" not in out


def test_aliases_and_case_insensitive(cli_env, capsys):
    _, _, commands, ctx, _ = cli_env
    assert commands.get("/Q") is commands.get("/exit")
    assert commands.dispatch("/QUIT", ctx) is False


def test_unknown_command_reports_and_continues(cli_env, capsys):
    _, _, commands, ctx, _ = cli_env
    assert commands.dispatch("/nope", ctx) is True
    assert "неизвестная команда" in _out(capsys)


def test_duplicate_registration_rejected(cli_env):
    _, _, commands, _, _ = cli_env
    with pytest.raises(ValueError):
        commands.register(SlashCommand("/x", "x", lambda c, a: True, aliases=("/help",)))


def test_model_and_clear_commands(cli_env, capsys):
    agent, _, commands, ctx, _ = cli_env
    commands.dispatch("/model Alpha-2-Max", ctx)
    assert agent.model == "Alpha-2-Max"
    commands.dispatch("/model", ctx)
    assert "Alpha-2-Max" in _out(capsys)
    agent.conversation.add_user("старое")
    commands.dispatch("/clear", ctx)
    assert len(agent.conversation) == 0


# ------------------------------- REPL ------------------------------- #
def _reader(*items):
    queue = list(items)

    def read():
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return read


def test_repl_ctrl_c_at_prompt_does_not_exit(cli_env, capsys):
    agent, ui, commands, _, _ = cli_env
    read = _reader(KeyboardInterrupt(), "/exit")
    assert run_repl(agent, ui, commands, read_input=read) == 0
    out = _out(capsys)
    assert "Ctrl+D" in out and "/model" in out  # подсказка + баннер из реестра


def test_repl_survives_errors_in_turn(cli_env, capsys, monkeypatch):
    agent, ui, commands, _, provider = cli_env

    def broken(_line):
        raise RuntimeError("что-то сломалось")

    monkeypatch.setattr(agent, "run_turn", broken)
    read = _reader("сделай", "ещё раз", EOFError())
    assert run_repl(agent, ui, commands, read_input=read) == 0
    assert _out(capsys).count("что-то сломалось") == 2


# ---------------------------- one-shot e2e ---------------------------- #
@pytest.fixture
def oneshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIGACREW_ACCESS_KEY", "dummy")
    monkeypatch.setenv("GIGACREW_AUTH_URL", "https://auth.test/oauth")
    monkeypatch.setenv("GIGACREW_API_URL", "https://api.test/v1")
    monkeypatch.setenv("GIGACREW_MODEL", "M-default")
    for name in (
        "GIGACREW_TIMEOUT",
        "GIGACREW_TEMPERATURE",
        "GIGACREW_CONTEXT_TOKENS",
        "GIGACREW_SAVE_CHATS",
        "GIGACREW_MODE",
        "GIGACREW_AUTO_COMPACT",
        "GIGACREW_COMPACT_THRESHOLD",
        "GIGACREW_AUTO_MEASURE",
        "GIGACREW_PROVIDER",
        "GIGACREW_PLUGIN_NAME",
    ):
        monkeypatch.delenv(name, raising=False)

    def use(provider):
        monkeypatch.setattr(app, "_make_provider", lambda _cfg, _ui: provider)

    return use


def test_oneshot_end_to_end(tmp_path, oneshot, capsys):
    (tmp_path / "hello.txt").write_text("привет из файла", encoding="utf-8")
    provider = ScriptedProvider(
        [tool_turn("read_file", {"path": "hello.txt"}), text_turn("В файле приветствие.")]
    )
    oneshot(provider)
    code = app.main(["-C", str(tmp_path), "-p", "что в hello.txt?", "--no-color", "--no-stream"])
    assert code == 0
    out = capsys.readouterr().out
    assert "В файле приветствие." in out and "read_file" in out
    assert "привет из файла" in provider.requests[-1]["messages"][-1].content
    # чат сохранён — его можно продолжить через -c
    (saved,) = ChatStore(Workspace(tmp_path)).recent()
    assert saved.title == "что в hello.txt?" and saved.requests == 1


def test_oneshot_ctrl_c_exit_code(tmp_path, oneshot):
    oneshot(ScriptedProvider([KeyboardInterrupt()]))
    assert app.main(["-C", str(tmp_path), "-p", "x", "--no-color"]) == 130


def test_oneshot_llm_error_exit_code(tmp_path, oneshot):
    from gigacrew.llm.base import LLMError

    oneshot(ScriptedProvider([LLMError("сеть упала")]))
    assert app.main(["-C", str(tmp_path), "-p", "x", "--no-color"]) == 2


def test_missing_project_dir(tmp_path, oneshot):
    assert app.main(["-C", str(tmp_path / "nope"), "-p", "x", "--no-color"]) == 1


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        app.main(["--version"])
    assert exc.value.code == 0
    assert "gigacrew" in capsys.readouterr().out


def test_repl_status_and_auto_approve_banner(cli_env, capsys):
    from gigacrew.cli.repl import status_of

    agent, ui, commands, _, _ = cli_env  # cli_env собран с auto_approve=True
    status = status_of(agent)
    assert status.model == agent.model and status.auto_approve is True
    assert status.context_budget == agent.context_budget
    assert run_repl(agent, ui, commands, read_input=_reader("/exit")) == 0
    assert "авто-подтверждение" in _out(capsys)


def test_repl_prefills_typeahead_and_uses_esc(cli_env, capsys):
    import contextlib

    agent, ui, commands, _, provider = cli_env
    (agent.workspace.root / ".git").mkdir()  # автоиндекс при старте — только в репозитории

    class FakeEsc:
        enabled = True
        entered = 0

        def __enter__(self):
            self.entered += 1
            return self

        def __exit__(self, *exc):
            return None

        def paused(self):
            return contextlib.nullcontext()

        typeahead = ["", "набрано во время хода"]  # после построения индекса, после хода

        def take_typeahead(self):
            return self.typeahead.pop(0) if self.typeahead else ""

    calls = []
    answers = ["привет", "/exit"]

    def read(**kwargs):
        calls.append(kwargs)
        return answers.pop(0)

    esc = FakeEsc()
    assert run_repl(agent, ui, commands, read_input=read, interrupt=esc) == 0
    assert esc.entered == 2  # построение индекса и ход — внутри перехвата Esc, /exit — нет
    assert calls == [{}, {"default": "набрано во время хода"}]
    assert ui._interrupt_hint == "Esc — прервать"


def test_ctrl_c_discards_prefilled_typeahead(cli_env):
    import contextlib

    agent, ui, commands, _, _ = cli_env

    class FakeEsc:
        enabled = True

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def paused(self):
            return contextlib.nullcontext()

        typeahead = ["", "набрано"]

        def take_typeahead(self):
            return self.typeahead.pop(0) if self.typeahead else ""

    calls = []
    answers = ["привет", KeyboardInterrupt(), "/exit"]

    def read(**kwargs):
        calls.append(kwargs)
        item = answers.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    run_repl(agent, ui, commands, read_input=read, interrupt=FakeEsc())
    assert calls == [{}, {"default": "набрано"}, {}]


def test_plain_reader_keeps_typeahead(monkeypatch):
    from gigacrew.cli import prompt

    monkeypatch.setattr("builtins.input", lambda p: " и ещё")
    assert prompt._read_plain("набрано") == "набрано и ещё"


# ---------------------------- сохранение чатов ---------------------------- #
def _main(tmp_path, *args: str) -> int:
    return app.main(["-C", str(tmp_path), "--no-color", "--no-stream", *args])


def _store(tmp_path) -> ChatStore:
    return ChatStore(Workspace(tmp_path))


def _history(provider) -> list[tuple[str, str]]:
    return [(m.role, m.content) for m in provider.requests[-1]["messages"][1:]]


def test_oneshot_no_save(tmp_path, oneshot, monkeypatch):
    oneshot(ScriptedProvider([text_turn("ok")]))
    assert _main(tmp_path, "-p", "x", "--no-save") == 0
    monkeypatch.setenv("GIGACREW_SAVE_CHATS", "0")
    assert _main(tmp_path, "-p", "x") == 0
    assert not (tmp_path / ".gigacrew").exists()


def test_oneshot_continue_appends_to_latest_chat(tmp_path, oneshot):
    oneshot(ScriptedProvider([text_turn("первый ответ")]))
    assert _main(tmp_path, "-p", "первый") == 0
    provider = ScriptedProvider([text_turn("второй ответ")])
    oneshot(provider)
    assert _main(tmp_path, "-c", "-p", "второй") == 0
    assert _history(provider) == [
        ("user", "первый"),
        ("assistant", "первый ответ"),
        ("user", "второй"),
    ]
    (chat,) = _store(tmp_path).recent()
    assert chat.title == "первый" and chat.requests == 2 and chat.preview == "второй ответ"


def test_continue_without_chats_starts_new(tmp_path, oneshot, capsys):
    oneshot(ScriptedProvider([text_turn("ok")]))
    assert _main(tmp_path, "-c", "-p", "x") == 0
    assert "сохранённых чатов нет" in capsys.readouterr().out


def test_resume_by_id_and_unknown_id(tmp_path, oneshot, capsys):
    oneshot(ScriptedProvider([text_turn("a")]))
    _main(tmp_path, "-p", "первый")
    (chat,) = _store(tmp_path).recent()
    provider = ScriptedProvider([text_turn("b")])
    oneshot(provider)
    assert _main(tmp_path, "--resume", chat.id[:15], "-p", "дальше") == 0
    assert _history(provider)[0] == ("user", "первый")
    assert _main(tmp_path, "-r", "1999", "-p", "x") == 1
    assert "не найден" in capsys.readouterr().out


def test_resume_without_id_opens_picker(tmp_path, oneshot, monkeypatch):
    oneshot(ScriptedProvider([text_turn("a"), text_turn("b")]))
    _main(tmp_path, "-p", "первый")
    _main(tmp_path, "--no-save", "-p", "мимо")  # не сохраняется — в списке один чат
    shown = []

    def pick(self, chats, current_id=""):
        shown.append(chats)
        return chats[0]

    monkeypatch.setattr(Console, "pick_chat", pick)
    provider = ScriptedProvider([text_turn("c")])
    oneshot(provider)
    assert _main(tmp_path, "-r", "-p", "дальше") == 0
    assert [c.title for c in shown[0]] == ["первый"]
    assert _history(provider)[0] == ("user", "первый")
    # отмена выбора — новый чат
    monkeypatch.setattr(Console, "pick_chat", lambda self, chats, current_id="": None)
    provider = ScriptedProvider([text_turn("d")])
    oneshot(provider)
    assert _main(tmp_path, "-r", "-p", "новый") == 0
    assert _history(provider) == [("user", "новый")]
    assert len(_store(tmp_path).recent()) == 2


def test_continue_and_resume_are_exclusive(tmp_path, oneshot):
    with pytest.raises(SystemExit):
        _main(tmp_path, "-c", "-r", "x")


def test_interrupted_oneshot_is_saved_consistently(tmp_path, oneshot):
    oneshot(ScriptedProvider([tool_turn("read_file", {"path": "a.txt"}), KeyboardInterrupt()]))
    assert _main(tmp_path, "-p", "прочитай") == 130
    store = _store(tmp_path)
    (chat,) = store.recent()
    conversation = store.load(chat.id).conversation
    assert conversation.pending_calls() == []
    assert [m.role for m in conversation.messages] == ["user", "assistant", "function"]


@pytest.fixture
def repl_env(cli_env):
    agent, ui, commands, ctx, provider = cli_env
    ctx.chats = ChatRecorder(ChatStore(agent.workspace))
    return agent, ui, commands, ctx, provider


def test_repl_autosaves_after_each_turn(repl_env):
    agent, ui, commands, ctx, provider = repl_env
    provider._turns = [text_turn("ответ 1"), KeyboardInterrupt(), text_turn("ответ 3")]
    read = _reader("первый", "второй", "третий", "/exit")
    assert run_repl(agent, ui, commands, read_input=read, chats=ctx.chats) == 0
    (chat,) = ctx.chats.store.recent()
    assert chat.id == ctx.chats.chat_id and chat.requests == 3 and chat.preview == "ответ 3"


def test_repl_save_error_warns_once(repl_env, capsys, monkeypatch):
    agent, ui, commands, ctx, provider = repl_env

    def boom(*a, **kw):
        raise OSError("диск только для чтения")

    monkeypatch.setattr(ctx.chats.store, "save", boom)
    read = _reader("раз", "два", "/exit")
    assert run_repl(agent, ui, commands, read_input=read, chats=ctx.chats) == 0
    assert _out(capsys).count("диск только для чтения") == 1


def test_clear_starts_new_chat_and_resume_returns(repl_env, capsys):
    agent, _, commands, ctx, provider = repl_env
    provider._turns = [text_turn("старый ответ"), text_turn("новый ответ")]
    agent.run_turn("старый вопрос")
    ctx.chats.save(agent.conversation)
    old_id = ctx.chats.chat_id
    commands.dispatch("/clear", ctx)
    assert ctx.chats.chat_id != old_id and len(agent.conversation) == 0
    assert agent.chat_id == ctx.chats.chat_id  # у нового чата — свой файл плана

    commands.dispatch(f"/resume {old_id}", ctx)
    assert ctx.chats.chat_id == old_id
    assert agent.plan_path.name == f"{old_id}.md"  # продолженный чат находит свой план
    out = _out(capsys)
    assert "продолжаем чат «старый вопрос»" in out and "старый ответ" in out
    agent.run_turn("ещё")
    assert _history(provider)[0] == ("user", "старый вопрос")


def test_compact_command_summarizes_and_saves(repl_env, capsys):
    agent, _, commands, ctx, provider = repl_env
    commands.dispatch("/compact", ctx)
    assert "сжимать нечего" in _out(capsys)

    provider._turns = [text_turn("старый ответ " + "y" * 400), text_turn("новый ответ")]
    provider.summaries = [text_turn("СВОДКА")]
    agent.run_turn("старый вопрос " + "x" * 400)
    assert commands.dispatch("/compact только суть", ctx) is True
    out = _out(capsys)
    assert "контекст сжат" in out and "свёрнуто 2 сообщения" in out
    assert "только суть" in provider.summary_requests[0]["messages"][1].content

    # Чат сохранён вместе с кратким содержанием: продолжение видит сводку, не историю.
    saved = ctx.chats.store.load(ctx.chats.chat_id)
    assert saved.conversation.summary is not None and saved.info.title.startswith("старый вопрос")
    commands.dispatch("/clear", ctx)
    commands.dispatch(f"/resume {saved.info.id}", ctx)
    assert "старый вопрос" in _out(capsys)  # показ последнего обмена — из полного журнала
    agent.run_turn("дальше")
    assert _history(provider) == [("user", "дальше")]
    assert "СВОДКА" in provider.requests[-1]["messages"][0].content


def test_compact_command_errors_keep_history(repl_env, capsys):
    from gigacrew.llm.base import LLMError

    agent, _, commands, ctx, provider = repl_env
    provider._turns = [text_turn("ответ")]
    agent.run_turn("вопрос")
    provider.summaries = [LLMError("сеть упала"), KeyboardInterrupt()]
    commands.dispatch("/compact", ctx)
    assert "не удалось сжать контекст: сеть упала" in _out(capsys)
    commands.dispatch("/compact", ctx)
    assert "сжатие прервано" in _out(capsys)
    assert agent.conversation.summary is None


def test_resume_picker_and_errors(repl_env, capsys, monkeypatch):
    agent, ui, commands, ctx, _ = repl_env
    commands.dispatch("/resume", ctx)
    assert "сохранённых чатов пока нет" in _out(capsys)

    agent.conversation.add_user("сохранённый")
    ctx.chats.save(agent.conversation)
    saved_id = ctx.chats.chat_id
    commands.dispatch("/clear", ctx)
    calls = []

    def pick(chats, current_id=""):
        calls.append((chats, current_id))
        return None  # отмена — ничего не меняется

    monkeypatch.setattr(ui, "pick_chat", pick)
    commands.dispatch("/chats", ctx)  # алиас
    assert [c.id for c in calls[0][0]] == [saved_id] and calls[0][1] == ctx.chats.chat_id
    assert len(agent.conversation) == 0

    monkeypatch.setattr(ui, "pick_chat", lambda chats, current_id="": chats[0])
    commands.dispatch("/resume", ctx)
    assert agent.conversation.messages[0].content == "сохранённый"

    commands.dispatch("/resume nope", ctx)
    assert "не найден" in _out(capsys)
    ctx.chats = None
    commands.dispatch("/resume", ctx)
    assert "недоступны" in _out(capsys)


def test_repl_shows_resumed_chat_after_banner(repl_env, capsys):
    agent, ui, commands, ctx, _ = repl_env
    agent.conversation.add_user("прошлый вопрос")
    info = ctx.chats.store.save("20260101-000000-abcd", agent.conversation)
    run_repl(agent, ui, commands, read_input=_reader("/exit"), chats=ctx.chats, resumed=info)
    out = _out(capsys)
    assert out.index("модель") < out.index("продолжаем чат «прошлый вопрос»")


# ------------------------------ индекс проекта ------------------------------ #
def _index_complete(agent) -> bool:
    from gigacrew.project.index import ProjectIndex

    with ProjectIndex(agent.workspace) as ix:
        return ix.is_complete()


def test_repl_builds_index_before_first_input(cli_env, capsys):
    agent, ui, commands, _, _ = cli_env
    (agent.workspace.root / ".git").mkdir()  # автоиндекс — только в git-репозитории
    (agent.workspace.root / "mod.py").write_text("def f():\n    pass\n", encoding="utf-8")
    seen = []

    def read(**kwargs):
        seen.append(_index_complete(agent))  # к первому вводу индекс уже готов
        return "/exit"

    assert run_repl(agent, ui, commands, read_input=read) == 0
    assert seen == [True]
    out = _out(capsys)
    assert "индексирую проект…" in out and "индекс проекта: 1 файл, 1 определение" in out

    # индекс готов — при следующем запуске не строится заново
    assert run_repl(agent, ui, commands, read_input=_reader("/exit")) == 0
    assert "индексирую" not in _out(capsys)


def test_repl_does_not_index_outside_git_repo(cli_env, capsys):
    """Запуск в $HOME или другой не-проектной папке не обходит всё её дерево."""
    agent, ui, commands, ctx, _ = cli_env
    (agent.workspace.root / "mod.py").write_text("def f():\n    pass\n", encoding="utf-8")
    assert run_repl(agent, ui, commands, read_input=_reader("/exit")) == 0
    out = _out(capsys)
    assert "не в git-репозитории" in out and "индексирую" not in out
    assert not (agent.workspace.root / ".gigacrew").exists()
    commands.dispatch("/index", ctx)  # вручную — строится
    assert "индекс проекта: 1 файл, 1 определение" in _out(capsys)


def test_repl_warns_when_tls_verification_is_off(tmp_path, capsys):
    from dataclasses import replace

    from gigacrew.cli.repl import TLS_OFF_WARNING

    agent, ui, ctx, _ = _measure_env(tmp_path, auto_approve=True)
    run_repl(agent, ui, ctx.commands, read_input=_reader("/exit"), index_on_start=False)
    assert " ".join(TLS_OFF_WARNING.split()[:4]) in _out(capsys)
    agent._cfg = replace(agent.config, verify_ssl=True)
    run_repl(agent, ui, ctx.commands, read_input=_reader("/exit"), index_on_start=False)
    assert "TLS" not in _out(capsys)


def test_repl_without_index_on_start(cli_env):
    agent, ui, commands, _, _ = cli_env
    run_repl(agent, ui, commands, read_input=_reader("/exit"), index_on_start=False)
    assert not (agent.workspace.root / ".gigacrew").exists()


def test_cancelled_index_build_lets_user_continue(cli_env, capsys, monkeypatch):
    from gigacrew.project import index as index_mod

    agent, ui, commands, _, _ = cli_env
    (agent.workspace.root / ".git").mkdir()
    for i in range(3):
        (agent.workspace.root / f"m{i}.py").write_text("x = 1\n", encoding="utf-8")
    real_refresh = index_mod.ProjectIndex.refresh

    def interrupted(self, base=None, *, progress=None):
        def stop(n):
            if n == 2:
                raise KeyboardInterrupt  # так приходит Esc/Ctrl+C
            progress(n)

        return real_refresh(self, base, progress=stop)

    monkeypatch.setattr(index_mod.ProjectIndex, "refresh", interrupted)
    assert run_repl(agent, ui, commands, read_input=_reader("/exit")) == 0
    assert "построение индекса прервано (просмотрено: 1 файл" in _out(capsys)
    assert not _index_complete(agent)

    # следующий запуск достраивает индекс
    monkeypatch.setattr(index_mod.ProjectIndex, "refresh", real_refresh)
    assert run_repl(agent, ui, commands, read_input=_reader("/exit")) == 0
    assert "индекс проекта: 3 файла" in _out(capsys)


def test_index_build_error_does_not_block_repl(cli_env, capsys, monkeypatch):
    from gigacrew.project import index as index_mod

    agent, ui, commands, _, _ = cli_env
    (agent.workspace.root / ".git").mkdir()

    def broken(self):
        raise OSError("только чтение")

    monkeypatch.setattr(index_mod.ProjectIndex, "open", broken)
    assert run_repl(agent, ui, commands, read_input=_reader("/exit")) == 0
    assert "индекс проекта недоступен: только чтение" in _out(capsys)


def test_index_command(cli_env, capsys):
    agent, ui, commands, ctx, _ = cli_env
    (agent.workspace.root / "a.py").write_text("class A:\n    pass\n", encoding="utf-8")
    assert commands.dispatch("/index", ctx) is True
    out = _out(capsys)
    assert "индекс проекта: 1 файл, 1 определение" in out
    assert "языки: python 1" in out and "добавлено 1" in out
    commands.dispatch("/index", ctx)
    assert "изменений нет" in _out(capsys)
    commands.dispatch("/index rebuild", ctx)
    assert "добавлено 1" in _out(capsys)
    commands.dispatch("/index что-то", ctx)
    assert "использование: /index [rebuild]" in _out(capsys)


# ------------------------- модели и окно контекста ------------------------- #
def _models_json(home):
    import json

    path = home / ".gigacrew" / "models.json"
    return json.loads(path.read_text(encoding="utf-8"))["models"] if path.exists() else {}


def test_test_context_measures_and_saves(tmp_path, oneshot, capsys, _isolated_home):
    provider = WindowProvider(32_768)
    oneshot(provider)
    assert _main(tmp_path, "--test-context", "Beta-32B") == 0
    out = capsys.readouterr().out
    assert "отказ (422)" in out and "прошло" in out and "окно Beta-32B:" in out
    entry = _models_json(_isolated_home)["Beta-32B"]
    assert 32_000 <= entry["context_window"] <= 32_768
    assert {model for model, _ in provider.measured} == {"Beta-32B"}


def test_test_context_errors(tmp_path, oneshot, capsys, _isolated_home):
    from gigacrew.llm.base import LLMError

    oneshot(ScriptedProvider())
    assert _main(tmp_path, "--test-context", "M") == 1
    assert "не поддерживает" in capsys.readouterr().out
    oneshot(WindowProvider(32_768, error=LLMError("нет такой модели")))
    assert _main(tmp_path, "--test-context", "M") == 2
    assert "не удалось замерить окно M" in capsys.readouterr().out
    oneshot(WindowProvider(32_768, error=KeyboardInterrupt()))
    assert _main(tmp_path, "--test-context", "M") == 130
    assert _models_json(_isolated_home) == {}


def test_test_context_reports_failed_save(tmp_path, oneshot, capsys, monkeypatch):
    from gigacrew.llm.model_windows import ModelWindows

    def readonly(self, result, *, base_url=""):
        raise OSError("только для чтения")

    monkeypatch.setattr(ModelWindows, "record", readonly)
    oneshot(WindowProvider(32_768))
    assert _main(tmp_path, "--test-context", "M") == 1
    out = capsys.readouterr().out
    assert "не сохранён" in out and "записано в" not in out


def test_oneshot_does_not_measure_unknown_window(tmp_path, oneshot, capsys, monkeypatch):
    """Замер оплачивается — в -p (в том числе в CI) он не делается, только подсказка."""
    provider = WindowProvider(8_192, [text_turn("ok")])
    oneshot(provider)
    assert _main(tmp_path, "-m", "M1", "-p", "x") == 0
    out = " ".join(capsys.readouterr().out.split())  # без переносов по ширине терминала
    assert provider.measured == []
    assert "окно контекста M1 не замерено — считаем 32k" in out
    assert "gigacrew --test-context M1" in out
    monkeypatch.setenv("GIGACREW_CONTEXT_TOKENS", "8000")  # явный бюджет — и подсказка не нужна
    oneshot(WindowProvider(8_192, [text_turn("ok")]))
    assert _main(tmp_path, "-m", "M2", "-p", "x") == 0
    assert "не замерено" not in capsys.readouterr().out


def _measure_env(tmp_path, *, auto_approve=False, window=16_384, **provider_kw):
    cfg = Config(access_key="x", project_root=tmp_path, stream=False, auto_approve=auto_approve)
    provider = WindowProvider(window, [text_turn("ответ")], **provider_kw)
    ui = Console(no_color=True)
    agent = Agent(provider, build_default_registry(), cfg, ui)
    ctx = CommandContext(agent=agent, ui=ui, commands=default_commands())
    return agent, ui, ctx, provider


def test_repl_asks_before_measuring(tmp_path, capsys, monkeypatch):
    agent, ui, ctx, provider = _measure_env(tmp_path)
    questions = []
    monkeypatch.setattr(ui, "interactive", lambda: True)
    monkeypatch.setattr(ui, "ask", lambda q, **kw: questions.append(q) or False)
    run_repl(agent, ui, ctx.commands, read_input=_reader("/model Beta-14B", "/exit"))
    out = _out(capsys)
    assert provider.measured == []
    assert len(questions) == 2  # при старте — для текущей модели, затем для Beta-14B
    assert "оплачиваются" in questions[0] and "пока считаем окно" in out
    # отказ запомнен: повторный выбор той же модели не спрашивает снова
    ctx.declined_windows.update({agent.model})
    commands = ctx.commands
    commands.dispatch(f"/model {agent.model}", ctx)
    assert len(questions) == 2
    monkeypatch.setattr(ui, "ask", lambda q, **kw: questions.append(q) or True)
    commands.dispatch("/model Beta-32B", ctx)
    assert {model for model, _ in provider.measured} == {"Beta-32B"}
    assert 16_000 <= agent.context_window <= 16_384


def test_repl_without_terminal_does_not_ask(tmp_path, capsys, monkeypatch):
    agent, ui, ctx, provider = _measure_env(tmp_path)
    monkeypatch.setattr(ui, "interactive", lambda: False)
    monkeypatch.setattr(ui, "ask", lambda q, **kw: pytest.fail("вопрос без терминала"))
    run_repl(agent, ui, ctx.commands, read_input=_reader("/exit"))
    assert provider.measured == [] and "--test-context" in _out(capsys)


def test_auto_measure_can_be_disabled(tmp_path, capsys, monkeypatch):
    from dataclasses import replace

    from gigacrew.cli.models import ensure_context_window

    agent, ui, ctx, provider = _measure_env(tmp_path, auto_approve=True)
    agent._cfg = replace(agent.config, auto_measure=False)
    ensure_context_window(agent, ui)
    assert provider.measured == [] and _out(capsys) == ""
    monkeypatch.setenv("GIGACREW_AUTO_MEASURE", "0")
    assert Config.load(project_root=tmp_path).auto_measure is False


def test_failed_measurement_falls_back(tmp_path, capsys, _isolated_home):
    from gigacrew.llm.base import LLMError

    agent, ui, ctx, provider = _measure_env(tmp_path, auto_approve=True, error=LLMError("сбой"))
    ctx.commands.dispatch("/model M1", ctx)
    out = _out(capsys)
    assert "не удалось замерить окно M1" in out and "пока считаем окно M1 равным 32k" in out
    assert _models_json(_isolated_home) == {}


def test_oneshot_exit_code_when_stopped_by_guard(tmp_path, oneshot):
    missing = tool_turn("read_file", {"path": "нет.txt"})
    oneshot(ScriptedProvider([missing] * 4 + [text_turn("не вышло")]))
    assert _main(tmp_path, "-p", "прочитай") == app.EXIT_STOPPED == 3


def test_oneshot_internal_error_is_reported(tmp_path, oneshot, capsys):
    oneshot(ScriptedProvider([RuntimeError("сломалось")]))
    assert _main(tmp_path, "-p", "вопрос") == 1
    out = capsys.readouterr().out
    assert "внутренняя ошибка: RuntimeError: сломалось" in out and "Traceback" not in out
    (saved,) = _store(tmp_path).recent()  # чат с запросом всё равно сохранён
    assert saved.title == "вопрос"


@pytest.fixture
def window_env(tmp_path, capsys):
    cfg = Config(
        access_key="x",
        project_root=tmp_path,
        stream=False,
        auto_approve=True,
        model="Alpha-3-Ultra",
    )
    provider = WindowProvider(16_384, [text_turn("ответ")])
    ui = Console(no_color=True)
    agent = Agent(provider, build_default_registry(), cfg, ui)
    commands = default_commands()
    ctx = CommandContext(
        agent=agent, ui=ui, commands=commands, chats=ChatRecorder(ChatStore(agent.workspace))
    )
    return agent, commands, ctx, provider


def test_model_command_measures_and_saves_chat(window_env, capsys):
    agent, commands, ctx, provider = window_env
    agent.run_turn("вопрос")
    commands.dispatch("/model Beta-14B", ctx)
    assert agent.model == "Beta-14B" and 16_000 <= agent.context_window <= 16_384
    assert {model for model, _ in provider.measured} == {"Beta-14B"}
    (chat,) = ctx.chats.store.recent()
    assert chat.model == "Beta-14B"  # смена модели сохранена без нового хода
    provider.measured.clear()
    commands.dispatch("/model Alpha-3-Ultra", ctx)
    commands.dispatch("/model Beta-14B", ctx)  # уже замерена
    assert {model for model, _ in provider.measured} == {"Alpha-3-Ultra"}


def test_model_command_lists_catalog(window_env, capsys, monkeypatch):
    from gigacrew.cli.models import ModelCatalog

    agent, commands, ctx, provider = window_env
    monkeypatch.setattr(provider, "list_models", lambda: ["Alpha-3-Ultra", "Beta-14B"])
    ctx.models = ModelCatalog(provider)
    ctx.models.start()
    ctx.models._thread.join(5)
    assert ctx.models.choices(agent.windows, agent.model) == [
        ("Alpha-3-Ultra", "текущая · окно не замерено"),
        ("Beta-14B", "окно не замерено"),
    ]
    commands.dispatch("/model", ctx)
    out = _out(capsys)
    assert "доступные модели" in out and "• Beta-14B — окно не замерено" in out
    commands.dispatch("/model Unknown", ctx)
    assert "нет в списке доступных" in _out(capsys)
    commands.dispatch("/model beta-14b", ctx)  # регистр в имени модели исправляется
    assert agent.model == "Beta-14B" and "нет в списке" not in _out(capsys)


def test_model_catalog_swallows_errors():
    from gigacrew.cli.models import ModelCatalog
    from gigacrew.llm.base import LLMError

    provider = ScriptedProvider()

    def boom():
        raise LLMError("сеть")

    provider.list_models = boom
    catalog = ModelCatalog(provider)
    assert catalog.models is None
    assert catalog.load() == [] and catalog.models == []


def test_resume_restores_chat_model(repl_env, capsys):
    agent, _, commands, ctx, provider = repl_env
    provider._turns = [text_turn("старый ответ"), text_turn("ещё")]
    agent.set_model("M-old")
    agent.run_turn("старый вопрос")
    ctx.chats.save(agent.conversation, model=agent.model)
    old_id = ctx.chats.chat_id
    commands.dispatch("/clear", ctx)
    commands.dispatch("/model M-new", ctx)
    commands.dispatch(f"/resume {old_id}", ctx)
    assert agent.model == "M-old" and "модель чата: M-old" in _out(capsys)
    agent.run_turn("ещё")
    assert provider.requests[-1]["model"] == "M-old"


def test_continue_restores_model_unless_overridden(tmp_path, oneshot):
    oneshot(ScriptedProvider([text_turn("a")]))
    assert _main(tmp_path, "-m", "M-chat", "-p", "первый") == 0
    provider = ScriptedProvider([text_turn("b")])
    oneshot(provider)
    assert _main(tmp_path, "-c", "-p", "второй") == 0
    assert provider.requests[-1]["model"] == "M-chat"
    provider = ScriptedProvider([text_turn("c")])
    oneshot(provider)
    assert _main(tmp_path, "-c", "-m", "M-cli", "-p", "третий") == 0
    assert provider.requests[-1]["model"] == "M-cli"
    (chat,) = _store(tmp_path).recent()
    assert chat.model == "M-cli"


# ----------------------------- режимы разрешений ----------------------------- #
def test_mode_command(cli_env, capsys):
    agent, _, commands, ctx, _ = cli_env
    assert commands.dispatch("/mode", ctx) is True
    out = _out(capsys)
    assert "режим: ручной" in out and "plan — план" in out and "Shift+Tab" in out
    commands.dispatch("/mode plan", ctx)
    assert agent.mode is PermissionMode.PLAN
    assert "режим: план" in _out(capsys)
    commands.dispatch("/mode EDITS", ctx)
    assert agent.mode is PermissionMode.ACCEPT_EDITS
    commands.dispatch("/mode auto", ctx)
    assert agent.mode is PermissionMode.ACCEPT_EDITS
    assert "неизвестный режим" in _out(capsys)


def test_repl_shows_mode_and_wires_shift_tab(cli_env, capsys):
    import contextlib

    from gigacrew.cli.repl import status_of

    agent, ui, commands, _, _ = cli_env

    class FakeEsc:
        enabled = True
        on_backtab = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def paused(self):
            return contextlib.nullcontext()

        def take_typeahead(self):
            return ""

    esc = FakeEsc()
    assert run_repl(agent, ui, commands, read_input=_reader("/exit"), interrupt=esc) == 0
    out = _out(capsys)
    assert "режим" in out and "ручной" in out and "Shift+Tab" in out  # баннер
    # Shift+Tab во время хода — смена режима агента и подсказка в индикаторе
    esc.on_backtab()
    assert agent.mode is PermissionMode.ACCEPT_EDITS
    assert status_of(agent).mode is PermissionMode.ACCEPT_EDITS
    assert "авто-правки (Shift+Tab)" in ui._turn_hint()


def test_oneshot_mode_flag_and_env(tmp_path, oneshot, monkeypatch):
    provider = ScriptedProvider([text_turn("план"), text_turn("план")])
    oneshot(provider)
    assert _main(tmp_path, "--mode", "plan", "-p", "x", "--no-save") == 0
    monkeypatch.setenv("GIGACREW_MODE", "plan")
    assert _main(tmp_path, "-p", "x", "--no-save") == 0
    assert all("# Plan mode" in r["messages"][0].content for r in provider.requests)
    with pytest.raises(SystemExit):
        _main(tmp_path, "--mode", "auto", "-p", "x")


def test_oneshot_plan_without_approver_ends_with_plan(tmp_path, oneshot, capsys, monkeypatch):
    monkeypatch.setattr(ChatStore, "new_id", staticmethod(lambda now=None: "chat1"))
    plan = ".gigacrew/plans/chat1.md"
    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": plan, "content": "1. поправить a.py"}),
            tool_turn("exit_plan_mode", {}),
            text_turn("План: 1. поправить a.py"),
        ]
    )
    oneshot(provider)
    assert _main(tmp_path, "--mode", "plan", "-p", "спланируй", "--no-save") == 0
    assert plan in provider.requests[0]["messages"][0].content  # путь — в правилах режима
    function_msgs = [m for m in provider.requests[-1]["messages"] if m.role == "function"]
    assert "Одобрить план некому" in function_msgs[-1].content and plan in function_msgs[-1].content
    assert (tmp_path / plan).read_text(encoding="utf-8") == "1. поправить a.py"
    assert "План: 1. поправить a.py" in capsys.readouterr().out


def test_oneshot_plan_left_in_text_ends_without_extra_requests(
    tmp_path, oneshot, capsys, monkeypatch
):
    monkeypatch.setattr(ChatStore, "new_id", staticmethod(lambda now=None: "chat1"))
    plan = ".gigacrew/plans/chat1.md"
    provider = ScriptedProvider(
        [
            tool_turn("write_file", {"path": plan, "content": "1. поправить a.py"}),
            text_turn("План: 1. поправить a.py"),  # без exit_plan_mode
            text_turn("лишний запрос"),
        ]
    )
    oneshot(provider)
    assert _main(tmp_path, "--mode", "plan", "-p", "спланируй", "--no-save") == 0
    assert len(provider.requests) == 2  # ни напоминания, ни самопроверки
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "План: 1. поправить a.py" in output and plan in output  # где лежит план


# ------------------------ провайдер с каталогом моделей ------------------------ #
def _flat(capsys) -> str:
    return " ".join(capsys.readouterr().out.split())  # без переносов по ширине терминала


def test_make_provider_follows_the_setting(tmp_path):
    from gigacrew.llm.api import ApiProvider
    from gigacrew.llm.gigacode import GigaCodeProvider

    ui = Console(no_color=True)
    api = app._make_provider(
        Config(access_key="k", auth_url="https://auth.test", base_url="https://api.test/v1"), ui
    )
    giga = app._make_provider(
        Config(provider="gigacode", base_url="https://code.test", plugin_name="demo"), ui
    )
    assert isinstance(api, ApiProvider) and isinstance(giga, GigaCodeProvider)
    api.close()
    giga.close()


def test_gigacode_without_client_name_stops_before_any_request(
    tmp_path, oneshot, monkeypatch, capsys
):
    monkeypatch.setenv("GIGACREW_PROVIDER", "gigacode")
    monkeypatch.setattr(app, "_make_provider", lambda *_a: pytest.fail("провайдер не нужен"))
    assert _main(tmp_path, "-p", "x") == 1
    assert "GIGACREW_PLUGIN_NAME" in _flat(capsys)


def test_catalogue_gives_the_default_model_and_its_window(tmp_path, oneshot, monkeypatch, capsys):
    monkeypatch.delenv("GIGACREW_MODEL")
    provider = CatalogueProvider([text_turn("ok")])
    oneshot(provider)
    assert _main(tmp_path, "-p", "x") == 0
    assert provider.connects == 1 and provider.requests[-1]["model"] == "cat-a"
    assert "не замерено" not in _flat(capsys)  # окно — из каталога, замер не нужен
    assert provider.measured == []


def test_model_outside_the_catalogue_is_refused(tmp_path, oneshot, capsys):
    provider = CatalogueProvider([text_turn("ok")])
    oneshot(provider)
    assert _main(tmp_path, "-p", "x") == 1  # GIGACREW_MODEL=M-default из фикстуры
    out = _flat(capsys)
    assert "модели M-default нет в каталоге бэкенда; доступны: cat-a, cat-b" in out
    assert "--list-models" in out and provider.requests == []
    assert _main(tmp_path, "--list-models") == 0  # список работает и с чужой моделью
    assert "cat-b" in _flat(capsys)
    assert _main(tmp_path, "-m", "cat-b", "-p", "x") == 0
    assert provider.requests[-1]["model"] == "cat-b"
    assert _main(tmp_path, "--test-context", "чужая") == 1
    assert "модели чужая нет в каталоге" in _flat(capsys)


def test_connect_interrupted_or_failed(tmp_path, oneshot, capsys):
    from gigacrew.llm.base import LLMError

    oneshot(CatalogueProvider(connect_errors=[KeyboardInterrupt()]))
    assert _main(tmp_path, "-m", "cat-a", "-p", "x") == 130
    assert "вход прерван" in _flat(capsys)
    oneshot(CatalogueProvider(connect_errors=[LLMError("бэкенд недоступен")]))
    assert _main(tmp_path, "-m", "cat-a", "--list-models") == 2
    assert "не удалось подключиться: бэкенд недоступен" in _flat(capsys)


def test_continue_with_a_model_missing_from_the_catalogue(tmp_path, oneshot, monkeypatch, capsys):
    oneshot(ScriptedProvider([text_turn("a")]))
    assert _main(tmp_path, "-m", "old-model", "-p", "первый") == 0
    monkeypatch.delenv("GIGACREW_MODEL")
    provider = CatalogueProvider([text_turn("b")])
    oneshot(provider)
    assert _main(tmp_path, "-c", "-p", "второй") == 0
    assert "модели чата old-model нет в каталоге бэкенда — модель: cat-a" in _flat(capsys)
    assert provider.requests[-1]["model"] == "cat-a"


def _catalogue_env(tmp_path, provider, **cfg):
    config = Config(access_key="x", project_root=tmp_path, stream=False, model="cat-a", **cfg)
    ui = Console(no_color=True)
    agent = Agent(provider, build_default_registry(), config, ui)
    ctx = CommandContext(
        agent=agent,
        ui=ui,
        commands=default_commands(),
        chats=ChatRecorder(ChatStore(agent.workspace)),
    )
    return agent, ui, ctx


def test_model_command_refuses_models_outside_the_catalogue(tmp_path, capsys):
    agent, _, ctx = _catalogue_env(tmp_path, CatalogueProvider())
    ctx.commands.dispatch("/model чужая", ctx)
    assert agent.model == "cat-a"
    assert "модели чужая нет в каталоге бэкенда; доступны: cat-a, cat-b" in _flat(capsys)
    ctx.commands.dispatch("/model CAT-B", ctx)  # регистр исправляется
    assert agent.model == "cat-b"
    ctx.commands.dispatch("/model", ctx)
    out = _flat(capsys)
    assert "доступные модели" in out and "• cat-a" in out


def test_resume_command_keeps_the_model_missing_from_the_catalogue(tmp_path, capsys):
    agent, _, ctx = _catalogue_env(tmp_path, CatalogueProvider([text_turn("ответ")]))
    agent.conversation.add_user("старый вопрос")
    info = ctx.chats.store.save("20260101-000000-abcd", agent.conversation, model="old-model")
    ctx.commands.dispatch(f"/resume {info.id}", ctx)
    assert agent.model == "cat-a"
    assert "модели чата old-model нет в каталоге бэкенда — модель: cat-a" in _flat(capsys)


def test_repl_connects_before_each_turn(tmp_path, capsys):
    from gigacrew.llm.base import LLMError

    provider = CatalogueProvider(
        [text_turn("ответ")], connect_errors=[LLMError("вход не подтверждён")]
    )
    agent, ui, ctx = _catalogue_env(tmp_path, provider, auto_approve=True)
    read = _reader("раз", "два", "/exit")
    assert run_repl(agent, ui, ctx.commands, read_input=read, index_on_start=False) == 0
    assert provider.connects == 2
    assert "вход не подтверждён" in _flat(capsys)
    assert len(provider.requests) == 1  # первый запрос не ушёл: вход не удался


def test_repl_warns_about_tls_for_the_backend_token(tmp_path, capsys):
    from dataclasses import replace

    agent, ui, ctx = _catalogue_env(tmp_path, CatalogueProvider(), auto_approve=True)
    agent._cfg = replace(agent.config, provider="gigacode", access_key=None)
    run_repl(agent, ui, ctx.commands, read_input=_reader("/exit"), index_on_start=False)
    assert "ключ или токен можно перехватить" in _flat(capsys)


def test_browser_is_opened_only_with_a_display():
    from gigacrew.cli.login import can_open_browser

    assert can_open_browser({"DISPLAY": ":0"}, "linux")
    assert can_open_browser({"WAYLAND_DISPLAY": "wayland-0"}, "linux")
    assert not can_open_browser({}, "linux")
    assert not can_open_browser({"DISPLAY": ":0", "SSH_CONNECTION": "1 2 3 4"}, "linux")
    assert can_open_browser({}, "darwin") and can_open_browser({}, "win32")
    assert not can_open_browser({"SSH_TTY": "/dev/pts/1"}, "darwin")


def test_open_in_browser_runs_xdg_open(monkeypatch):
    import subprocess
    import sys

    from gigacrew.cli import login

    calls = []
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(login, "can_open_browser", lambda: True)
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: calls.append((cmd, kw)))
    assert login.open_in_browser("https://x.test/a") is True
    ((cmd, kw),) = calls
    assert cmd == ["xdg-open", "https://x.test/a"]
    assert kw["start_new_session"] and kw["stdout"] is subprocess.DEVNULL

    def missing(*_a, **_kw):
        raise OSError("нет xdg-open")

    monkeypatch.setattr(subprocess, "Popen", missing)
    assert login.open_in_browser("https://x.test/a") is False
    monkeypatch.setattr(login, "can_open_browser", lambda: False)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("без GUI не запускаем"))
    assert login.open_in_browser("https://x.test/a") is False


def test_login_notice_prints_the_whole_link(capsys):
    from gigacrew.cli.login import login_notifier

    ui = Console(no_color=True, width=40)
    url = "https://gitverse.test/signin?device=" + "a" * 64
    login_notifier(ui, opener=lambda _u: False)(url)
    out = capsys.readouterr().out
    assert f"  {url}\n" in out  # одной строкой, без переносов по ширине терминала
    text = " ".join(out.split())
    assert "откройте ссылку" in text and "Ctrl+C — отмена" in text
    login_notifier(ui, opener=lambda _u: True)(url)
    assert "открываю страницу входа" in _flat(capsys)
