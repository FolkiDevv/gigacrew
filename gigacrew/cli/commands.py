"""Слеш-команды REPL (/help, /model, ...).

Команды регистрируются в :class:`CommandRegistry`; из него же строятся справка,
подсказки в баннере и автодополнение ввода (:mod:`gigacrew.cli.prompt`). Новая команда
— это один :class:`SlashCommand`.
"""

from __future__ import annotations

import contextlib
import difflib
import re
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field

from gigacrew.agent.chat_store import ChatRecorder, ChatStoreError, SavedChat
from gigacrew.agent.loop import Agent
from gigacrew.cli.indexing import describe_index, run_indexing
from gigacrew.cli.models import ModelCatalog, describe_model, ensure_context_window
from gigacrew.cli.prompt import KEY_HELP
from gigacrew.llm.base import LLMError
from gigacrew.permissions import MODE_CYCLE, PermissionMode, parse_mode
from gigacrew.project import semantic
from gigacrew.project.index import ProjectIndex
from gigacrew.ui.console import Console


@dataclass
class CommandContext:
    agent: Agent
    ui: Console
    commands: CommandRegistry
    chats: ChatRecorder | None = None  # None — чаты не сохраняются (тесты)
    # Прерывание долгих команд клавишей Esc (как хода агента); None — только Ctrl+C.
    interrupt: AbstractContextManager[object] | None = None
    models: ModelCatalog | None = None  # список моделей для /model (None — не загружался)
    # Модели, замер окна которых пользователь отклонил в этой сессии (не спрашивать снова).
    declined_windows: set[str] = field(default_factory=set)


# Обработчик получает контекст и аргумент (текст после имени команды).
# Возвращает False, если REPL нужно завершить.
Handler = Callable[[CommandContext, str], bool]


@dataclass(frozen=True)
class SlashCommand:
    name: str  # с ведущим слешем: "/model"
    summary: str  # кратко — для баннера и справки
    handler: Handler
    usage: str = ""  # синтаксис для справки: "/model [имя]"
    aliases: tuple[str, ...] = ()


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        self._lookup: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        for key in (command.name, *command.aliases):
            key = key.lower()
            if key in self._lookup:
                raise ValueError(f"Команда {key} уже зарегистрирована")
            self._lookup[key] = command
        self._commands[command.name] = command

    def get(self, name: str) -> SlashCommand | None:
        return self._lookup.get(name.lower())

    def __iter__(self) -> Iterator[SlashCommand]:
        return iter(self._commands.values())

    def dispatch(self, line: str, ctx: CommandContext) -> bool:
        """Выполняет команду из строки. False — завершить REPL."""
        parts = line.strip().split(maxsplit=1)
        name = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""
        command = self.get(name)
        if command is None:
            names = [n for cmd in self for n in (cmd.name, *cmd.aliases)]
            close = difflib.get_close_matches(name.lower(), names, n=1, cutoff=0.6)
            hint = f" — возможно, {close[0]}?" if close else ""
            ctx.ui.error(f"неизвестная команда: {name}{hint} (список — /help)")
            return True
        return command.handler(ctx, arg)


# Команда REPL — это слеш + слово (/help, /model ...). Чистый путь
# (/home/...) или начало регэкспа содержит дополнительные слеши и НЕ считается
# командой, а уходит обычным запросом агенту.
_REPL_COMMAND_RE = re.compile(r"^/[a-zA-Zа-яА-Я][\w-]*$")


def is_repl_command(line: str) -> bool:
    if not line.startswith("/"):
        return False
    first = line.split(maxsplit=1)[0]
    return bool(_REPL_COMMAND_RE.match(first))


# ------------------------------- команды ------------------------------- #
def _help(ctx: CommandContext, _arg: str) -> bool:
    rows = []
    for cmd in ctx.commands:
        alias = f" (также {', '.join(cmd.aliases)})" if cmd.aliases else ""
        rows.append((cmd.usage or cmd.name, f"{cmd.summary}{alias}"))
    ctx.ui.help(rows, KEY_HELP)
    return True


def _model(ctx: CommandContext, arg: str) -> bool:
    agent = ctx.agent
    known = ctx.models.models if ctx.models is not None else None
    if not arg:
        lines = [f"текущая модель: {agent.model} ({describe_model(agent.model, agent.windows)})"]
        if known:
            lines.append("доступные модели:")
            lines += [f"  • {name} — {describe_model(name, agent.windows)}" for name in known]
        ctx.ui.info("\n".join(lines))
        return True
    if known and arg not in known:
        # имена моделей в API регистрозависимы: «model-large» → «Model-Large»
        arg = next((name for name in known if name.lower() == arg.lower()), arg)
    agent.set_model(arg)
    if known and arg not in known:
        ctx.ui.warn(f"модели {arg} нет в списке доступных чат-моделей")
    ctx.ui.info(f"модель теперь: {arg} (применится со следующего запроса)")
    ensure_context_window(agent, ctx.ui, interrupt=ctx.interrupt, declined=ctx.declined_windows)
    if ctx.chats is not None:  # модель — часть чата: сохраняем, не дожидаясь хода
        warning = ctx.chats.save(agent.conversation, model=agent.model)
        if warning:
            ctx.ui.warn(warning)
    return True


def mode_choices(current: PermissionMode) -> list[tuple[str, str]]:
    """Варианты аргумента ``/mode`` для автодополнения: (имя, пояснение)."""
    return [
        (mode.value, f"{mode.label}{' · текущий' if mode is current else ''} — {mode.description}")
        for mode in MODE_CYCLE
    ]


def _mode(ctx: CommandContext, arg: str) -> bool:
    agent = ctx.agent
    if not arg:
        lines = [f"режим: {agent.mode.label} — {agent.mode.description}"]
        lines.append("режимы (Shift+Tab — следующий по кругу):")
        lines += [f"  • {m.value} — {m.label}: {m.description}" for m in MODE_CYCLE]
        ctx.ui.info("\n".join(lines))
        return True
    try:
        mode = parse_mode(arg)
    except ValueError as e:
        ctx.ui.error(f"{e} (использование: /mode [manual|edits|plan])")
        return True
    agent.set_mode(mode)
    ctx.ui.info(f"режим: {mode.label} — {mode.description}")
    return True


def _clear(ctx: CommandContext, _arg: str) -> bool:
    if ctx.chats is None:
        ctx.agent.reset()
    else:
        ctx.chats.start_new()  # прежний чат остаётся сохранённым — его вернёт /resume
        ctx.agent.reset(chat_id=ctx.chats.chat_id)
    ctx.ui.info("история очищена, начат новый чат")
    return True


def _compact(ctx: CommandContext, arg: str) -> bool:
    agent = ctx.agent
    guard = ctx.interrupt if ctx.interrupt is not None else contextlib.nullcontext()
    try:
        with guard:
            result = agent.compact(arg)
    except KeyboardInterrupt:
        ctx.ui.warn("сжатие прервано — история не изменилась")
        return True
    except LLMError as e:
        ctx.ui.error(f"не удалось сжать контекст: {e}")
        return True
    if result is None:
        ctx.ui.info("сжимать нечего: история пуста или уже сжата")
        return True
    if ctx.chats is not None:  # краткое содержание — часть чата: сохраняем сразу
        warning = ctx.chats.save(agent.conversation, model=agent.model)
        if warning:
            ctx.ui.warn(warning)
    return True


RESUME_LIMIT = 100  # сколько последних чатов показывает селектор


def resume_chat(
    agent: Agent, chats: ChatRecorder, saved: SavedChat, *, keep_model: bool = False
) -> None:
    """Продолжить сохранённый чат: дальнейшие ходы дописываются в него же.

    Модель чата становится текущей, если не ``keep_model`` (модель задана явно, ``-m``).
    Файл плана — тоже этого чата.
    """
    agent.reset(saved.conversation, chat_id=saved.info.id)
    if saved.info.model and not keep_model:
        agent.set_model(saved.info.model)
    chats.switch_to(saved.info)


def _resume(ctx: CommandContext, arg: str) -> bool:
    if ctx.chats is None:
        ctx.ui.warn("сохранённые чаты недоступны")
        return True
    store = ctx.chats.store
    try:
        if arg:
            info = store.find(arg)
        else:
            recent = store.recent(limit=RESUME_LIMIT)
            if not recent:
                ctx.ui.info("сохранённых чатов пока нет")
                return True
            info = ctx.ui.pick_chat(recent, ctx.chats.chat_id)
            if info is None:
                return True
        saved = store.load(info.id)
    except ChatStoreError as e:
        ctx.ui.error(str(e))
        return True
    previous = ctx.agent.model
    resume_chat(ctx.agent, ctx.chats, saved)
    ctx.ui.chat_resumed(saved.info, saved.conversation.messages)
    if ctx.agent.model != previous:
        ctx.ui.info(f"модель чата: {ctx.agent.model}")
        ensure_context_window(
            ctx.agent, ctx.ui, interrupt=ctx.interrupt, declined=ctx.declined_windows
        )
    return True


def _index(ctx: CommandContext, arg: str) -> bool:
    if arg not in ("", "rebuild"):
        ctx.ui.error("использование: /index [rebuild]")
        return True
    workspace = ctx.agent.workspace
    index = ProjectIndex(workspace)
    result = run_indexing(index, ctx.ui, ctx.interrupt, rebuild=arg == "rebuild")
    if result is not None:
        ty = semantic.status(workspace) if ctx.agent.config.ty else "выключен (GIGACREW_TY=0)"
        ctx.ui.info(describe_index(*result, ty_status=ty))
    return True


def _exit(ctx: CommandContext, _arg: str) -> bool:
    ctx.ui.system("до встречи!")
    return False


def default_commands() -> CommandRegistry:
    registry = CommandRegistry()
    registry.register(SlashCommand("/help", "справка", _help))
    registry.register(SlashCommand("/model", "сменить модель", _model, usage="/model [имя]"))
    registry.register(
        SlashCommand(
            "/mode", "режим: ручной, авто-правки, план", _mode, usage="/mode [manual|edits|plan]"
        )
    )
    registry.register(SlashCommand("/clear", "новый чат", _clear))
    registry.register(
        SlashCommand(
            "/compact",
            "сжать историю в краткое содержание",
            _compact,
            usage="/compact [пожелания]",
        )
    )
    registry.register(
        SlashCommand(
            "/resume",
            "открыть сохранённый чат",
            _resume,
            usage="/resume [id]",
            aliases=("/chats",),
        )
    )
    registry.register(
        SlashCommand("/index", "обновить индекс проекта", _index, usage="/index [rebuild]")
    )
    registry.register(SlashCommand("/exit", "выход", _exit, aliases=("/quit", "/q")))
    return registry
