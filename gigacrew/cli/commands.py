"""Слеш-команды REPL (/help, /model, ...).

Команды регистрируются в :class:`CommandRegistry`; из него же строятся справка,
подсказки в баннере и автодополнение ввода (:mod:`gigacrew.cli.prompt`). Новая команда
— это один :class:`SlashCommand`.

Скиллы (:mod:`gigacrew.project.skills`) тоже вызываются как команды — ``/<имя>
[аргументы]``, но не выполняются здесь: :func:`expand_skill` превращает строку в
запрос агенту. Встроенная команда важнее скилла с тем же именем.
"""

from __future__ import annotations

import contextlib
import difflib
import re
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field

from gigacrew.agent.chat_store import ChatRecorder, ChatStoreError, SavedChat
from gigacrew.agent.decisions import redact
from gigacrew.agent.loop import Agent
from gigacrew.agent.prompts import skill_invocation_prompt
from gigacrew.cli.hooks import (
    describe_hooks,
    install_request_hook,
    load_hooks,
    restart_session,
)
from gigacrew.cli.indexing import describe_index, run_indexing
from gigacrew.cli.models import ModelCatalog, describe_model, ensure_context_window
from gigacrew.cli.prompt import KEY_HELP
from gigacrew.llm.base import LLMError, LLMProvider
from gigacrew.permissions import MODE_CYCLE, PermissionMode, parse_mode
from gigacrew.project import semantic
from gigacrew.project.index import ProjectIndex
from gigacrew.project.skills import COMMAND_NAME_PATTERN, SkillCatalog
from gigacrew.ui.console import Console


def save_chat(chats: ChatRecorder, agent: Agent, ui: Console) -> None:
    """Сохранить чат. Если файл чата изменила другая сессия, диалог сохраняется новым
    чатом — тогда и план этой сессии переезжает к нему (:meth:`Agent.continue_as`).
    """
    warning = chats.save(agent.conversation, model=agent.model)
    if chats.chat_id != agent.chat_id:
        agent.continue_as(chats.chat_id)
    if warning:
        ui.warn(warning)


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
    skills: SkillCatalog | None = None  # для /help и подсказок (None — без скиллов)
    trust_hooks: bool = False  # --trust-hooks: /hooks reload не спрашивает подтверждение


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
            if ctx.skills is not None:
                names += [skill.command for skill in ctx.skills]
            close = difflib.get_close_matches(name.lower(), names, n=1, cutoff=0.6)
            hint = f" — возможно, {close[0]}?" if close else ""
            ctx.ui.error(f"неизвестная команда: {name}{hint} (список — /help)")
            return True
        return command.handler(ctx, arg)


# Команда REPL — это слеш + слово (/help, /model ...). Чистый путь
# (/home/...) или начало регэкспа содержит дополнительные слеши и НЕ считается
# командой, а уходит обычным запросом агенту.
_REPL_COMMAND_RE = re.compile(rf"^/{COMMAND_NAME_PATTERN}$")


def is_repl_command(line: str) -> bool:
    if not line.startswith("/"):
        return False
    first = line.split(maxsplit=1)[0]
    return bool(_REPL_COMMAND_RE.match(first))


def expand_skill(line: str, commands: CommandRegistry, skills: SkillCatalog) -> str | None:
    """Запрос агенту для строки ``/<скилл> [аргументы]``.

    None — это не скилл (обычный текст, встроенная команда или неизвестное имя).
    Скилл, который не удалось прочитать, — :class:`OSError` (текст для пользователя).
    """
    if not is_repl_command(line):
        return None
    parts = line.strip().split(maxsplit=1)
    if commands.get(parts[0]) is not None:
        return None
    skill = skills.get(parts[0])
    if skill is None:
        return None
    try:
        body = skill.body()
    except (OSError, UnicodeDecodeError) as e:
        raise OSError(f"не удалось прочитать скилл {skill.name} ({skill.path}): {e}") from None
    args = parts[1].strip() if len(parts) > 1 else ""
    return skill_invocation_prompt(skill, body, line.strip(), args)


# ------------------------------- команды ------------------------------- #
def _help(ctx: CommandContext, _arg: str) -> bool:
    rows = []
    for cmd in ctx.commands:
        alias = f" (также {', '.join(cmd.aliases)})" if cmd.aliases else ""
        rows.append((cmd.usage or cmd.name, f"{cmd.summary}{alias}"))
    if ctx.skills is not None:
        rows += [
            (f"{skill.command} [аргументы]", f"скилл · {skill.description or skill.path}")
            for skill in ctx.skills
            if ctx.commands.get(skill.command) is None
        ]
    ctx.ui.help(rows, KEY_HELP)
    return True


def catalogue_ids(provider: LLMProvider) -> list[str] | None:
    """Модели каталога провайдера — выбрать можно только их; None — каталога нет."""
    models = provider.known_models()
    return None if models is None else [m.id for m in models]


def _model(ctx: CommandContext, arg: str) -> bool:
    agent = ctx.agent
    catalogue = catalogue_ids(agent.provider)
    known = catalogue if catalogue is not None else ctx.models.models if ctx.models else None
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
    if catalogue is not None and arg not in catalogue:
        ctx.ui.error(f"модели {arg} нет в каталоге бэкенда; доступны: {', '.join(catalogue)}")
        return True
    agent.set_model(arg)
    if known and arg not in known:
        ctx.ui.warn(f"модели {arg} нет в списке доступных чат-моделей")
    ctx.ui.info(f"модель теперь: {arg} (применится со следующего запроса)")
    ensure_context_window(agent, ctx.ui, interrupt=ctx.interrupt, declined=ctx.declined_windows)
    if ctx.chats is not None:  # модель — часть чата: сохраняем, не дожидаясь хода
        save_chat(ctx.chats, agent, ctx.ui)
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
        if agent.mode is PermissionMode.AUTO and agent.reviewer.enabled:
            lines.append(f"  {agent.reviewer.describe(agent.model)}")
        lines.append("режимы (Shift+Tab — следующий по кругу):")
        lines += [f"  • {m.value} — {m.label}: {m.description}" for m in MODE_CYCLE]
        ctx.ui.info("\n".join(lines))
        _warn_reviewer(ctx)
        return True
    try:
        mode = parse_mode(arg)
    except ValueError as e:
        ctx.ui.error(f"{e} (использование: {MODE_USAGE})")
        return True
    agent.set_mode(mode)
    text = f"режим: {mode.label} — {mode.description}"
    if mode is PermissionMode.AUTO and agent.reviewer.enabled:
        text += f"\n  {agent.reviewer.describe(agent.model)}"
    ctx.ui.info(text)
    _warn_reviewer(ctx)
    return True


def _warn_reviewer(ctx: CommandContext) -> None:
    """В режиме «авто» с отключённым рецензентом — напомнить, что команды спрашиваются."""
    agent = ctx.agent
    if agent.mode is PermissionMode.AUTO and not agent.reviewer.enabled:
        ctx.ui.warn(agent.reviewer.warning)


PERMISSIONS_USAGE = "/permissions [forget N | clear | log]"
MODE_USAGE = "/mode [" + "|".join(mode.value for mode in MODE_CYCLE) + "]"


def _permissions(ctx: CommandContext, arg: str) -> bool:
    """Сохранённые разрешения «всегда» для команд и последние решения."""
    memory = ctx.agent.decisions
    words = arg.split()
    action = words[0].lower() if words else ""
    if action in {"forget", "rm", "удалить"}:
        # isdecimal, а не isdigit: «²» — цифра, но int() её не разберёт
        if len(words) != 2 or not words[1].isdecimal():
            ctx.ui.error(f"использование: {PERMISSIONS_USAGE}")
            return True
        try:
            rule = memory.forget(int(words[1]))
        except IndexError:
            ctx.ui.error(f"нет разрешения №{words[1]} (список — /permissions)")
        except OSError as e:
            ctx.ui.error(f"не удалось сохранить: {e}")
        else:
            ctx.ui.info(f"разрешение удалено: {redact(rule.describe())}")
        return True
    if action in {"clear", "очистить"}:
        try:
            n = memory.clear()
        except OSError as e:
            ctx.ui.error(f"не удалось сохранить: {e}")
        else:
            ctx.ui.info(f"удалено разрешений: {n}")
        return True
    if action in {"log", "журнал"}:
        items = memory.recent()
        if not items:
            ctx.ui.info("журнал решений пуст" + ("" if memory.log else " (не ведётся: --no-save)"))
            return True
        lines = []
        for item in items:
            mark = {"yes": "да", "always": "всегда", "no": "нет", "no_answer": "без ответа"}.get(
                str(item.get("decision")),
                str(item.get("decision")),
            )
            review = item.get("review") if isinstance(item.get("review"), dict) else {}
            if item.get("source") == "reviewer":
                mark = "разрешил рецензент"
            where = "" if item.get("cwd") in {None, "."} else f" [{item.get('cwd')}]"
            text = f"  • {mark}{where}: {item.get('command', '')}"
            if item.get("feedback"):
                text += f" — «{item['feedback']}»"
            elif item.get("source") == "reviewer" and review.get("reason"):
                text += f" — {review['reason']}"
            # Журнал прежних версий вырезал меньше форм секретов, пояснение рецензента
            # не вырезается вовсе — на экран не выводим ни то, ни другое.
            text = redact(text)
            lines.append(text)
        ctx.ui.info("последние решения:\n" + "\n".join(lines))
        return True
    if action:
        ctx.ui.error(f"использование: {PERMISSIONS_USAGE}")
        return True
    if not memory.rules:
        lines = ["сохранённых разрешений нет: ответ «a» на вопрос о команде запомнит её"]
    else:
        lines = ["разрешено без вопросов:"]
        for i, rule in enumerate(memory.rules, 1):
            used = f", применялось {rule.hits}" if rule.hits else ""
            # Правило — команда как есть: прежние версии сохраняли и команды с паролем.
            lines.append(f"  {i}. {redact(rule.describe())}{used}")
        lines.append("удалить — /permissions forget N, все — /permissions clear")
    if memory.approvals_path is not None and memory.remember:
        lines.append(f"файл: {memory.approvals_path}")
    elif memory.directory is None or not memory.remember:
        lines.append("разрешения хранятся до конца сессии (GIGACREW_REMEMBER_APPROVALS=0)")
    ctx.ui.info("\n".join(lines))
    return True


def _clear(ctx: CommandContext, _arg: str) -> bool:
    def switch() -> None:
        if ctx.chats is None:
            ctx.agent.reset()
        else:
            ctx.chats.start_new()  # прежний чат остаётся сохранённым — его вернёт /resume
            ctx.agent.reset(chat_id=ctx.chats.chat_id)

    restart_session(ctx.agent, ctx.ui, end_reason="clear", source="clear", switch=switch)
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
        save_chat(ctx.chats, agent, ctx.ui)
    return True


RESUME_LIMIT = 100  # сколько последних чатов показывает селектор


def resume_chat(
    agent: Agent,
    chats: ChatRecorder,
    saved: SavedChat,
    *,
    keep_model: bool = False,
) -> str | None:
    """Продолжить сохранённый чат: дальнейшие ходы дописываются в него же.

    Модель чата становится текущей, если не ``keep_model`` (модель задана явно, ``-m``)
    и если она есть в каталоге провайдера; иначе модель не меняется, а модель чата
    возвращается — вызывающий предупреждает. Файл плана — тоже этого чата.
    """
    skipped = None
    # Сначала модель: сохранённый расход токенов чата посчитан в её токенах.
    model = saved.info.model
    if model and not keep_model:
        catalogue = catalogue_ids(agent.provider)
        if catalogue is None or model in catalogue:
            agent.set_model(model)
        else:
            skipped = model
    agent.reset(saved.conversation, chat_id=saved.info.id)
    chats.switch_to(saved.info)
    return skipped


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
    chats = ctx.chats
    skipped = restart_session(
        ctx.agent,
        ctx.ui,
        end_reason="resume",
        source="resume",
        switch=lambda: resume_chat(ctx.agent, chats, saved),
    )
    ctx.ui.chat_resumed(saved.info, saved.conversation.messages, ctx.agent.call_info)
    if skipped:
        ctx.ui.warn(f"модели чата {skipped} нет в каталоге бэкенда — модель: {ctx.agent.model}")
    if ctx.agent.model != previous:
        ctx.ui.info(f"модель чата: {ctx.agent.model}")
        ensure_context_window(
            ctx.agent,
            ctx.ui,
            interrupt=ctx.interrupt,
            declined=ctx.declined_windows,
        )
    return True


def _index(ctx: CommandContext, arg: str) -> bool:
    if arg not in {"", "rebuild"}:
        ctx.ui.error("использование: /index [rebuild]")
        return True
    workspace = ctx.agent.workspace
    index = ProjectIndex(workspace)
    result = run_indexing(index, ctx.ui, ctx.interrupt, rebuild=arg == "rebuild")
    if result is not None:
        ty = semantic.status(workspace) if ctx.agent.config.ty else "выключен (GIGACREW_TY=0)"
        ctx.ui.info(describe_index(*result, ty_status=ty))
    return True


def _hooks(ctx: CommandContext, arg: str) -> bool:
    if arg not in {"", "reload"}:
        ctx.ui.error("использование: /hooks [reload]")
        return True
    if arg == "reload":
        agent = ctx.agent
        agent.hooks = load_hooks(
            agent.workspace,
            ctx.ui,
            interactive=ctx.ui.interactive(),
            trust=ctx.trust_hooks,
        )
        install_request_hook(agent.provider, agent.hooks, ctx.ui, agent.config)
        ctx.ui.info("настройки хуков перечитаны")
    ctx.ui.info(describe_hooks(ctx.agent.hooks.settings))
    return True


def _exit(ctx: CommandContext, _arg: str) -> bool:
    ctx.ui.system("до встречи!")
    return False


def default_commands() -> CommandRegistry:
    registry = CommandRegistry()
    registry.register(SlashCommand("/help", "справка", _help))
    registry.register(SlashCommand("/model", "сменить модель", _model, usage="/model [имя]"))
    registry.register(SlashCommand("/mode", "режим: авто или план", _mode, usage=MODE_USAGE))
    registry.register(
        SlashCommand(
            "/permissions",
            "разрешённые без вопросов команды и журнал решений",
            _permissions,
            usage=PERMISSIONS_USAGE,
        ),
    )
    registry.register(SlashCommand("/clear", "новый чат", _clear))
    registry.register(
        SlashCommand(
            "/compact",
            "сжать историю в краткое содержание",
            _compact,
            usage="/compact [пожелания]",
        ),
    )
    registry.register(
        SlashCommand(
            "/resume",
            "открыть сохранённый чат",
            _resume,
            usage="/resume [id]",
            aliases=("/chats",),
        ),
    )
    registry.register(
        SlashCommand("/index", "обновить индекс проекта", _index, usage="/index [rebuild]"),
    )
    registry.register(SlashCommand("/hooks", "хуки пользователя", _hooks, usage="/hooks [reload]"))
    registry.register(SlashCommand("/exit", "выход", _exit, aliases=("/quit", "/q")))
    return registry
