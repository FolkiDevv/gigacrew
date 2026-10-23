"""Слеш-команды REPL (/help, /model, ...).

Команды регистрируются в :class:`CommandRegistry`; из него же строятся справка,
подсказки в баннере и автодополнение ввода (:mod:`gigacrew.cli.prompt`). Новая команда
— это один :class:`SlashCommand`.

Скиллы (:mod:`gigacrew.project.skills`) тоже вызываются как команды — ``/<имя>
[аргументы]``, но не выполняются здесь: :func:`expand_skill` превращает строку в
запрос агенту. Встроенная команда важнее скилла с тем же именем.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field

from gigacrew.agent.chat_store import ChatInfo, ChatRecorder, ChatStoreError, SavedChat
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

# Насколько неизвестная команда должна походить на известную, чтобы её подсказать.
_SUGGESTION_CUTOFF = 0.6


def save_chat(chats: ChatRecorder, agent: Agent, ui: Console) -> None:
    """Сохранить чат.

    Если файл чата изменила другая сессия, диалог сохраняется новым чатом — тогда и план
    этой сессии переезжает к нему (:meth:`Agent.continue_as`).
    """
    warning = chats.save(agent.conversation, model=agent.model)
    if chats.chat_id != agent.chat_id:
        agent.continue_as(chats.chat_id)
    if warning:
        ui.warn(warning)


@dataclass
class CommandContext:
    """Всё, что нужно обработчику слеш-команды: агент, интерфейс, чаты и настройки REPL."""

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
    """Слеш-команда REPL: имя, описание для справки и обработчик."""

    name: str  # с ведущим слешем: "/model"
    summary: str  # кратко — для баннера и справки
    handler: Handler
    usage: str = ""  # синтаксис для справки: "/model [имя]"
    aliases: tuple[str, ...] = ()


class CommandRegistry:
    """Зарегистрированные слеш-команды: поиск по имени и псевдонимам, выполнение строки."""

    def __init__(self) -> None:
        """Создать пустой реестр."""
        self._commands: dict[str, SlashCommand] = {}
        self._lookup: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        """Добавить команду; занятое имя или псевдоним — :class:`ValueError`."""
        for name in (command.name, *command.aliases):
            key = name.lower()
            if key in self._lookup:
                raise ValueError(f"Команда {key} уже зарегистрирована")
            self._lookup[key] = command
        self._commands[command.name] = command

    def get(self, name: str) -> SlashCommand | None:
        """Команда по имени или псевдониму (без учёта регистра); None — такой нет."""
        return self._lookup.get(name.lower())

    def __iter__(self) -> Iterator[SlashCommand]:
        """Команды в порядке регистрации (каждая — один раз, без псевдонимов)."""
        return iter(self._commands.values())

    def dispatch(self, line: str, ctx: CommandContext) -> bool:
        """Выполняет команду из строки. False — завершить REPL."""
        parts = line.strip().split(maxsplit=1)
        name = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""
        command = self.get(name)
        if command is None:
            names = [known for cmd in self for known in (cmd.name, *cmd.aliases)]
            if ctx.skills is not None:
                names += [skill.command for skill in ctx.skills]
            close = difflib.get_close_matches(
                name.lower(),
                names,
                n=1,
                cutoff=_SUGGESTION_CUTOFF,
            )
            hint = f" — возможно, {close[0]}?" if close else ""
            ctx.ui.error(f"неизвестная команда: {name}{hint} (список — /help)")
            return True
        return command.handler(ctx, arg)


# Команда REPL — это слеш + слово (/help, /model ...). Чистый путь
# (/home/...) или начало регэкспа содержит дополнительные слеши и НЕ считается
# командой, а уходит обычным запросом агенту.
_REPL_COMMAND_RE = re.compile(rf"^/{COMMAND_NAME_PATTERN}$")


def is_repl_command(line: str) -> bool:
    """Строка — слеш-команда REPL (или скилл), а не обычный запрос агенту."""
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
    except (OSError, UnicodeDecodeError) as error:
        raise OSError(
            f"не удалось прочитать скилл {skill.name} ({skill.path}): {error}",
        ) from None
    args = parts[1].strip() if len(parts) > 1 else ""
    return skill_invocation_prompt(skill, body, line.strip(), args)


# ------------------------------- команды ------------------------------- #
def _help(ctx: CommandContext, _arg: str) -> bool:
    rows = []
    for cmd in ctx.commands:
        aliases = ", ".join(cmd.aliases)
        alias = f" (также {aliases})" if cmd.aliases else ""
        rows.append((cmd.usage or cmd.name, f"{cmd.summary}{alias}"))
    if ctx.skills is not None:
        for skill in ctx.skills:
            if ctx.commands.get(skill.command) is None:
                about = skill.description or skill.path
                rows.append((f"{skill.command} [аргументы]", f"скилл · {about}"))
    ctx.ui.help(rows, KEY_HELP)
    return True


def catalogue_ids(provider: LLMProvider) -> list[str] | None:
    """Модели каталога провайдера — выбрать можно только их; None — каталога нет."""
    models = provider.known_models()
    return None if models is None else [model.id for model in models]


def _model(ctx: CommandContext, arg: str) -> bool:
    agent = ctx.agent
    catalogue = catalogue_ids(agent.provider)
    known = catalogue
    if known is None and ctx.models:
        known = ctx.models.models
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
        available = ", ".join(catalogue)
        ctx.ui.error(f"модели {arg} нет в каталоге бэкенда; доступны: {available}")
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
    choices = []
    for mode in MODE_CYCLE:
        marker = " · текущий" if mode is current else ""
        choices.append((mode.value, f"{mode.label}{marker} — {mode.description}"))
    return choices


def _mode(ctx: CommandContext, arg: str) -> bool:
    agent = ctx.agent
    if not arg:
        current = agent.mode
        lines = [f"режим: {current.label} — {current.description}"]
        if current is PermissionMode.AUTO and agent.reviewer.enabled:
            reviewer_text = agent.reviewer.describe(agent.model)
            lines.append(f"  {reviewer_text}")
        lines.append("режимы (Shift+Tab — следующий по кругу):")
        lines += [
            f"  • {option.value} — {option.label}: {option.description}" for option in MODE_CYCLE
        ]
        ctx.ui.info("\n".join(lines))
        _warn_reviewer(ctx)
        return True
    try:
        mode = parse_mode(arg)
    except ValueError as error:
        ctx.ui.error(f"{error} (использование: {MODE_USAGE})")
        return True
    agent.set_mode(mode)
    text = f"режим: {mode.label} — {mode.description}"
    if mode is PermissionMode.AUTO and agent.reviewer.enabled:
        reviewer_text = agent.reviewer.describe(agent.model)
        text = f"{text}\n  {reviewer_text}"
    ctx.ui.info(text)
    _warn_reviewer(ctx)
    return True


def _warn_reviewer(ctx: CommandContext) -> None:
    """В режиме «авто» с отключённым рецензентом — напомнить, что команды спрашиваются."""
    agent = ctx.agent
    if agent.mode is PermissionMode.AUTO and not agent.reviewer.enabled:
        ctx.ui.warn(agent.reviewer.warning)


PERMISSIONS_USAGE = "/permissions [forget N | clear | log]"
_MODE_NAMES = "|".join(mode.value for mode in MODE_CYCLE)
MODE_USAGE = f"/mode [{_MODE_NAMES}]"


def _permissions(ctx: CommandContext, arg: str) -> bool:
    """Сохранённые разрешения «всегда» для команд и последние решения."""
    words = arg.split()
    action = words[0].lower() if words else ""
    if action in {"forget", "rm", "удалить"}:
        _forget_permission(ctx, words)
    elif action in {"clear", "очистить"}:
        _clear_permissions(ctx)
    elif action in {"log", "журнал"}:
        _show_decisions(ctx)
    elif action:
        ctx.ui.error(f"использование: {PERMISSIONS_USAGE}")
    else:
        _show_permissions(ctx)
    return True


def _forget_permission(ctx: CommandContext, words: list[str]) -> None:
    """``/permissions forget N``: удалить разрешение с номером N из списка."""
    # isdecimal, а не isdigit: «²» — цифра, но int() её не разберёт
    if len(words) != 2 or not words[1].isdecimal():
        ctx.ui.error(f"использование: {PERMISSIONS_USAGE}")
        return
    memory = ctx.agent.decisions
    try:
        rule = memory.forget(int(words[1]))
    except IndexError:
        ctx.ui.error(f"нет разрешения №{words[1]} (список — /permissions)")
    except OSError as error:
        ctx.ui.error(f"не удалось сохранить: {error}")
    else:
        ctx.ui.info(f"разрешение удалено: {redact(rule.describe())}")


def _clear_permissions(ctx: CommandContext) -> None:
    """``/permissions clear``: удалить все сохранённые разрешения."""
    memory = ctx.agent.decisions
    try:
        removed = memory.clear()
    except OSError as error:
        ctx.ui.error(f"не удалось сохранить: {error}")
    else:
        ctx.ui.info(f"удалено разрешений: {removed}")


def _show_decisions(ctx: CommandContext) -> None:
    """``/permissions log``: последние решения из журнала."""
    memory = ctx.agent.decisions
    decisions = memory.recent()
    if not decisions:
        reason = "" if memory.log else " (не ведётся: --no-save)"
        ctx.ui.info(f"журнал решений пуст{reason}")
        return
    lines = [_describe_decision(decision) for decision in decisions]
    listing = "\n".join(lines)
    ctx.ui.info(f"последние решения:\n{listing}")


def _describe_decision(entry: Mapping[str, object]) -> str:
    """Строка журнала решений для ``/permissions log`` (секреты вырезаны)."""
    decision = str(entry.get("decision"))
    mark = {"yes": "да", "always": "всегда", "no": "нет", "no_answer": "без ответа"}.get(
        decision,
        decision,
    )
    raw_review = entry.get("review")
    review = raw_review if isinstance(raw_review, dict) else {}
    if entry.get("source") == "reviewer":
        mark = "разрешил рецензент"
    cwd = entry.get("cwd")
    where = "" if cwd in {None, "."} else f" [{cwd}]"
    text = f"  • {mark}{where}: {entry.get('command', '')}"
    if entry.get("feedback"):
        text = f"{text} — «{entry['feedback']}»"
    elif entry.get("source") == "reviewer" and review.get("reason"):
        text = f"{text} — {review['reason']}"
    # Журнал прежних версий вырезал меньше форм секретов, пояснение рецензента
    # не вырезается вовсе — на экран не выводим ни то, ни другое.
    return redact(text)


def _show_permissions(ctx: CommandContext) -> None:
    """``/permissions``: сохранённые разрешения и где они хранятся."""
    memory = ctx.agent.decisions
    if memory.rules:
        lines = ["разрешено без вопросов:"]
        for number, rule in enumerate(memory.rules, 1):
            used = f", применялось {rule.hits}" if rule.hits else ""
            # Правило — команда как есть: прежние версии сохраняли и команды с паролем.
            lines.append(f"  {number}. {redact(rule.describe())}{used}")
        lines.append("удалить — /permissions forget N, все — /permissions clear")
    else:
        lines = ["сохранённых разрешений нет: ответ «a» на вопрос о команде запомнит её"]
    if memory.approvals_path is not None and memory.remember:
        lines.append(f"файл: {memory.approvals_path}")
    elif memory.directory is None or not memory.remember:
        lines.append("разрешения хранятся до конца сессии (GIGACREW_REMEMBER_APPROVALS=0)")
    ctx.ui.info("\n".join(lines))


def _start_new_chat(ctx: CommandContext) -> None:
    """Сбросить историю агента и начать новый чат (для ``/clear``)."""
    if ctx.chats is None:
        ctx.agent.reset()
    else:
        ctx.chats.start_new()  # прежний чат остаётся сохранённым — его вернёт /resume
        ctx.agent.reset(chat_id=ctx.chats.chat_id)


def _clear(ctx: CommandContext, _arg: str) -> bool:
    restart_session(
        ctx.agent,
        ctx.ui,
        end_reason="clear",
        source="clear",
        switch=lambda: _start_new_chat(ctx),
    )
    ctx.ui.info("история очищена, начат новый чат")
    return True


def _compact(ctx: CommandContext, arg: str) -> bool:
    agent = ctx.agent
    guard = nullcontext() if ctx.interrupt is None else ctx.interrupt
    try:
        with guard:
            compacted = agent.compact(arg)
    except KeyboardInterrupt:
        ctx.ui.warn("сжатие прервано — история не изменилась")
        return True
    except LLMError as error:
        ctx.ui.error(f"не удалось сжать контекст: {error}")
        return True
    if compacted is None:
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


def _choose_chat(ctx: CommandContext, chats: ChatRecorder, arg: str) -> ChatInfo | None:
    """Чат для ``/resume``: по ссылке из аргумента или выбранный в списке; None — не выбран."""
    if arg:
        return chats.store.find(arg)
    recent = chats.store.recent(limit=RESUME_LIMIT)
    if not recent:
        ctx.ui.info("сохранённых чатов пока нет")
        return None
    return ctx.ui.pick_chat(recent, chats.chat_id)


def _resume(ctx: CommandContext, arg: str) -> bool:
    if ctx.chats is None:
        ctx.ui.warn("сохранённые чаты недоступны")
        return True
    chats = ctx.chats
    try:
        chosen = _choose_chat(ctx, chats, arg)
        if chosen is None:
            return True
        saved = chats.store.load(chosen.id)
    except ChatStoreError as error:
        ctx.ui.error(str(error))
        return True
    agent = ctx.agent
    previous = agent.model
    skipped = restart_session(
        agent,
        ctx.ui,
        end_reason="resume",
        source="resume",
        switch=lambda: resume_chat(agent, chats, saved),
    )
    ctx.ui.chat_resumed(saved.info, saved.conversation.messages, agent.call_info)
    if skipped:
        ctx.ui.warn(f"модели чата {skipped} нет в каталоге бэкенда — модель: {agent.model}")
    if agent.model != previous:
        ctx.ui.info(f"модель чата: {agent.model}")
        ensure_context_window(
            agent,
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
    indexed = run_indexing(index, ctx.ui, ctx.interrupt, rebuild=arg == "rebuild")
    if indexed is not None:
        ty = semantic.status(workspace) if ctx.agent.config.ty else "выключен (GIGACREW_TY=0)"
        ctx.ui.info(describe_index(*indexed, ty_status=ty))
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
    """Реестр встроенных слеш-команд REPL."""
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
