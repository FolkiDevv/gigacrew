"""Интерактивный режим (REPL). Строка ввода — :mod:`gigacrew.cli.prompt`.

Семантика клавиш:
  * Shift+Tab — сменить режим разрешений (и на приглашении, и во время хода);
  * Ctrl+C на приглашении — сбросить ввод (не выход);
  * Ctrl+C или Esc во время ответа — прервать текущий ход агента;
  * Esc Esc на приглашении — очистить ввод;
  * Ctrl+D (EOF) или /exit — выход.

Диалог сохраняется после каждого хода — и прерванного тоже (см. :func:`autosave`).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata

from gigacrew.agent.chat_store import ChatInfo, ChatRecorder
from gigacrew.agent.hooks import PromptBlockedError
from gigacrew.agent.loop import Agent
from gigacrew.cli.commands import (
    CommandContext,
    CommandRegistry,
    expand_skill,
    is_repl_command,
    mode_choices,
    save_chat,
)
from gigacrew.cli.hooks import hook_session
from gigacrew.cli.indexing import ensure_index
from gigacrew.cli.interrupt import EscInterrupt
from gigacrew.cli.models import ModelCatalog, ensure_context_window
from gigacrew.cli.prompt import ArgChoices, InputReader, StatusInfo, make_input_reader
from gigacrew.llm.base import LLMError
from gigacrew.project.skills import SkillCatalog
from gigacrew.ui.console import Console
from gigacrew.ui.format import mode_badge

# Частые команды под стартовым экраном; остальные — в /help.
BANNER_HINTS = (
    ("/help", "все команды"),
    ("/model", "модель"),
    ("/resume", "чаты"),
    ("/clear", "новый чат"),
)


def status_of(agent: Agent) -> StatusInfo:
    """Данные для статус-строки (вычисляются при каждой отрисовке приглашения)."""
    return StatusInfo(
        model=agent.model,
        context_tokens=agent.context_tokens,
        context_budget=agent.context_budget,
        billed_tokens=agent.billed_tokens,
        auto_approve=agent.config.auto_approve,
        mode=agent.mode,
    )


def autosave(chats: ChatRecorder | None, agent: Agent, ui: Console) -> None:
    """Сохранить чат после хода. Ход к этому моменту уже согласован (``repair``)."""
    if chats is None:
        return
    save_chat(chats, agent, ui)


def esc_interrupt_for(
    ui: Console,
    interrupt: EscInterrupt | None = None,
    agent: Agent | None = None,
) -> EscInterrupt:
    """Прерывание ходов клавишей Esc (если stdin — терминал) + подсказка в индикаторе.

    С ``agent`` Shift+Tab во время хода переключает его режим, индикатор хода
    показывает текущий режим, а Esc во время суб-агента останавливает только его
    (первый — с итогом, повторный — обрывает), Ctrl+C — весь ход.
    """
    interrupt = EscInterrupt() if interrupt is None else interrupt
    if interrupt.enabled:
        ui.set_interrupt_keys("Esc — прервать", interrupt.paused, stop_subagent=agent is not None)
        if agent is not None:
            interrupt.on_backtab = agent.cycle_mode
            interrupt.on_escape = agent.request_subagent_stop
            ui.set_mode_hint(lambda: f"{mode_badge(agent.mode)} (Shift+Tab)")
    return interrupt


# Публичная точка входа: параметры передаются по имени из app.py и тестов.
def run_repl(  # noqa: WPS211
    agent: Agent,
    ui: Console,
    commands: CommandRegistry,
    *,
    read_input: InputReader | None = None,
    interrupt: EscInterrupt | None = None,
    chats: ChatRecorder | None = None,
    resumed: ChatInfo | None = None,
    index_on_start: bool = True,
    models: ModelCatalog | None = None,
    skills: SkillCatalog | None = None,
    trust_hooks: bool = False,
) -> int:
    """Запустить интерактивный режим и вернуть код выхода.

    ``interrupt`` подменяется в тестах; по умолчанию Esc слушается только вместе с
    настоящей строкой ввода (``read_input is None``). ``index_on_start`` — до первого
    ввода построить индекс проекта, если его нет (:func:`~gigacrew.cli.indexing.ensure_index`).

    ``chats`` — куда сохранять диалог (None — не сохранять); ``resumed`` — чат,
    продолженный при запуске (``--continue``/``--resume``): после баннера показываются
    его последние обмены.

    ``models`` — каталог моделей для ``/model``; по умолчанию с настоящей строкой
    ввода список загружается в фоне. Замер окна незамеренной модели предлагается
    до первого ввода (:func:`~gigacrew.cli.models.ensure_context_window`).

    ``skills`` — скиллы (``/<имя> [аргументы]``); по умолчанию — из ``.gigacode/skills``
    проекта и ``~/.gigacode/skills``, перечитываются перед каждым вводом.

    Хуки SessionStart — перед первым вводом, SessionEnd — при выходе (Ctrl+D, ``/exit``);
    ``trust_hooks`` (``--trust-hooks``) — для ``/hooks reload``.
    """
    if interrupt is None:
        interrupt = EscInterrupt(enabled=None if read_input is None else False)
    esc = esc_interrupt_for(ui, interrupt, agent)
    if models is None and read_input is None:
        models = ModelCatalog(agent.provider)
    skills = SkillCatalog(agent.workspace) if skills is None else skills
    read = read_input or _make_reader(agent, ui, commands, models, skills)
    _show_banner(agent, ui, commands, resumed)
    # Ввод не принимается, пока строится индекс (Esc/Ctrl+C — отменить построение).
    if index_on_start:
        ensure_index(agent.workspace, ui, esc)
    declined: set[str] = set()  # замер окна отклонён — до конца сессии не спрашиваем
    ensure_context_window(agent, ui, interrupt=esc, declined=declined)
    if models is not None:
        models.start()  # после замера: не делить с ним соединение; ввод сети не ждёт
    ctx = CommandContext(
        agent=agent,
        ui=ui,
        commands=commands,
        chats=chats,
        interrupt=esc,
        models=models,
        declined_windows=declined,
        skills=skills,
        trust_hooks=trust_hooks,
    )
    repl = _Repl(
        agent=agent,
        ui=ui,
        commands=commands,
        ctx=ctx,
        read=read,
        esc=esc,
        skills=skills,
        chats=chats,
    )
    source = "startup" if resumed is None else "resume"
    with hook_session(agent, ui, source=source, end_reason="prompt_input_exit"):
        return repl.loop()


def _make_reader(
    agent: Agent,
    ui: Console,
    commands: CommandRegistry,
    catalog: ModelCatalog | None,
    skills: SkillCatalog,
) -> InputReader:
    """Настоящая строка ввода: автодополнение команд, аргументов и скиллов, статус-строка."""
    arg_choices: dict[str, ArgChoices] = {"/mode": lambda: mode_choices(agent.mode)}
    if catalog is not None:
        arg_choices["/model"] = lambda: catalog.choices(agent.windows, agent.model)
    return make_input_reader(
        commands=commands,
        workspace=agent.workspace,
        status=lambda: status_of(agent),
        no_color=ui.no_color,
        arg_choices=arg_choices,
        on_cycle_mode=agent.cycle_mode,
        skills=lambda: [(skill.command, skill.description) for skill in skills],
    )


def _show_banner(
    agent: Agent,
    ui: Console,
    commands: CommandRegistry,
    resumed: ChatInfo | None,
) -> None:
    """Стартовый экран; для продолженного чата — его последние обмены."""
    ui.banner(
        version=metadata.version("gigacrew"),
        model=agent.model,
        root=str(agent.workspace.root),
        hints=[(name, desc) for name, desc in BANNER_HINTS if commands.get(name) is not None],
        auto_approve=agent.config.auto_approve,
        mode=agent.mode,
    )
    if resumed is not None:
        ui.chat_resumed(resumed, agent.conversation.messages, agent.call_info)


def _report_turn_error(ui: Console, error: PromptBlockedError | LLMError) -> None:
    """Сообщить об ожидаемой ошибке хода: отказ хука или ошибка модели."""
    if isinstance(error, PromptBlockedError):
        ui.warn(f"запрос отклонён хуком UserPromptSubmit: {error.reason}")
    else:
        ui.error(str(error))


@dataclass
class _Repl:
    """Состояние цикла ввода REPL."""

    agent: Agent
    ui: Console
    commands: CommandRegistry
    ctx: CommandContext
    read: InputReader
    esc: EscInterrupt
    skills: SkillCatalog
    chats: ChatRecorder | None
    typeahead: str = ""

    def loop(self) -> int:
        """Цикл ввода REPL до выхода."""
        # Набранное во время хода — в следующую строку ввода.
        self.typeahead = self.esc.take_typeahead()

        while True:
            self.skills.refresh()  # для автодополнения: скиллы могли поменяться за ход
            line = self._read_line()
            if line is None:
                return 0
            if line and not self._execute(line):
                return 0

    def _read_line(self) -> str | None:
        """Прочитать строку ввода; None — выход (EOF), пустая строка — ввод сброшен."""
        try:
            # Набранное во время хода подставляется один раз: Ctrl+C его сбрасывает.
            default = self.typeahead
            self.typeahead = ""
            return (self.read(default=default) if default else self.read()).strip()
        except EOFError:
            self.ui.system("\nдо встречи!")
            return None
        except KeyboardInterrupt:
            self.ui.system("(Ctrl+D или /exit — выход)")
            return ""

    def _execute(self, line: str) -> bool:
        """Выполнить строку: скилл, слеш-команду или ход агента. False — выход из REPL."""
        if is_repl_command(line):
            self.skills.refresh()  # скилл могли установить, пока открыто приглашение
        try:
            request = expand_skill(line, self.commands, self.skills)
        except OSError as exc:
            self.ui.error(str(exc))
            return True
        if request is None and is_repl_command(line):
            return self._dispatch(line)
        self._run_turn(line if request is None else request)
        return True

    def _dispatch(self, line: str) -> bool:
        """Выполнить слеш-команду. False — выход из REPL."""
        try:
            if not self.commands.dispatch(line, self.ctx):
                return False
        except Exception as exc:  # как и ошибка в ходе: сбой команды не закрывает REPL
            self.ui.stop_live()
            name = line.split(maxsplit=1)[0]
            self.ui.error(f"внутренняя ошибка в {name}: {type(exc).__name__}: {exc}")
        self.typeahead = self.esc.take_typeahead()
        return True

    def _run_turn(self, line: str) -> None:
        """Ход агента с обработкой ошибок, затем автосохранение чата."""
        try:
            self._turn(line)
        except (PromptBlockedError, LLMError) as error:
            self.ui.stop_live()
            _report_turn_error(self.ui, error)
        except KeyboardInterrupt:
            self.ui.stop_live()
            self.ui.system("\n(прервано)")
        except Exception as exc:  # ошибка внутри хода не должна закрывать REPL
            self.ui.stop_live()
            self.ui.error(f"внутренняя ошибка: {type(exc).__name__}: {exc}")
        autosave(self.chats, self.agent, self.ui)
        self.typeahead = self.esc.take_typeahead()

    def _turn(self, line: str) -> None:
        """Ход агента: подключение к бэкенду и обработка запроса."""
        self.ui.print()
        # Токен бэкенда — до индикатора хода: истёк — ошибка с просьбой пройти
        # авторизацию в GigaCode, а не сбой посреди хода.
        self.agent.provider.connect()
        with self.esc:
            self.agent.run_turn(line)
        self.ui.print()
