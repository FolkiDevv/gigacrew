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

from gigacrew import __version__
from gigacrew.agent.chat_store import ChatInfo, ChatRecorder
from gigacrew.agent.hooks import PromptBlocked
from gigacrew.agent.loop import Agent
from gigacrew.cli.commands import (
    CommandContext,
    CommandRegistry,
    expand_skill,
    is_repl_command,
    mode_choices,
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

TLS_OFF_WARNING = (
    "проверка TLS-сертификата сервера API выключена — ключ или токен можно перехватить "
    "подменой сервера; задайте GIGACREW_CA_BUNDLE (корневой сертификат сервера, PEM)"
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
    warning = chats.save(agent.conversation, model=agent.model)
    if warning:
        ui.warn(warning)


def esc_interrupt_for(
    ui: Console, interrupt: EscInterrupt | None = None, agent: Agent | None = None
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


def run_repl(
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
    """``interrupt`` подменяется в тестах; по умолчанию Esc слушается только вместе с
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
    ``trust_hooks`` (``--trust-hooks``) — для ``/hooks reload``."""
    if interrupt is None:
        interrupt = EscInterrupt(enabled=None if read_input is None else False)
    esc = esc_interrupt_for(ui, interrupt, agent)
    if models is None and read_input is None:
        models = ModelCatalog(agent.provider)
    catalog = models
    skills = SkillCatalog(agent.workspace) if skills is None else skills
    arg_choices: dict[str, ArgChoices] = {"/mode": lambda: mode_choices(agent.mode)}
    if catalog is not None:
        arg_choices["/model"] = lambda: catalog.choices(agent.windows, agent.model)
    read = read_input or make_input_reader(
        commands=commands,
        workspace=agent.workspace,
        status=lambda: status_of(agent),
        no_color=ui.no_color,
        arg_choices=arg_choices,
        on_cycle_mode=agent.cycle_mode,
        skills=lambda: [(skill.command, skill.description) for skill in skills],
    )
    ui.banner(
        version=__version__,
        model=agent.model,
        root=str(agent.workspace.root),
        hints=[(cmd.name, cmd.summary) for cmd in commands],
        auto_approve=agent.config.auto_approve,
        mode=agent.mode,
    )
    cfg = agent.config
    if cfg.auth_mode in ("oauth", "cli_token") and not cfg.verify_ssl:
        ui.warn(TLS_OFF_WARNING)
    if resumed is not None:
        ui.chat_resumed(resumed, agent.conversation.messages, agent.describe_call)
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
    source = "resume" if resumed is not None else "startup"
    with hook_session(agent, ui, source=source, end_reason="prompt_input_exit"):
        return _loop(agent, ui, commands, ctx, read, esc, skills, chats)


def _loop(
    agent: Agent,
    ui: Console,
    commands: CommandRegistry,
    ctx: CommandContext,
    read: InputReader,
    esc: EscInterrupt,
    skills: SkillCatalog,
    chats: ChatRecorder | None,
) -> int:
    """Цикл ввода REPL до выхода."""
    typeahead = esc.take_typeahead()  # набранное во время хода — в следующую строку ввода

    while True:
        skills.refresh()  # для автодополнения: скиллы могли поменяться за ход
        try:
            # Набранное во время хода подставляется один раз: Ctrl+C его сбрасывает.
            default, typeahead = typeahead, ""
            line = (read(default=default) if default else read()).strip()
        except EOFError:
            ui.system("\nдо встречи!")
            return 0
        except KeyboardInterrupt:
            ui.system("(Ctrl+D или /exit — выход)")
            continue
        if not line:
            continue

        if is_repl_command(line):
            skills.refresh()  # скилл могли установить, пока открыто приглашение
        try:
            request = expand_skill(line, commands, skills)
        except OSError as e:
            ui.error(str(e))
            continue
        if request is not None:
            line = request
        elif is_repl_command(line):
            if not commands.dispatch(line, ctx):
                return 0
            typeahead = esc.take_typeahead()
            continue

        try:
            ui.print()
            # Токен бэкенда — до индикатора хода: истёк — ошибка с просьбой пройти
            # авторизацию в GigaCode, а не сбой посреди хода.
            agent.provider.connect()
            with esc:
                agent.run_turn(line)
            ui.print()
        except PromptBlocked as e:
            ui.stop_live()
            ui.warn(f"запрос отклонён хуком UserPromptSubmit: {e.reason}")
        except LLMError as e:
            ui.stop_live()
            ui.error(str(e))
        except KeyboardInterrupt:
            ui.stop_live()
            ui.system("\n(прервано)")
        except Exception as e:  # ошибка внутри хода не должна закрывать REPL
            ui.stop_live()
            ui.error(f"внутренняя ошибка: {type(e).__name__}: {e}")
        autosave(chats, agent, ui)
        typeahead = esc.take_typeahead()
