"""Точка входа gigacrew и сборка компонентов.

Режимы:
  * интерактивный REPL (по умолчанию);
  * одноразовый запрос: ``gigacrew -p "сделай X"`` (или ``-p "/скилл аргументы"``);
  * диагностика: ``gigacrew --list-models`` (чат-модели);
  * замер окна контекста: ``gigacrew --test-context МОДЕЛЬ`` — результат
    сохраняется в ``~/.gigacode/models.json``. Окно незамеренной модели
    замеряется и автоматически — при запуске и смене модели.

Провайдер — ``GIGACREW_PROVIDER``: API v2 (по умолчанию) или бэкенд GigaCode
(``gigacode``): токен — из файла официального CLI GigaCode, модели и их окна — из
каталога бэкенда, модель по умолчанию — его же.

Чаты сохраняются в ``~/.gigacode/projects/<ключ>/chats/`` (``--no-save`` — нет; прежние
из ``.gigacode/chats/`` проекта переносятся туда); ``-c`` продолжает последний, ``-r [ID]`` —
выбранный (без ID — селектор чатов).

Хуки пользователя — из ``~/.gigacode/settings.json`` и ``.gigacode/settings*.json``
проекта (:mod:`gigacrew.cli.hooks`), после подтверждения или с ``--trust-hooks``; хук
PreRequest срабатывает перед каждым запросом к бэкенду GigaCode.

Коды возврата: 0 — успех, 1 — ошибка конфигурации/внутренняя, 2 — ошибка LLM,
3 — ход остановлен ограничителем (лимит шагов, серия ошибок, зацикливание) или запрос
отклонён хуком, 130 — прервано пользователем (Ctrl+C).
"""

from __future__ import annotations

import argparse
import dataclasses
from contextlib import closing
from importlib import metadata
from pathlib import Path

from gigacrew.agent.chat_store import ChatRecorder, ChatStore, ChatStoreError, SavedChat
from gigacrew.agent.decisions import DecisionMemory
from gigacrew.agent.hooks import HookRunner, PromptBlockedError
from gigacrew.agent.loop import Agent
from gigacrew.agent.reviewer import check_reviewer
from gigacrew.cli.commands import RESUME_LIMIT, default_commands, expand_skill, resume_chat
from gigacrew.cli.hooks import hook_session, install_request_hook, load_hooks
from gigacrew.cli.models import (
    describe_result,
    ensure_context_window,
    measure_context_window,
    save_window,
)
from gigacrew.cli.repl import autosave, esc_interrupt_for, run_repl
from gigacrew.config import Config, ConfigError
from gigacrew.llm.api import ApiProvider
from gigacrew.llm.base import LLMError, LLMProvider
from gigacrew.llm.gigacode import GigaCodeProvider
from gigacrew.llm.model_windows import ModelWindows
from gigacrew.permissions import PermissionMode, parse_mode
from gigacrew.project.skills import SkillCatalog
from gigacrew.project.workspace import Workspace
from gigacrew.tools.base import build_default_registry
from gigacrew.ui.console import Console

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_LLM_ERROR = 2
EXIT_STOPPED = 3
EXIT_INTERRUPTED = 130

_STORE_TRUE = "store_true"


def build_parser() -> argparse.ArgumentParser:
    """Разбор аргументов командной строки ``gigacrew``."""
    parser = argparse.ArgumentParser(
        prog="gigacrew",
        description="Консольный помощник для работы с кодом проекта.",
    )
    _add_run_arguments(parser)
    chat = parser.add_mutually_exclusive_group()
    chat.add_argument(
        "-c",
        "--continue",
        dest="continue_chat",
        action=_STORE_TRUE,
        help="Продолжить последний сохранённый чат",
    )
    chat.add_argument(
        "-r",
        "--resume",
        nargs="?",
        const="",
        metavar="ID",
        help="Продолжить сохранённый чат по id (без id — выбрать из списка)",
    )
    parser.add_argument(
        "--no-save",
        action=_STORE_TRUE,
        help="Не сохранять чат и журнал решений о командах (~/.gigacode/projects/)",
    )
    parser.add_argument(
        "--trust-hooks",
        action=_STORE_TRUE,
        help="Запускать хуки (~/.gigacode/settings.json, .gigacode/settings*.json) без "
        "подтверждения — для -p в CI; подтверждение не запоминается",
    )
    parser.add_argument("--no-color", action=_STORE_TRUE, help="Отключить цвет")
    parser.add_argument(
        "--no-stream",
        action=_STORE_TRUE,
        help="Отключить потоковый вывод (ответ печатается целиком в конце)",
    )
    parser.add_argument(
        "--list-models",
        action=_STORE_TRUE,
        help="Показать доступные чат-модели и выйти",
    )
    parser.add_argument(
        "--test-context",
        metavar="MODEL",
        help="Замерить окно контекста модели (пробные запросы), сохранить "
        "в ~/.gigacode/models.json и выйти",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"gigacrew {metadata.version('gigacrew')}",
    )
    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Аргументы запуска: запрос, модель, корень проекта, подтверждения и режим."""
    parser.add_argument("-p", "--prompt", help="Одноразовый запрос (без интерактива)")
    parser.add_argument("-m", "--model", help="Модель (переопределяет GIGACREW_MODEL)")
    parser.add_argument(
        "-C",
        "--dir",
        default=".",
        help="Корень проекта (по умолчанию текущая папка)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action=_STORE_TRUE,
        help="Авто-подтверждение операций, кроме опасных (rm -rf, git reset --hard, правка "
        ".env, файлы вне проекта…) и команд, которые классификатор не смог проверить: их "
        "в режиме auto смотрит рецензент, затем вы. Файлы с секретами read_file и поиск "
        "не читают и с -y",
    )
    parser.add_argument(
        "--yes-all",
        action=_STORE_TRUE,
        help="Авто-подтверждение всех операций, включая опасные и (в режиме auto) "
        "непроверенные команды — используйте осознанно",
    )
    modes = ",".join(mode.value for mode in PermissionMode)
    parser.add_argument(
        "--mode",
        type=_mode_arg,
        metavar=f"{{{modes}}}",
        help="Начальный режим: auto (по умолчанию) — правки файлов без вопросов, "
        "изменяющие команды проверяет модель-рецензент (GIGACREW_REVIEWER_MODEL, по "
        "умолчанию — модель чата), сомнительные и опасные — с подтверждением; plan — "
        "только исследование и план (Shift+Tab — сменить). Команды, которые только "
        "читают, — без вопросов",
    )


def _mode_arg(text: str) -> PermissionMode:
    """``--mode``: режим по имени (прежние ``manual``/``edits`` — это ``auto``)."""
    try:
        return parse_mode(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _make_provider(config: Config) -> LLMProvider:
    if config.provider == "gigacode":
        return GigaCodeProvider(config)
    return ApiProvider(config)


def tls_warning(config: Config) -> str | None:
    """Предупреждение о выключенной проверке TLS (только явный GIGACREW_VERIFY_SSL=0)."""
    if config.verify_ssl:
        return None
    what = "запросы и код проекта" if config.auth_mode == "mtls" else "ключ или токен"
    return (
        f"проверка TLS-сертификата сервера выключена (GIGACREW_VERIFY_SSL=0) — {what} можно "
        "перехватить подменой сервера; если корень сертификата сервера не в стандартном "
        "хранилище, задайте GIGACREW_CA_BUNDLE (PEM) вместо отключения проверки"
    )


def _not_in_catalogue(model: str, known: list[str]) -> str:
    available = ", ".join(known)
    return f"модели {model} нет в каталоге бэкенда; доступны: {available}"


def run_oneshot(
    agent: Agent,
    ui: Console,
    prompt: str,
    chats: ChatRecorder | None = None,
    *,
    resumed: bool = False,
) -> int:
    """Один ход; хуки SessionStart/SessionEnd — вокруг него.

    SessionEnd — после сохранения чата. ``resumed`` — чат продолжен (``-c``/``-r``).
    """
    esc = esc_interrupt_for(ui, agent=agent)
    source = "resume" if resumed else "startup"
    with hook_session(agent, ui, source=source, end_reason="other"):
        try:
            with esc:
                agent.run_turn(prompt)
        except (PromptBlockedError, LLMError) as error:
            ui.stop_live()
            if isinstance(error, PromptBlockedError):
                ui.warn(f"запрос отклонён хуком UserPromptSubmit: {error.reason}")
                return EXIT_STOPPED
            ui.error(str(error))
            return EXIT_LLM_ERROR
        except KeyboardInterrupt:
            ui.stop_live()
            ui.system("\n(прервано)")
            return EXIT_INTERRUPTED
        except Exception as error:  # как в REPL: сообщение вместо traceback
            ui.stop_live()
            ui.error(f"внутренняя ошибка: {type(error).__name__}: {error}")
            return EXIT_ERROR
        finally:
            autosave(chats, agent, ui)
    stats = agent.last_turn
    return EXIT_STOPPED if stats is not None and stats.stop_reason else EXIT_OK


def run_test_context(
    provider: LLMProvider,
    windows: ModelWindows,
    model: str,
    ui: Console,
    *,
    base_url: str,
) -> int:
    """``--test-context``: принудительный замер окна ``model`` с выводом каждой пробы."""
    if not provider.supports_measure:
        ui.error("провайдер не поддерживает замер окна контекста")
        return EXIT_ERROR
    ui.info(f"замер окна контекста {model}: пробные запросы с ответом в 1 токен")
    try:
        measurement = measure_context_window(provider, model, ui, verbose=True)
    except KeyboardInterrupt:
        ui.system("\n(замер прерван)")
        return EXIT_INTERRUPTED
    if measurement is None:
        return EXIT_LLM_ERROR
    ui.success(describe_result(measurement))
    if not save_window(windows, measurement, ui, base_url=base_url):
        return EXIT_ERROR
    ui.system(f"записано в {windows.path}")
    return EXIT_OK


def _chat_to_resume(args: argparse.Namespace, store: ChatStore, ui: Console) -> SavedChat | None:
    """Чат для ``--continue``/``--resume``; None — начать новый. Ошибка — ChatStoreError."""
    if args.continue_chat:
        saved = store.latest()
        if saved is None:
            ui.info("сохранённых чатов нет — начат новый чат")
        return saved
    if args.resume is None:
        return None
    if args.resume:
        return store.load(store.find(args.resume).id)
    recent = store.recent(limit=RESUME_LIMIT)
    if not recent:
        ui.info("сохранённых чатов нет — начат новый чат")
        return None
    picked = ui.pick_chat(recent)
    return None if picked is None else store.load(picked.id)


def _load_config(args: argparse.Namespace, ui: Console) -> Config | None:
    """Проверка аргументов и конфигурация запуска; None — ошибка (сообщение выведено)."""
    # -p "$TASK" с пустой переменной — ошибка: иначе запустился бы REPL, а в CI (stdin
    # уже в конце) он сразу выходит с кодом 0, как будто запрос выполнен.
    for flag, given in (("-p", args.prompt), ("--test-context", args.test_context)):
        if given is not None and not given.strip():
            ui.error(f"{flag}: пустое значение")
            return None
    if args.prompt is not None:
        ui.keyboard_only()  # -p: ответы на вопросы — не из перенаправленного stdin
    root = Path(args.dir)
    if not root.is_dir():
        ui.error(f"каталог проекта не найден: {root}")
        return None
    try:
        config = Config.load(
            project_root=root,
            model=args.model,
            auto_approve=args.yes,
            yes_all=args.yes_all,
            mode=args.mode,
            stream=not args.no_stream,
            save_chats=not args.no_save,
        )
        # До проверки реквизитов: адрес API из .env проекта пропущен — пусть будет видно почему.
        if config.ignored_env_warning:
            ui.warn(config.ignored_env_warning)
        config.require_credentials()
    except ConfigError as error:
        ui.error(f"ошибка конфигурации: {error}")
        return None
    except RuntimeError as error:
        ui.error(str(error))
        return None
    return config


def _list_models(provider: LLMProvider, ui: Console) -> int:
    """``--list-models``: доступные чат-модели."""
    try:
        models = provider.list_models()
    except LLMError as error:
        ui.error(f"не удалось получить список моделей: {error}")
        return EXIT_LLM_ERROR
    listing = "\n".join(f"  • {model}" for model in models)
    ui.info(f"Доступные модели:\n{listing}")
    return EXIT_OK


def _build_agent(
    config: Config,
    provider: LLMProvider,
    ui: Console,
    windows: ModelWindows,
    hooks: HookRunner,
    known: list[str] | None,
) -> Agent | None:
    """Агент с моделью из настроек или из каталога ``known``; None — модели нет.

    ``known`` — модели каталога бэкенда (None — каталога нет, модель любая). Ошибка
    выбора модели выводится здесь же.
    """
    if known is not None:
        if not config.model:  # модель по умолчанию — бэкенда
            config = dataclasses.replace(config, model=provider.model)
        elif config.model not in known:
            missing = _not_in_catalogue(config.model, known)
            ui.error(f"{missing} (список — --list-models)")
            return None
    if not config.model:
        ui.error(
            "не задана модель: укажите GIGACREW_MODEL или --model (список — --list-models)",
        )
        return None
    decisions, warning = DecisionMemory.load(
        config.project_root,
        remember=config.remember_approvals,
        log=config.save_chats,
    )
    if warning:
        ui.warn(warning)
    reviewer, warning = check_reviewer(provider, config.reviewer_model)
    if warning:
        ui.warn(warning)
    return Agent(
        provider,
        build_default_registry(),
        config,
        ui,
        windows=windows,
        decisions=decisions,
        reviewer=reviewer,
        hooks=hooks,
    )


def _run_session(args: argparse.Namespace, agent: Agent, config: Config, ui: Console) -> int:
    """Чат (новый или продолженный) и режим работы: одноразовый запрос ``-p`` или REPL."""
    store = ChatStore(agent.workspace)
    for notice in store.migrate():  # чаты с прежнего места — из .gigacode/chats проекта
        ui.info(notice)
    chats = ChatRecorder(store, enabled=config.save_chats)
    agent.chat_id = chats.chat_id  # файл плана назван по id чата
    try:
        saved = _chat_to_resume(args, store, ui)
    except ChatStoreError as error:
        ui.error(str(error))
        return EXIT_ERROR
    if saved is not None:
        # Модель чата, если модель не задана явно через -m.
        skipped = resume_chat(agent, chats, saved, keep_model=args.model is not None)
        if skipped:
            ui.warn(f"модели чата {skipped} нет в каталоге бэкенда — модель: {agent.model}")
    commands = default_commands()
    if args.prompt is not None:
        skills = SkillCatalog(agent.workspace)
        skills.refresh()
        try:  # -p "/скилл аргументы" — как в REPL
            prompt = expand_skill(args.prompt, commands, skills) or args.prompt
        except OSError as error:
            ui.error(str(error))
            return EXIT_ERROR
        ensure_context_window(agent, ui, interactive=False)
        return run_oneshot(agent, ui, prompt, chats, resumed=saved is not None)
    resumed = None if saved is None else saved.meta
    return run_repl(
        agent,
        ui,
        commands,
        chats=chats,
        resumed=resumed,
        trust_hooks=args.trust_hooks,
    )


def _run(args: argparse.Namespace, ui: Console, config: Config, provider: LLMProvider) -> int:
    """Работа с созданным провайдером: хуки, подключение и выбранный режим запуска."""
    # Хуки — до подключения: подтверждение спрашивается до начала работы, а хук
    # PreRequest идёт уже перед запросом конфигурации (может обновить токен).
    hooks = load_hooks(
        Workspace(config.project_root),
        ui,
        interactive=args.prompt is None and ui.interactive(),
        trust=args.trust_hooks,
    )
    install_request_hook(provider, hooks, ui, config)
    try:
        provider.connect()  # токен и каталог моделей (бэкенд GigaCode)
    except KeyboardInterrupt:
        ui.system("\n(подключение прервано)")
        return EXIT_INTERRUPTED
    except LLMError as error:
        ui.error(f"не удалось подключиться: {error}")
        return EXIT_LLM_ERROR
    if args.list_models:
        return _list_models(provider, ui)

    catalogue = provider.known_models()  # None — каталога нет, модель любая
    known = None if catalogue is None else [entry.id for entry in catalogue]
    windows, warning = ModelWindows.load(base_url=config.base_url)
    if warning:
        ui.warn(warning)
    if catalogue is not None:
        windows.report({entry.id: entry.context_window for entry in catalogue})
    if args.test_context is not None:
        model = args.test_context.strip()
        if known is not None and model not in known:
            ui.error(_not_in_catalogue(model, known))
            return EXIT_ERROR
        return run_test_context(provider, windows, model, ui, base_url=config.base_url)

    agent = _build_agent(config, provider, ui, windows, hooks, known)
    if agent is None:
        return EXIT_ERROR
    return _run_session(args, agent, config, ui)


def main(argv: list[str] | None = None) -> int:
    """Точка входа ``gigacrew``; возвращает код выхода (``EXIT_*``)."""
    args = build_parser().parse_args(argv)
    ui = Console(no_color=args.no_color)
    config = _load_config(args, ui)
    if config is None:
        return EXIT_ERROR
    # До первого запроса и во всех режимах: ключ уходит уже при подключении.
    warning = tls_warning(config)
    if warning:
        ui.warn(warning)

    try:
        provider = _make_provider(config)
    except Exception as error:
        ui.error(f"не удалось инициализировать провайдера: {error}")
        return EXIT_ERROR

    with closing(provider):
        return _run(args, ui, config, provider)
