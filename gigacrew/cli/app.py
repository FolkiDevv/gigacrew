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
from pathlib import Path

from gigacrew import __version__
from gigacrew.agent.chat_store import ChatRecorder, ChatStore, ChatStoreError, SavedChat
from gigacrew.agent.decisions import DecisionMemory
from gigacrew.agent.hooks import PromptBlocked
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gigacrew",
        description="Консольный помощник для работы с кодом проекта.",
    )
    p.add_argument("-p", "--prompt", help="Одноразовый запрос (без интерактива)")
    p.add_argument("-m", "--model", help="Модель (переопределяет GIGACREW_MODEL)")
    p.add_argument("-C", "--dir", default=".", help="Корень проекта (по умолчанию текущая папка)")
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Авто-подтверждение операций, кроме опасных (rm -rf, git reset --hard, "
        "секреты, файлы вне проекта…)",
    )
    p.add_argument(
        "--yes-all",
        action="store_true",
        help="Авто-подтверждение всех операций, включая опасные (используйте осознанно)",
    )
    p.add_argument(
        "--mode",
        type=_mode_arg,
        metavar="{" + ",".join(mode.value for mode in PermissionMode) + "}",
        help="Начальный режим: auto (по умолчанию) — правки файлов без вопросов, "
        "изменяющие команды проверяет модель-рецензент (GIGACREW_REVIEWER_MODEL, по "
        "умолчанию — модель чата), сомнительные и опасные — с подтверждением; plan — "
        "только исследование и план (Shift+Tab — сменить). Команды, которые только "
        "читают, — без вопросов",
    )
    chat = p.add_mutually_exclusive_group()
    chat.add_argument(
        "-c",
        "--continue",
        dest="continue_chat",
        action="store_true",
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
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Не сохранять чат и журнал решений о командах (~/.gigacode/projects/)",
    )
    p.add_argument(
        "--trust-hooks",
        action="store_true",
        help="Запускать хуки (~/.gigacode/settings.json, .gigacode/settings*.json) без "
        "подтверждения — для -p в CI; подтверждение не запоминается",
    )
    p.add_argument("--no-color", action="store_true", help="Отключить цвет")
    p.add_argument(
        "--no-stream",
        action="store_true",
        help="Отключить потоковый вывод (ответ печатается целиком в конце)",
    )
    p.add_argument(
        "--list-models", action="store_true", help="Показать доступные чат-модели и выйти"
    )
    p.add_argument(
        "--test-context",
        metavar="MODEL",
        help="Замерить окно контекста модели (пробные запросы), сохранить "
        "в ~/.gigacode/models.json и выйти",
    )
    p.add_argument("--version", action="version", version=f"gigacrew {__version__}")
    return p


def _mode_arg(text: str) -> PermissionMode:
    """``--mode``: режим по имени (прежние ``manual``/``edits`` — это ``auto``)."""
    try:
        return parse_mode(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


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
    return f"модели {model} нет в каталоге бэкенда; доступны: {', '.join(known)}"


def run_oneshot(
    agent: Agent,
    ui: Console,
    prompt: str,
    chats: ChatRecorder | None = None,
    *,
    resumed: bool = False,
) -> int:
    """Один ход; хуки SessionStart/SessionEnd — вокруг него (SessionEnd — после
    сохранения чата). ``resumed`` — чат продолжен (``-c``/``-r``)."""
    esc = esc_interrupt_for(ui, agent=agent)
    source = "resume" if resumed else "startup"
    with hook_session(agent, ui, source=source, end_reason="other"):
        try:
            with esc:
                agent.run_turn(prompt)
        except PromptBlocked as e:
            ui.stop_live()
            ui.warn(f"запрос отклонён хуком UserPromptSubmit: {e.reason}")
            return EXIT_STOPPED
        except LLMError as e:
            ui.stop_live()
            ui.error(str(e))
            return EXIT_LLM_ERROR
        except KeyboardInterrupt:
            ui.stop_live()
            ui.system("\n(прервано)")
            return EXIT_INTERRUPTED
        except Exception as e:  # как в REPL: сообщение вместо traceback
            ui.stop_live()
            ui.error(f"внутренняя ошибка: {type(e).__name__}: {e}")
            return EXIT_ERROR
        finally:
            autosave(chats, agent, ui)
    stats = agent.last_turn
    return EXIT_STOPPED if stats is not None and stats.stop_reason else EXIT_OK


def run_test_context(
    provider: LLMProvider, windows: ModelWindows, model: str, ui: Console, *, base_url: str
) -> int:
    """``--test-context``: принудительный замер окна ``model`` с выводом каждой пробы."""
    if not provider.supports_measure:
        ui.error("провайдер не поддерживает замер окна контекста")
        return EXIT_ERROR
    ui.info(f"замер окна контекста {model}: пробные запросы с ответом в 1 токен")
    try:
        result = measure_context_window(provider, model, ui, verbose=True)
    except KeyboardInterrupt:
        ui.system("\n(замер прерван)")
        return EXIT_INTERRUPTED
    if result is None:
        return EXIT_LLM_ERROR
    ui.success(describe_result(result))
    if not save_window(windows, result, ui, base_url=base_url):
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
    info = ui.pick_chat(recent)
    return None if info is None else store.load(info.id)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ui = Console(no_color=args.no_color)
    # -p "$TASK" с пустой переменной — ошибка: иначе запустился бы REPL, а в CI (stdin
    # уже в конце) он сразу выходит с кодом 0, как будто запрос выполнен.
    for flag, value in (("-p", args.prompt), ("--test-context", args.test_context)):
        if value is not None and not value.strip():
            ui.error(f"{flag}: пустое значение")
            return EXIT_ERROR
    if args.prompt is not None:
        ui.keyboard_only()  # -p: ответы на вопросы — не из перенаправленного stdin
    root = Path(args.dir)
    if not root.is_dir():
        ui.error(f"каталог проекта не найден: {root}")
        return EXIT_ERROR
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
    except ConfigError as e:
        ui.error(f"ошибка конфигурации: {e}")
        return EXIT_ERROR
    except RuntimeError as e:
        ui.error(str(e))
        return EXIT_ERROR
    # До первого запроса и во всех режимах: ключ уходит уже при подключении.
    warning = tls_warning(config)
    if warning:
        ui.warn(warning)

    try:
        provider = _make_provider(config)
    except Exception as e:
        ui.error(f"не удалось инициализировать провайдера: {e}")
        return EXIT_ERROR

    try:
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
        except LLMError as e:
            ui.error(f"не удалось подключиться: {e}")
            return EXIT_LLM_ERROR
        if args.list_models:
            try:
                models = provider.list_models()
            except LLMError as e:
                ui.error(f"не удалось получить список моделей: {e}")
                return EXIT_LLM_ERROR
            ui.info("Доступные модели:\n" + "\n".join(f"  • {m}" for m in models))
            return EXIT_OK

        catalogue = provider.known_models()  # None — каталога нет, модель любая
        known = None if catalogue is None else [m.id for m in catalogue]
        windows, warning = ModelWindows.load(base_url=config.base_url)
        if warning:
            ui.warn(warning)
        if catalogue is not None:
            windows.report({m.id: m.context_window for m in catalogue})
        if args.test_context is not None:
            model = args.test_context.strip()
            if known is not None and model not in known:
                ui.error(_not_in_catalogue(model, known))
                return EXIT_ERROR
            return run_test_context(provider, windows, model, ui, base_url=config.base_url)

        if known is not None:
            if not config.model:  # модель по умолчанию — бэкенда
                config = dataclasses.replace(config, model=provider.model)
            elif config.model not in known:
                ui.error(_not_in_catalogue(config.model, known) + " (список — --list-models)")
                return EXIT_ERROR
        if not config.model:
            ui.error(
                "не задана модель: укажите GIGACREW_MODEL или --model (список — --list-models)"
            )
            return EXIT_ERROR
        decisions, warning = DecisionMemory.load(
            config.project_root, remember=config.remember_approvals, log=config.save_chats
        )
        if warning:
            ui.warn(warning)
        reviewer, warning = check_reviewer(provider, config.reviewer_model)
        if warning:
            ui.warn(warning)
        agent = Agent(
            provider,
            build_default_registry(),
            config,
            ui,
            windows=windows,
            decisions=decisions,
            reviewer=reviewer,
            hooks=hooks,
        )
        store = ChatStore(agent.workspace)
        for notice in store.migrate():  # чаты с прежнего места — из .gigacode/chats проекта
            ui.info(notice)
        chats = ChatRecorder(store, enabled=config.save_chats)
        agent.chat_id = chats.chat_id  # файл плана назван по id чата
        try:
            saved = _chat_to_resume(args, store, ui)
        except ChatStoreError as e:
            ui.error(str(e))
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
            except OSError as e:
                ui.error(str(e))
                return EXIT_ERROR
            ensure_context_window(agent, ui, interactive=False)
            return run_oneshot(agent, ui, prompt, chats, resumed=saved is not None)
        resumed = saved.info if saved is not None else None
        return run_repl(
            agent, ui, commands, chats=chats, resumed=resumed, trust_hooks=args.trust_hooks
        )
    finally:
        provider.close()
