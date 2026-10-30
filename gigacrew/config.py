"""Конфигурация gigacrew.

Источники настроек (в порядке приоритета):
  1. Явные аргументы командной строки (передаются в Config.load()).
  2. Переменные окружения.
  3. Файл пользователя ``~/.gigacode/.env`` (общий для всех проектов).
  4. Файл .env в корне проекта.
  5. Файл .env в текущей директории (если она не корень проекта).
  6. Значения по умолчанию.

Файлы .env проекта и текущей директории — содержимое репозитория, поэтому настройки
безопасности (адреса API, проверка TLS, корневые сертификаты, каталог GigaCode,
клиентский сертификат — :data:`SECURITY_KEYS`) из них не берутся: только из окружения
и файла пользователя.

Файлы .env читаются в отдельный словарь и НЕ попадают в ``os.environ`` — иначе
секреты унаследовали бы все процессы, которые запускает агент.
Секреты (ключ доступа к API) НЕ хранятся в коде — только в окружении / .env.
Адреса API и модель тоже не имеют значений по умолчанию — их задаёт пользователь.
"""

from __future__ import annotations

import os
import re
import ssl
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from gigacrew.permissions import PermissionMode, parse_mode
from gigacrew.project.workspace import user_data_dir

# Значения по умолчанию и границы числовых настроек (время — в секундах, доли — в процентах).
_DEFAULT_TIMEOUT = 120
_DEFAULT_READ_TIMEOUT = 600
_DEFAULT_REVIEWER_TIMEOUT = 60
_DEFAULT_RATE_LIMIT_WAIT = 120
_DEFAULT_TEMPERATURE = 0.2
_MIN_TEMPERATURE = float(0)
_MAX_TEMPERATURE = 2.0
_MIN_CONTEXT_TOKENS = 4_000
_DEFAULT_SHARE_PERCENT = 80
_MIN_SHARE_PERCENT = 10
_MAX_SHARE_PERCENT = 95
_PERCENT = 100
_DEFAULT_SUBAGENT_MAX_STEPS = 25
_DEFAULT_SUBAGENT_TIMEOUT = 600
_MIN_SUBAGENT_TIMEOUT = 10


class ConfigError(ValueError):
    """Некорректное значение настройки (сообщение пригодно для показа пользователю)."""


def read_dotenv(path: Path) -> dict[str, str]:
    """Минимальный парсер .env: KEY=VALUE, поддержка # комментариев и кавычек.

    Возвращает словарь; глобальное окружение процесса не изменяется.
    """
    settings: dict[str, str] = {}
    if not path.is_file():
        return settings
    try:
        # utf-8-sig: BOM (PowerShell 5, старый Блокнот) иначе приклеился бы к первому ключу,
        # и первая настройка — например, GIGACREW_VERIFY_SSL=1 — молча пропала бы.
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Не удалось прочитать {path}: {exc}") from exc
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw_setting = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if key:
            settings[key] = _dotenv_value(raw_setting)
    return settings


_INLINE_COMMENT_RE = re.compile(r"\s#")


def _dotenv_value(raw: str) -> str:
    """Значение из .env без кавычек и комментария в конце строки.

    В кавычках — до парной закрывающей (остальное — комментарий), без кавычек — до « #»
    (``MODEL=model-large # быстрая``).
    """
    setting = raw.strip()
    if setting[:1] in {"'", '"'}:
        end = setting.find(setting[0], 1)
        return setting if end == -1 else setting[1:end]
    return _INLINE_COMMENT_RE.split(setting, maxsplit=1)[0].rstrip()


# Настройки безопасности: их задают только окружение и ~/.gigacode/.env, но не .env проекта
# (или текущей папки) — это содержимое репозитория, его может прислать кто угодно.
#   * GIGACREW_API_URL — куда идут токен, клиентский сертификат и код проекта (у бэкенда
#     GigaCode — и адрес его конфигурации, из которой берётся адрес чата);
#   * GIGACREW_AUTH_URL — куда уходит Basic-ключ OAuth;
#   * GIGACREW_VERIFY_SSL, GIGACREW_CA_BUNDLE — кому верить как серверу;
#   * GIGACODE_HOME — откуда берутся токен официального CLI и его доверенные корни (certs/);
#   * GIGACREW_CERT, GIGACREW_KEY — чьим сертификатом подписаны запросы: с чужим запросы
#     и код проекта ушли бы под учётной записью автора репозитория.
# Ключ доступа (GIGACREW_ACCESS_KEY) из .env проекта по-прежнему читается: на чужой адрес
# его уже не отправить. User-Agent и scope ничего не перенаправляют, модель и режим
# разрешений — обычные настройки (режим из .env может лишь ужесточить работу).
SECURITY_KEYS: tuple[str, ...] = (
    "GIGACREW_API_URL",
    "GIGACREW_AUTH_URL",
    "GIGACREW_VERIFY_SSL",
    "GIGACREW_CA_BUNDLE",
    "GIGACODE_HOME",
    "GIGACREW_CERT",
    "GIGACREW_KEY",
)
USER_ENV_HINT = "~/.gigacode/.env"


def user_env_file() -> Path:
    """``~/.gigacode/.env`` — настройки пользователя, общие для всех проектов (доверенные)."""
    return user_data_dir() / ".env"


@dataclass(frozen=True)
class Environment:
    """Сводное окружение и настройки безопасности, пропущенные в .env проекта."""

    variables: dict[str, str]
    ignored: tuple[tuple[str, str], ...] = ()  # (ключ, файл .env)


def collect_environment(
    root: Path,
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    trusted_dotenv: bool = False,
) -> Environment:
    """Сводное окружение: environ > ~/.gigacode/.env > <root>/.env > <cwd>/.env.

    Из ``<root>/.env`` и ``<cwd>/.env`` настройки безопасности (:data:`SECURITY_KEYS`) не
    берутся; ``trusted_dotenv`` — берутся (собственный .env разработчиков gigacrew для
    live-тестов). Пропущенные возвращаются, только если итоговое значение другое.
    Порядок детерминирован; одинаковые пути читаются один раз.
    """
    cwd = (cwd or Path.cwd()).resolve()
    root = root.resolve()
    user_file = user_env_file()
    merged: dict[str, str] = {}
    # Пропущенные настройки безопасности: (ключ, файл .env, значение).
    skipped: list[tuple[str, str, str]] = []
    # От младшего источника к старшему: каждый следующий перекрывает предыдущий.
    for path in (root / ".env",) if cwd == root else (cwd / ".env", root / ".env"):
        if path == user_file:
            continue  # проект — сам ~/.gigacode: файл прочитается ниже, как доверенный
        project_settings = read_dotenv(path)
        if not trusted_dotenv:
            skipped.extend(
                (key, str(path), setting)
                for key, setting in project_settings.items()
                if key in SECURITY_KEYS and setting.strip()
            )
            project_settings = {
                key: setting
                for key, setting in project_settings.items()
                if key not in SECURITY_KEYS
            }
        merged.update(project_settings)
    merged.update(read_dotenv(user_file))
    merged.update(os.environ if environ is None else environ)
    return Environment(
        merged,
        tuple((key, source) for key, source, setting in skipped if merged.get(key) != setting),
    )


def load_environment(
    root: Path,
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    trusted_dotenv: bool = False,
) -> dict[str, str]:
    """Сводное окружение (см. :func:`collect_environment`)."""
    collected = collect_environment(root, environ=environ, cwd=cwd, trusted_dotenv=trusted_dotenv)
    return collected.variables


def _env_bool(env: Mapping[str, str], name: str, *, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    norm = raw.strip().lower()
    if norm in {"1", "true", "yes", "on"}:
        return True
    if norm in {"0", "false", "no", "off"}:
        return False
    # Опечатка не должна молча выключать настройку — например, проверку TLS у GigaCode.
    raise ConfigError(f"{name}: ожидается 1/0, true/false, yes/no или on/off, получено {raw!r}")


def _env_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        number = int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name}: ожидается целое число, получено {raw!r}") from exc
    if number < minimum:
        raise ConfigError(f"{name}: значение должно быть не меньше {minimum}, получено {number}")
    if maximum is not None and number > maximum:
        raise ConfigError(f"{name}: значение должно быть не больше {maximum}, получено {number}")
    return number


def _env_optional_int(env: Mapping[str, str], name: str, *, minimum: int = 1) -> int | None:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return None
    return _env_int(env, name, 0, minimum=minimum)


def _env_mode(env: Mapping[str, str], name: str) -> PermissionMode:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return PermissionMode.AUTO
    try:
        return parse_mode(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}: {exc}") from exc


def _env_choice(env: Mapping[str, str], name: str, choices: tuple[str, ...]) -> str:
    """Значение из списка (первое — по умолчанию)."""
    raw = (env.get(name) or "").strip().lower()
    if not raw:
        return choices[0]
    if raw not in choices:
        allowed = ", ".join(choices)
        raise ConfigError(f"{name}: ожидается одно из {allowed}, получено {raw!r}")
    return raw


_HEADER_VALUE_RE = re.compile(r"[\x20-\x7e]+")


def _env_header(env: Mapping[str, str], name: str) -> str | None:
    """Значение HTTP-заголовка: печатный ASCII (пробелы по краям обрезаются); пусто — None.

    Иначе запрос не собрать — ошибка при запуске, а не на первом запросе.
    """
    header = (env.get(name) or "").strip()
    if not header:
        return None
    if not _HEADER_VALUE_RE.fullmatch(header):
        raise ConfigError(f"{name}: допустимы только печатные ASCII-символы, получено {header!r}")
    return header


def _env_float(env: Mapping[str, str], name: str, default: float, *, lo: float, hi: float) -> float:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        number = float(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name}: ожидается число, получено {raw!r}") from exc
    if not lo <= number <= hi:
        raise ConfigError(
            f"{name}: значение должно быть в диапазоне [{lo}; {hi}], получено {number}",
        )
    return number


@dataclass(frozen=True)
class Config:
    """Сводная конфигурация приложения. Неизменяема после загрузки.

    Состояние, меняющееся во время работы (например, текущая модель после
    ``/model``), хранится в агенте, а не здесь.
    """

    # --- API ---
    # Провайдер: "api" — API v2 с авторизацией по ключу или сертификату (см. ниже);
    # "gigacode" — бэкенд GigaCode: токен из файла его официального CLI, модели из
    # каталога бэкенда.
    provider: str = "api"
    # Две схемы авторизации API v2 (выбирается автоматически по наличию реквизитов):
    #   * OAuth: задан access_key -> Basic-ключ обменивается на Bearer-токен на
    #     auth_url, запросы идут с заголовком Authorization;
    #   * mTLS: заданы cert+key -> клиентский сертификат предъявляется на
    #     TLS-уровне, токен и заголовок Authorization не нужны.
    # Адреса и модель значений по умолчанию не имеют: их задаёт пользователь.
    access_key: str | None = None
    cert: str | None = None  # путь к клиентскому сертификату (mTLS, PEM)
    key: str | None = None  # путь к приватному ключу (mTLS)
    scope: str | None = None  # scope OAuth-токена; не задан — не передаётся
    model: str = ""
    auth_url: str | None = None
    base_url: str = ""
    # Проверка TLS-сертификата сервера: включена по умолчанию у всех провайдеров — без неё
    # ключ, токен и сам код ушли бы любому, кто подменит сервер. Выключает её только
    # явный GIGACREW_VERIFY_SSL=0.
    verify_ssl: bool = True
    # Корневой сертификат для проверки сервера, чей корень не в стандартном хранилище.
    # У API v2 — вместо стандартных корней, у бэкенда GigaCode — в дополнение к ним.
    ca_bundle: str | None = None
    # User-Agent всех запросов к бэкенду GigaCode; значения по умолчанию тоже нет.
    user_agent: str | None = None
    # Каталог официального CLI GigaCode (GIGACODE_HOME, иначе ~/.gigacode): оттуда
    # читается токен бэкенда (gigacode_token.json).
    gigacode_home: str | None = None
    timeout: int = _DEFAULT_TIMEOUT
    # Ожидание ответа модели в запросах чата, с (не меньше timeout): вызовы инструментов
    # приходят целиком в конце ответа, и пока модель пишет длинный аргумент (план, файл),
    # байты не идут — обычный timeout оборвал бы такой ответ.
    read_timeout: int = _DEFAULT_READ_TIMEOUT
    # Сколько ждать при 429 (лимит запросов шлюза) на один запрос, с; 0 — не ждать.
    rate_limit_wait: int = _DEFAULT_RATE_LIMIT_WAIT

    # --- Агент / окружение ---
    project_root: Path = field(default_factory=Path.cwd)
    auto_approve: bool = False  # пропускать подтверждения, кроме опасных операций (-y)
    yes_all: bool = False  # пропускать и подтверждения опасных операций (--yes-all)
    # Команды, которые классификатор признал «только чтение» (ls, grep, git log, python
    # без записи), — без вопроса в любом режиме; False — каждая команда с подтверждением.
    safe_commands: bool = True
    # Ответ «всегда» на вопрос о команде сохраняется для проекта в ~/.gigacode/projects/
    # (переживает сессии); False — только до конца сессии.
    remember_approvals: bool = True
    # Начальный режим разрешений (авто / план); в работе режим меняется в агенте
    # (Shift+Tab, /mode).
    mode: PermissionMode = PermissionMode.AUTO
    # Модель-рецензент изменяющих команд в режиме «авто»; None — модель чата (та, что
    # выбрана сейчас, в том числе после /model).
    reviewer_model: str | None = None
    # Сколько ждать ответа рецензента, с: не дождались — команду подтверждает пользователь.
    reviewer_timeout: int = _DEFAULT_REVIEWER_TIMEOUT
    max_steps: int = 50  # предохранитель агентного цикла
    max_tool_failures: int = 4  # стоп при N неудачных вызовах подряд (анти-залипание)
    # Одинаковый вызов без изменений между повторами: на N-м — предупреждение
    # модели, следующий не выполняется и ход останавливается (анти-зацикливание).
    max_tool_repeats: int = 3
    temperature: float = _DEFAULT_TEMPERATURE
    stream: bool = True  # потоковый (посимвольный) вывод ответа модели
    # Возвращать ли модели её рассуждения: "turn" — в пределах хода (до следующего
    # запроса пользователя), как советуют вендоры моделей с рассуждениями; "off" — нет.
    # Модели без рассуждений это не касается: возвращать нечего, запрос тот же.
    reasoning: str = "turn"
    # Бюджет контекста (оценка в токенах), в который укладывается история при
    # отправке модели; старые сообщения сверх бюджета отбрасываются. None — от
    # замеренного окна модели (~/.gigacode/models.json), число — явное переопределение.
    context_budget_tokens: int | None = None
    # Сжимать историю в краткое содержание, когда она занимает compact_threshold
    # бюджета истории (бюджет за вычетом системного промпта и схем инструментов).
    auto_compact: bool = True
    compact_threshold: float = 0.8
    # Сохранять чаты в ~/.gigacode/projects/<ключ>/chats/ (продолжение — /resume, -c).
    save_chats: bool = True
    # Проект недоверенный (--untrusted-project, ревью чужих изменений): его файлы и
    # инструкции — данные, а не указания, для основного агента и суб-агентов.
    untrusted_project: bool = False
    # Предлагать замер окна незамеренной модели (пробы оплачиваются); False — только
    # по --test-context.
    auto_measure: bool = True
    # Точная навигация по Python-коду через ty (LSP-сервер, запускается лениво);
    # False — только индекс проекта.
    ty: bool = True
    # Ограничители одного запуска суб-агента (инструмент task): шаги, доля окна
    # контекста модели, которую может занять его контекст, и время, с. На исчерпании
    # суб-агент подводит итог.
    subagent_max_steps: int = _DEFAULT_SUBAGENT_MAX_STEPS
    subagent_context_share: float = 0.8
    subagent_timeout: int = _DEFAULT_SUBAGENT_TIMEOUT
    # Настройки безопасности, которые .env проекта задавал, но они пропущены:
    # (ключ, файл) — для предупреждения при запуске (:attr:`ignored_env_warning`).
    ignored_env: tuple[tuple[str, str], ...] = ()

    @classmethod
    # Параметры — только по имени: флаги командной строки поверх окружения и .env.
    def load(  # noqa: WPS211
        cls,
        *,
        project_root: Path | None = None,
        model: str | None = None,
        auto_approve: bool = False,
        yes_all: bool = False,
        mode: PermissionMode | None = None,
        stream: bool = True,
        save_chats: bool = True,
        untrusted_project: bool = False,
        environ: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        trusted_dotenv: bool = False,
    ) -> Config:
        """Собирает конфигурацию. Некорректные значения → :class:`ConfigError`.

        ``environ``/``cwd`` подменяются в тестах (по умолчанию — os.environ и Path.cwd()).
        ``trusted_dotenv`` — брать из .env проекта и настройки безопасности
        (:func:`collect_environment`).
        """
        root = Path(project_root or cwd or Path.cwd()).resolve()
        collected = collect_environment(
            root,
            environ=environ,
            cwd=cwd,
            trusted_dotenv=trusted_dotenv,
        )
        env = collected.variables

        access_key = env.get("GIGACREW_ACCESS_KEY") or None
        cert = env.get("GIGACREW_CERT") or None
        key = env.get("GIGACREW_KEY") or None

        base_url = env.get("GIGACREW_API_URL") or ""
        ca_bundle = env.get("GIGACREW_CA_BUNDLE") or None
        if ca_bundle and not Path(ca_bundle).expanduser().is_file():
            raise ConfigError(f"GIGACREW_CA_BUNDLE: файл не найден: {ca_bundle}")

        provider = _env_choice(env, "GIGACREW_PROVIDER", ("api", "gigacode"))
        return cls(
            provider=provider,
            access_key=access_key,
            cert=cert,
            key=key,
            scope=env.get("GIGACREW_SCOPE") or None,
            model=model or env.get("GIGACREW_MODEL") or "",
            auth_url=env.get("GIGACREW_AUTH_URL") or None,
            base_url=base_url.rstrip("/"),
            verify_ssl=_env_bool(env, "GIGACREW_VERIFY_SSL", default=True),
            ca_bundle=str(Path(ca_bundle).expanduser()) if ca_bundle else None,
            user_agent=_env_header(env, "GIGACREW_USER_AGENT"),
            gigacode_home=env.get("GIGACODE_HOME") or None,
            timeout=_env_int(env, "GIGACREW_TIMEOUT", _DEFAULT_TIMEOUT),
            read_timeout=_env_int(env, "GIGACREW_READ_TIMEOUT", _DEFAULT_READ_TIMEOUT),
            rate_limit_wait=_env_int(
                env,
                "GIGACREW_RATE_LIMIT_WAIT",
                _DEFAULT_RATE_LIMIT_WAIT,
                minimum=0,
            ),
            project_root=root,
            auto_approve=auto_approve or yes_all,
            yes_all=yes_all,
            mode=mode or _env_mode(env, "GIGACREW_MODE"),
            stream=stream,
            temperature=_env_float(
                env,
                "GIGACREW_TEMPERATURE",
                _DEFAULT_TEMPERATURE,
                lo=_MIN_TEMPERATURE,
                hi=_MAX_TEMPERATURE,
            ),
            reasoning=_env_choice(env, "GIGACREW_REASONING", ("turn", "off")),
            context_budget_tokens=_env_optional_int(
                env,
                "GIGACREW_CONTEXT_TOKENS",
                minimum=_MIN_CONTEXT_TOKENS,
            ),
            auto_compact=_env_bool(env, "GIGACREW_AUTO_COMPACT", default=True),
            compact_threshold=_env_int(
                env,
                "GIGACREW_COMPACT_THRESHOLD",
                _DEFAULT_SHARE_PERCENT,
                minimum=_MIN_SHARE_PERCENT,
                maximum=_MAX_SHARE_PERCENT,
            )
            / _PERCENT,
            save_chats=save_chats and _env_bool(env, "GIGACREW_SAVE_CHATS", default=True),
            untrusted_project=untrusted_project,
            auto_measure=_env_bool(env, "GIGACREW_AUTO_MEASURE", default=True),
            ty=_env_bool(env, "GIGACREW_TY", default=True),
            safe_commands=_env_bool(env, "GIGACREW_SAFE_COMMANDS", default=True),
            remember_approvals=_env_bool(env, "GIGACREW_REMEMBER_APPROVALS", default=True),
            reviewer_model=(env.get("GIGACREW_REVIEWER_MODEL") or "").strip() or None,
            reviewer_timeout=_env_int(
                env,
                "GIGACREW_REVIEWER_TIMEOUT",
                _DEFAULT_REVIEWER_TIMEOUT,
            ),
            subagent_max_steps=_env_int(
                env,
                "GIGACREW_SUBAGENT_MAX_STEPS",
                _DEFAULT_SUBAGENT_MAX_STEPS,
            ),
            subagent_context_share=_env_int(
                env,
                "GIGACREW_SUBAGENT_CONTEXT",
                _DEFAULT_SHARE_PERCENT,
                minimum=_MIN_SHARE_PERCENT,
                maximum=_MAX_SHARE_PERCENT,
            )
            / _PERCENT,
            subagent_timeout=_env_int(
                env,
                "GIGACREW_SUBAGENT_TIMEOUT",
                _DEFAULT_SUBAGENT_TIMEOUT,
                minimum=_MIN_SUBAGENT_TIMEOUT,
            ),
            ignored_env=collected.ignored,
        )

    @property
    def ignored_env_warning(self) -> str | None:
        """Предупреждение о пропущенных настройках безопасности из .env проекта."""
        if not self.ignored_env:
            return None
        keys = ", ".join(dict.fromkeys(key for key, _ in self.ignored_env))
        files = ", ".join(dict.fromkeys(path for _, path in self.ignored_env))
        return (
            f"в {files} не действуют настройки безопасности: {keys} — "
            "их может прислать репозиторий. Задайте их в переменных окружения или в "
            f"{USER_ENV_HINT} (файл пользователя, общий для всех проектов)"
        )

    @property
    def auth_mode(self) -> str:
        """Схема авторизации, выбранная по провайдеру и наличию реквизитов.

        ``"cli_token"`` — бэкенд GigaCode (токен официального CLI); ``"oauth"`` — есть
        access_key; ``"mtls"`` — есть cert+key (и нет ключа); ``"none"`` — реквизитов нет.
        """
        if self.provider == "gigacode":
            return "cli_token"
        if self.access_key:
            return "oauth"
        if self.cert and self.key:
            return "mtls"
        return "none"

    @property
    def gigacode_dir(self) -> Path:
        """Каталог официального CLI GigaCode: ``GIGACODE_HOME``, иначе ``~/.gigacode``."""
        if self.gigacode_home:
            return Path(self.gigacode_home).expanduser()
        return Path.home() / ".gigacode"

    def build_ssl_verify(self) -> ssl.SSLContext | bool:
        """Значение для httpx ``verify=``.

        Для mTLS строит SSL-контекст с клиентским сертификатом (cert+key); файлы
        проверяются на существование и загружаемость. Для OAuth — контекст проверки
        сервера (:meth:`_server_ssl_context`) или False (GIGACREW_VERIFY_SSL=0). Для бэкенда
        GigaCode — контекст с корнями официального CLI (:meth:`_gigacode_ssl_context`).
        """
        if self.auth_mode != "mtls":
            if not self.verify_ssl:
                return False
            if self.provider == "gigacode":
                return self._gigacode_ssl_context()
            return self._server_ssl_context()

        cert_path = Path(self.cert)  # type: ignore[arg-type]
        key_path = Path(self.key)  # type: ignore[arg-type]
        if not cert_path.is_file():
            raise FileNotFoundError(f"Клиентский сертификат не найден: {cert_path}")
        if not key_path.is_file():
            raise FileNotFoundError(f"Приватный ключ не найден: {key_path}")

        if self.verify_ssl:
            context = self._server_ssl_context()
        else:
            # Явный GIGACREW_VERIFY_SSL=0 — например, самоподписанный серверный сертификат.
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        try:
            context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
        except ssl.SSLError as exc:
            raise RuntimeError(f"Не удалось загрузить клиентский сертификат: {exc}") from exc
        return context

    def require_credentials(self) -> None:
        """Проверяет, что заданы реквизиты и адреса API (значений по умолчанию нет)."""
        if self.provider == "gigacode":
            self._require_gigacode_settings()
            return
        if self.auth_mode == "none":
            raise RuntimeError(
                "Не заданы реквизиты API. Укажите либо GIGACREW_ACCESS_KEY (OAuth) в .env или "
                "переменных окружения, либо пару GIGACREW_CERT и GIGACREW_KEY (mTLS) в "
                "переменных окружения или ~/.gigacode/.env. См. .env.example.",
            )
        if not self.base_url:
            raise RuntimeError(
                "Не задан адрес API. Укажите GIGACREW_API_URL в переменных окружения или "
                "~/.gigacode/.env (не в .env проекта). См. .env.example.",
            )
        if self.auth_mode == "oauth" and not self.auth_url:
            raise RuntimeError(
                "Не задан адрес выдачи токена. Для схемы OAuth укажите GIGACREW_AUTH_URL "
                "в переменных окружения или ~/.gigacode/.env (не в .env проекта). "
                "См. .env.example.",
            )

    def _server_ssl_context(self) -> ssl.SSLContext:
        """Проверка сервера API v2: корни из ``ca_bundle``, если он задан, иначе стандартные.

        Стандартные — как у httpx (certifi или SSL_CERT_FILE/SSL_CERT_DIR) и системное
        хранилище (туда ставят корпоративные корни).
        """
        if self.ca_bundle:
            try:
                return ssl.create_default_context(cafile=self.ca_bundle)
            except (OSError, ssl.SSLError) as exc:
                raise RuntimeError(
                    f"не удалось загрузить GIGACREW_CA_BUNDLE {self.ca_bundle}: {exc}",
                ) from exc
        context = httpx.create_ssl_context()
        context.load_default_certs()
        return context

    def _gigacode_ssl_context(self) -> ssl.SSLContext:
        """Проверка сервера GigaCode с теми же корнями, что у официального CLI.

        Корни: стандартные (как у httpx), его сертификаты — ``certs/*.pem`` и ``*.crt`` в
        :attr:`gigacode_dir` (туда CLI кладёт свой корпоративный корень) — и ``ca_bundle``.
        Файл, который не читается как сертификат, — ошибка, а не тихий пропуск.
        """
        context = httpx.create_ssl_context()
        certs = self.gigacode_dir / "certs"
        candidates: Iterable[Path] = certs.iterdir() if certs.is_dir() else ()
        files = sorted(
            path
            for path in candidates
            if path.suffix.lower() in {".pem", ".crt"} and path.is_file()
        )
        if self.ca_bundle:
            files.append(Path(self.ca_bundle))
        for path in files:
            try:
                context.load_verify_locations(cafile=str(path))
            except (OSError, ssl.SSLError) as exc:
                raise RuntimeError(
                    f"не удалось загрузить корневой сертификат {path}: {exc}",
                ) from exc
        return context

    def _require_gigacode_settings(self) -> None:
        """Проверяет адрес бэкенда GigaCode и User-Agent (значений по умолчанию нет)."""
        if not self.base_url:
            raise RuntimeError(
                "Не задан адрес бэкенда GigaCode. Укажите GIGACREW_API_URL в переменных "
                "окружения или ~/.gigacode/.env (не в .env проекта). См. .env.example.",
            )
        if not self.user_agent:
            raise RuntimeError(
                "Не задан User-Agent для запросов к GigaCode. Укажите GIGACREW_USER_AGENT "
                "в .env или переменных окружения. См. .env.example.",
            )
