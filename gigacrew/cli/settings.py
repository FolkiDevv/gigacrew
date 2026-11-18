"""Команда ``/settings`` (``/config``): меню настроек пользователя и перенос из ``.env``.

* ``/settings`` — меню :mod:`gigacrew.ui.settings_picker`: правка
  ``~/.gigacode/gigacrew.json``; что можно, применяется сразу, остальное — со
  следующего запуска. Без терминала — список настроек.
* ``/settings import-env`` — перенести настройки поведения из ``~/.gigacode/.env`` в
  ``gigacrew.json`` (строки из ``.env`` удаляются; подключение и секреты остаются).

Переменные окружения и ``.env`` главнее файла: перекрытая настройка видна в меню вместе
с источником, а её правка в файле до удаления переменной не действует.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from gigacrew.agent.loop import Agent
from gigacrew.agent.reviewer import check_reviewer
from gigacrew.config import (
    USER_ENV_HINT,
    Config,
    env_sources,
    load_environment,
    read_dotenv,
    user_env_file,
)
from gigacrew.project import preferences
from gigacrew.ui.console import Console
from gigacrew.ui.settings_picker import Change, SettingRow, SettingsState, edit_settings

__all__ = ("IMPORT_ENV", "SETTINGS_USAGE", "SettingsSession", "connection_lines", "import_env")

IMPORT_ENV = "import-env"
SETTINGS_USAGE = f"/settings [{IMPORT_ENV}]"
_BY_ENV: Mapping[str, preferences.SettingSpec] = MappingProxyType({
    spec.env: spec for spec in preferences.SETTINGS
})
_PROVIDERS: Mapping[str, str] = MappingProxyType({"api": "API v2", "gigacode": "бэкенд GigaCode"})
_AUTH: Mapping[str, str] = MappingProxyType({
    "oauth": "OAuth-ключ (GIGACREW_ACCESS_KEY)",
    "mtls": "клиентский сертификат (mTLS)",
    "cli_token": "токен официального CLI GigaCode",
    "none": "реквизиты не заданы",
})
_NOT_SET = "не задан"
# Не удалось записать файл настроек: ошибка записи или нечитаемый файл.
_STORE_ERRORS = (OSError, TypeError, ValueError)


def settings_command(agent: Agent, ui: Console, arg: str, models: Sequence[str] | None) -> bool:
    """``/settings``: меню настроек; ``/settings import-env`` — перенос из ``.env``."""
    action = arg.strip().lower()
    if action == IMPORT_ENV:
        import_env(ui, agent.config.project_root)
        return True
    if action:
        ui.error(f"использование: {SETTINGS_USAGE}")
        return True
    session = SettingsSession(agent, models)
    if not ui.interactive():
        ui.info(session.describe())
        return True
    state = SettingsState(
        session.rows(),
        session.change,
        title=f"Настройки · {preferences.PREFERENCES_HINT}",
        connection=connection_lines(agent.config),
    )
    ui.stop_live()
    try:
        edit_settings(state, no_color=ui.no_color)
    except (KeyboardInterrupt, EOFError):
        ui.system("(меню настроек закрыто)")
    for notice in session.summary():
        ui.info(notice)
    return True


class SettingsSession:
    """Строки меню и применение изменений: запись в файл и, где можно, в агента."""

    def __init__(
        self,
        agent: Agent,
        models: Sequence[str] | None,
        *,
        path: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        """Сессия меню для ``agent``; ``path``/``environ`` подменяются в тестах."""
        self._agent = agent
        self._models = tuple(models or ())
        self._path = path or preferences.preferences_file()
        root = agent.config.project_root
        self._env = load_environment(root, environ=environ)
        self._sources = env_sources(root, environ=environ)
        self._restart: dict[str, str] = {}  # ключ → подпись: меняли, действует при запуске

    def rows(self) -> list[SettingRow]:
        """Строки меню: действующее значение, задано ли в файле, кто его перекрывает."""
        saved = preferences.load_preferences(self._path).settings
        rows = []
        for spec in preferences.SETTINGS:
            source = self._sources.get(spec.env)
            stored = spec.key in saved
            setting = saved.get(spec.key, spec.default)
            if source is not None:
                setting = _env_setting(spec, self._env.get(spec.env, ""))
            choices = self._models if spec.kind is preferences.SettingKind.MODEL else ()
            rows.append(SettingRow(spec, setting, stored=stored, source=source, choices=choices))
        return rows

    def change(self, row: SettingRow, setting: preferences.Setting) -> Change:
        """Сохранить новое значение и применить его, если оно действует сразу."""
        spec = row.spec
        if spec.kind is preferences.SettingKind.MODEL and setting and self._models:
            # Как /model: имена моделей регистрозависимы; неизвестная модель по умолчанию
            # не дала бы gigacrew запуститься.
            known = {name.lower(): name for name in self._models}
            setting = known.get(str(setting).lower())
            if setting is None:
                names = ", ".join(self._models)
                return Change(row, f"такой модели нет; доступны: {names}", failed=True)
        try:
            preferences.store_setting(spec, setting, self._path)
        except _STORE_ERRORS as error:
            return Change(row, f"не сохранено: {error}", failed=True)
        stored = setting != spec.default
        if row.source is not None:
            message = f"сохранено, но действует {spec.env} ({row.source})"
            return Change(replace(row, stored=stored), message)
        updated = replace(row, setting=setting, stored=stored)
        if spec.restart:
            self._restart[spec.key] = spec.label
            return Change(updated, "сохранено · действует со следующего запуска")
        return Change(updated, self._apply(spec, setting))

    def summary(self) -> list[str]:
        """Итог после меню: что действует только со следующего запуска, что есть в .env."""
        notices = []
        if self._restart:
            labels = ", ".join(self._restart.values())
            notices.append(f"со следующего запуска gigacrew: {labels}")
        movable = _movable(read_dotenv(user_env_file()))
        if movable:
            notices.append(
                f"в {USER_ENV_HINT} задано настроек: {len(movable)} — они главнее "
                f"{preferences.PREFERENCES_HINT}; /settings {IMPORT_ENV} перенесёт их в "
                "файл настроек",
            )
        return notices

    def describe(self) -> str:
        """Список настроек текстом (без терминала): раздел, подпись, значение, источник."""
        lines = [f"настройки ({preferences.PREFERENCES_HINT}; меню — /settings в терминале):"]
        section = ""
        for row in self.rows():
            if row.spec.section != section:
                section = row.spec.section
                lines.append(f"  {preferences.SECTIONS[section]}")
            shown = row.spec.show(row.setting)
            source = f" ({row.source})" if row.source else ""
            label = row.spec.label
            lines.append(f"    {label}: {shown}{source}")
        return "\n".join(lines)

    def _apply(self, spec: preferences.SettingSpec, setting: preferences.Setting) -> str:
        """Применить к агенту; текст итога (с предупреждением о рецензенте)."""
        agent = self._agent
        config: Config = agent.config.with_setting(spec, setting)
        message = "сохранено · действует сразу"
        if spec.field != "reviewer_model":
            agent.apply_config(config)
            return message
        reviewer, warning = check_reviewer(agent.provider, config.reviewer_model)
        agent.apply_config(config, reviewer)
        return message if warning is None else f"{message}; {warning}"


def _env_setting(spec: preferences.SettingSpec, raw: str) -> preferences.Setting:
    """Значение переменной (её уже проверил запуск; .env с тех пор мог измениться)."""
    try:
        return spec.parse(raw)
    except ValueError:
        return spec.default


def _movable(entries: Mapping[str, str]) -> dict[str, str]:
    """Переменные поведения из ``.env``, которые можно перенести в файл настроек.

    Пустые не в счёт: они ничего не перекрывают, а перенос сбросил бы значение в файле.
    """
    return {key: raw for key, raw in entries.items() if key in _BY_ENV and raw.strip()}


def connection_lines(config: Config) -> list[tuple[str, str]]:
    """Подключение для меню (только просмотр): провайдер, адрес, авторизация, TLS."""
    lines = [
        ("провайдер", _PROVIDERS.get(config.provider, config.provider)),
        ("адрес", config.base_url or _NOT_SET),
        ("авторизация", _AUTH.get(config.auth_mode, config.auth_mode)),
    ]
    tls = "проверка сертификата сервера" if config.verify_ssl else "проверка ВЫКЛЮЧЕНА"
    if config.ca_bundle:
        tls = f"{tls}, корни: {config.ca_bundle}"
    lines.append(("TLS", tls))
    if config.provider == "gigacode":
        lines.append(("User-Agent", config.user_agent or _NOT_SET))
    return lines


def import_env(
    ui: Console,
    root: Path,
    *,
    env_file: Path | None = None,
    path: Path | None = None,
) -> None:
    """Перенести настройки поведения из ``~/.gigacode/.env`` в ``gigacrew.json``.

    Перенесённые строки удаляются из ``.env``; значение с ошибкой остаётся там
    (предупреждение). Подключение и секреты не трогаются. ``.env`` проекта ``root`` —
    тоже, а то, что он задаёт, остаётся и в ``~/.gigacode/.env``: файл настроек младше
    ``.env`` проекта, и после переноса действовало бы значение проекта.
    """
    env_file = env_file or user_env_file()
    movable = _movable(read_dotenv(env_file))
    for key in _movable(read_dotenv(root / ".env")).keys() & movable.keys():
        ui.warn(f"{key} остаётся в {USER_ENV_HINT}: её задаёт и .env проекта")
        movable.pop(key)
    if not movable:
        ui.info(f"в {USER_ENV_HINT} нет настроек, которые можно перенести")
        return
    moved: list[str] = []
    for key, raw in movable.items():
        spec = _BY_ENV[key]
        try:
            preferences.store_setting(spec, spec.parse(raw), path)
        except preferences.INVALID_SETTING as error:
            ui.warn(f"{key} остаётся в {USER_ENV_HINT}: {error}")
            continue
        except OSError as error:
            ui.error(f"не удалось записать {preferences.PREFERENCES_HINT}: {error}")
            break
        moved.append(key)
    if not moved:
        return
    try:
        _drop_lines(env_file, frozenset(moved))
    except (OSError, UnicodeDecodeError) as error:
        ui.error(f"перенесено, но не удалось убрать строки из {USER_ENV_HINT}: {error}")
        return
    names = ", ".join(moved)
    ui.success(
        f"перенесено в {preferences.PREFERENCES_HINT}: {names}; строки убраны из {USER_ENV_HINT}",
    )


def _drop_lines(env_file: Path, keys: frozenset[str]) -> None:
    """Удаляет из ``.env`` строки с переменными ``keys``; комментарии и прочее — как были."""
    lines = env_file.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    kept = [line for line in lines if _dotenv_key(line) not in keys]
    preferences.write_text_atomic(env_file, "".join(kept))


def _dotenv_key(line: str) -> str:
    """Имя переменной в строке ``.env`` (как у :func:`gigacrew.config.read_dotenv`)."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return ""
    key = stripped.partition("=")[0].strip()
    return key.removeprefix("export ").strip()
