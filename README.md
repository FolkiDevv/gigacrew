# gigacrew

Консольная утилита для работы с кодом локального проекта: навигация по исходникам, поиск, правка файлов и запуск команд в терминале.
Перед запуском скопируйте `.env.example` в `.env` и укажите в нём адрес API, модель и реквизиты доступа — значений по умолчанию нет.

## Установка

```sh
uv tool install gigacrew
# или
pipx install gigacrew
```

## Выпуск новой версии

Публикация на PyPI идёт из GitHub Actions через trusted publishing — API-токены нигде не хранятся.

1. Поднимите версию в `pyproject.toml` (`project.version`) и в `gigacrew/__init__.py` (`__version__`) — они должны совпадать.
2. По желанию прогоните репетицию: Actions → «Publish to TestPyPI» → Run workflow, и проверьте страницу пакета на test.pypi.org. Номер версии на PyPI нельзя переиспользовать даже после удаления.
3. Создайте и запушьте тег: `git tag vX.Y.Z && git push origin vX.Y.Z`. Workflow `release.yml` прогонит CI, сверит тег с версией, соберёт пакет, проверит колесо и опубликует его.

## Лицензия

MIT, см. [LICENSE](LICENSE).
