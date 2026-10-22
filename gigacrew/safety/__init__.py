"""Классификатор команд оболочки по степени безопасности.

Цель — не спрашивать пользователя о командах, которые только читают проект, и сохранить
вопрос (с причиной) для всего остального. Работает локально, без сети и моделей:

1. разбор команды деревом tree-sitter-bash (:mod:`gigacrew.safety.bash`);
2. политика программ — по имени, подкоманде, флагам и путям (:mod:`gigacrew.safety.programs`),
   отказ по умолчанию: незнакомое — «не знаю»;
3. статический анализ Python-кода из ``python -c``, heredoc и скриптов
   (:mod:`gigacrew.safety.python`).

Итог — :class:`Verdict`: уровень (READ/EXEC/WRITE/DANGER) или «не знаю» с причиной,
рабочий каталог и команда без ведущих ``cd`` (ключ для памяти решений), оценки частей.

Классификатор выключен на Windows: там команды выполняет cmd.exe, а не ``/bin/sh``.
"""

from gigacrew.safety.classifier import MAX_COMMAND as MAX_COMMAND
from gigacrew.safety.classifier import classify as classify
from gigacrew.safety.classifier import invisible_chars as invisible_chars
from gigacrew.safety.levels import RULES_VERSION as RULES_VERSION
from gigacrew.safety.levels import CommandLevel as CommandLevel
from gigacrew.safety.levels import Part as Part
from gigacrew.safety.levels import Verdict as Verdict
