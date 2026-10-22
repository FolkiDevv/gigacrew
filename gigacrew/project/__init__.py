"""Знание о проекте: рабочая папка .gigacode, файлы проекта, инструкции, индекс.

Слой не зависит от LLM и UI — на него опираются инструменты и ядро агента.
"""

from gigacrew.project.files import IGNORE_DIRS as IGNORE_DIRS
from gigacrew.project.files import build_file_tree as build_file_tree
from gigacrew.project.files import glob_match as glob_match
from gigacrew.project.files import is_excluded as is_excluded
from gigacrew.project.files import walk_files as walk_files
from gigacrew.project.index import INDEX_ERRORS as INDEX_ERRORS
from gigacrew.project.index import ProjectIndex as ProjectIndex
from gigacrew.project.instructions import INSTRUCTION_FILES as INSTRUCTION_FILES
from gigacrew.project.instructions import InstructionFile as InstructionFile
from gigacrew.project.instructions import NestedInstructions as NestedInstructions
from gigacrew.project.instructions import load_instructions as load_instructions
from gigacrew.project.symbols import Symbol as Symbol
from gigacrew.project.symbols import extract_symbols as extract_symbols
from gigacrew.project.symbols import language_of as language_of
from gigacrew.project.workspace import DATA_DIR_NAME as DATA_DIR_NAME
from gigacrew.project.workspace import Workspace as Workspace
