"""Перевод между MCP и function calling: имена инструментов, схемы параметров, результаты.

* Имя инструмента для модели — ``mcp__<сервер>__<инструмент>`` (как у Claude Code): из
  латинских букв, цифр, ``_`` и ``-``, не длиннее 64 символов.
* Схема параметров сервера — произвольная JSON Schema, а API принимает упрощённую:
  у каждого свойства одиночный ``type``, без ``anyOf``/``oneOf``/``allOf`` и ссылок
  ``$ref``. :func:`tool_schema` сворачивает схему к такой (проверку аргументов всё
  равно делает сервер).
* Результат вызова — список блоков (текст, изображения, ресурсы): модели уходит текст,
  остальное — короткой пометкой.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Final

from gigacrew.tools.process import truncate_middle

__all__ = [
    "MAX_RESULT_CHARS",
    "MCP_PREFIX",
    "render_result",
    "tool_name",
    "tool_schema",
]

MCP_PREFIX: Final = "mcp__"
MAX_NAME = 64  # длина имени функции, которую принимают API function calling
MAX_RESULT_CHARS = 30_000  # как у вывода run_shell
_HASH_LENGTH = 8
_MAX_DEPTH = 8  # вложенность схемы: глубже — объект без описания свойств
_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")
_JSON_TYPES = frozenset(("string", "number", "integer", "boolean", "array", "object"))
# Ключи JSON Schema, которые нужны свёртке.
_TYPE: Final = "type"
_OBJECT: Final = "object"
_PROPERTIES: Final = "properties"
_REQUIRED: Final = "required"
_DESCRIPTION: Final = "description"


def tool_name(server: str, tool: str) -> str:
    """Имя инструмента для модели: ``mcp__<сервер>__<инструмент>``.

    Недопустимые символы заменяются на ``_``; длинное имя укорачивается, а хвост
    из хеша полного имени сохраняет различие инструментов с общим началом.
    """
    name = _UNSAFE_NAME_RE.sub("_", f"{MCP_PREFIX}{server}__{tool}")
    if len(name) <= MAX_NAME:
        return name
    digest = hashlib.sha256(f"{server}\0{tool}".encode()).hexdigest()[:_HASH_LENGTH]
    head = name[: MAX_NAME - _HASH_LENGTH - 1]
    return f"{head}_{digest}"


def tool_schema(schema: object) -> dict[str, Any]:
    """Схема параметров инструмента в формате function calling API.

    Верхний уровень — всегда объект. Ссылки ``$ref`` внутри схемы (JSON Pointer:
    ``#/$defs/A``, ``#/definitions/A``, ``#/properties/x``) раскрываются, а рекурсивная —
    один раз на пути: дальше на её месте только тип, без свойств (иначе схема росла бы
    экспоненциально). Варианты ``anyOf``/``oneOf`` сворачиваются к первому не-``null``,
    ``allOf`` — объединяется. Остаются только ``type``, ``description``, ``properties``,
    ``required``, ``items`` и ``enum``.
    """
    if not isinstance(schema, dict):
        return {_TYPE: _OBJECT, _PROPERTIES: {}}
    converted = _convert(schema, _Refs(schema), 0)
    if converted[_TYPE] != _OBJECT:
        return {_TYPE: _OBJECT, _PROPERTIES: {}}
    converted.setdefault(_PROPERTIES, {})
    return converted


@dataclass(frozen=True)
class _Refs:
    """Корень схемы (для ``$ref``) и ссылки, уже раскрытые на текущем пути."""

    root: dict[str, Any]
    seen: frozenset[str] = frozenset()

    def follow(self, ref: str) -> tuple[dict[str, Any], _Refs]:
        """Цель ссылки и ссылки пути с ней; уже раскрытая на пути — только её тип."""
        target = _pointer(self.root, ref)
        if target is None:
            return {}, self
        if ref in self.seen:
            shallow = {_TYPE: _type_of(target)}
            if isinstance(target.get(_DESCRIPTION), str):
                shallow[_DESCRIPTION] = target[_DESCRIPTION]
            return shallow, self
        return target, _Refs(self.root, self.seen | {ref})

    def join(self, other: _Refs) -> _Refs:
        """Ссылки обоих путей (части ``allOf``)."""
        return _Refs(self.root, self.seen | other.seen)


def _pointer(root: dict[str, Any], ref: str) -> dict[str, Any] | None:
    """Цель ``$ref`` внутри схемы по JSON Pointer; внешняя ссылка или нет цели — None."""
    if not ref.startswith("#"):
        return None  # внешние документы не загружаются
    node: object = root
    for raw in ref[1:].split("/")[1:]:
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            node = node.get(key)
        elif isinstance(node, list) and key.isdigit() and int(key) < len(node):
            node = node[int(key)]
        else:
            return None
    return node if isinstance(node, dict) else None


def _convert(schema: dict[str, Any], refs: _Refs, depth: int) -> dict[str, Any]:
    """Один уровень схемы → упрощённая схема (см. :func:`tool_schema`)."""
    merged, refs = _resolve(schema, refs, depth)
    kind = _type_of(merged)
    converted: dict[str, Any] = {_TYPE: kind}
    description = merged.get(_DESCRIPTION)
    if isinstance(description, str) and description:
        converted[_DESCRIPTION] = description
    enum = merged.get("enum")
    if isinstance(enum, list):
        options = [option for option in enum if option is not None]
        if options:
            converted["enum"] = options
    if kind == "array":
        element = merged.get("items")
        if isinstance(element, dict) and depth < _MAX_DEPTH:
            converted["items"] = _convert(element, refs, depth + 1)
        else:
            converted["items"] = {_TYPE: "string"}
    if kind == _OBJECT:
        converted.update(_object_fields(merged, refs, depth))
    return converted


def _object_fields(schema: dict[str, Any], refs: _Refs, depth: int) -> dict[str, Any]:
    """``properties`` и ``required`` объекта (только из его свойств)."""
    raw = schema.get(_PROPERTIES)
    if not isinstance(raw, dict) or depth >= _MAX_DEPTH:
        return {}
    props = {
        str(name): _convert(prop, refs, depth + 1)
        for name, prop in raw.items()
        if isinstance(prop, dict)
    }
    fields: dict[str, Any] = {_PROPERTIES: props}
    required = schema.get(_REQUIRED)
    if isinstance(required, list):
        listed = [name for name in required if isinstance(name, str) and name in props]
        if listed:
            fields[_REQUIRED] = listed
    return fields


def _resolve(schema: dict[str, Any], refs: _Refs, depth: int) -> tuple[dict[str, Any], _Refs]:
    """Схема без ``$ref``, ``anyOf``/``oneOf`` и ``allOf`` (их поля — в одном объекте).

    Второе — ссылки пути с раскрытыми здесь: по ним свойства не раскроют их снова.
    """
    merged = dict(schema)
    for _ in range(_MAX_DEPTH):
        unfolded = _unfold(merged, refs, depth)
        if unfolded is None:
            break
        merged, refs = unfolded
    return merged, refs


def _unfold(
    schema: dict[str, Any],
    refs: _Refs,
    depth: int,
) -> tuple[dict[str, Any], _Refs] | None:
    """Один шаг :func:`_resolve`; None — раскрывать нечего."""
    merged = dict(schema)
    ref = merged.pop("$ref", None)
    variants = merged.pop("anyOf", None) or merged.pop("oneOf", None)
    parts = merged.pop("allOf", None)
    if ref is None and variants is None and parts is None:
        return None
    if isinstance(ref, str) and depth < _MAX_DEPTH:
        target, refs = refs.follow(ref)
        merged = {**target, **merged}  # описание поля важнее описания определения
    if isinstance(variants, list):
        merged = {**_first_variant(variants), **merged}
    if isinstance(parts, list):
        merged, refs = _merge_all(parts, merged, refs, depth)
    return merged, refs


def _first_variant(variants: list[Any]) -> dict[str, Any]:
    """Первый вариант union-типа, кроме ``null``."""
    candidates = [variant for variant in variants if isinstance(variant, dict)]
    for variant in candidates:
        if variant.get(_TYPE) != "null":
            return variant
    return candidates[0] if candidates else {}


def _merge_all(
    parts: list[Any],
    base: dict[str, Any],
    refs: _Refs,
    depth: int,
) -> tuple[dict[str, Any], _Refs]:
    """``allOf``: свойства и обязательные поля всех частей — в одном объекте.

    Каждая часть сначала раскрывается (``$ref``, вложенный ``allOf``): иначе поля второй
    и следующих ссылок потерялись бы.
    """
    merged = dict(base)
    props: dict[str, Any] = dict(merged.get(_PROPERTIES) or {})
    required: list[Any] = list(merged.get(_REQUIRED) or [])
    joined = refs
    for raw in parts:
        if not isinstance(raw, dict):
            continue
        part, part_refs = _resolve(raw, refs, depth + 1)
        joined = joined.join(part_refs)
        props.update(part.get(_PROPERTIES) or {})
        required.extend(part.get(_REQUIRED) or [])
        for key, setting in part.items():
            merged.setdefault(key, setting)
    if props:
        merged[_PROPERTIES] = props
    if required:
        merged[_REQUIRED] = required
    return merged, joined


def _type_of(schema: dict[str, Any]) -> str:
    """Одиночный тип: из ``type`` (список — первый не-null) или по виду схемы."""
    declared = schema.get(_TYPE)
    if isinstance(declared, list):
        declared = next((kind for kind in declared if kind != "null"), None)
    if isinstance(declared, str) and declared in _JSON_TYPES:
        return declared
    if _PROPERTIES in schema:
        return _OBJECT
    if "items" in schema:
        return "array"
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return _enum_type(enum)
    return "string"


def _enum_type(options: list[Any]) -> str:
    """Тип значений перечисления (по первому не-null)."""
    first = next((option for option in options if option is not None), "")
    if isinstance(first, bool):
        return "boolean"
    if isinstance(first, int):
        return "integer"
    if isinstance(first, float):
        return "number"
    return "string"


def render_result(reply: dict[str, Any]) -> tuple[str, bool]:
    """Результат ``tools/call`` → (текст для модели, ошибка ли это).

    Текстовые блоки — как есть; изображения и звук модель не видит — пометка с типом и
    размером; встроенный ресурс — его текст, ссылка на ресурс — адрес. Нет блоков —
    ``structuredContent`` в JSON. Длинный текст укорачивается посередине.
    """
    blocks = reply.get("content")
    parts = [_render_block(block) for block in blocks or () if isinstance(block, dict)]
    text = "\n\n".join(part for part in parts if part)
    structured = reply.get("structuredContent")
    if not text and structured is not None:
        text = json.dumps(structured, ensure_ascii=False, indent=2, default=str)
    failed = reply.get("isError") is True
    return truncate_middle(text or "(пустой ответ)", MAX_RESULT_CHARS), failed


def _render_block(block: dict[str, Any]) -> str:
    kind = block.get(_TYPE)
    if kind == "text":
        text = block.get("text")
        return text if isinstance(text, str) else ""
    if kind in {"image", "audio"}:
        mime = block.get("mimeType") or kind
        encoded = block.get("data")
        size = len(encoded) * 3 // 4 if isinstance(encoded, str) else 0
        return f"[{mime}, около {size} байт — модель его не видит]"
    if kind == "resource":
        return _render_resource(block.get("resource"))
    if kind == "resource_link":
        uri = block.get("uri") or ""
        about = [str(block[key]) for key in ("name", "description") if block.get(key)]
        joined = " — ".join(about)
        return f"[ресурс {uri}: {joined}]" if about else f"[ресурс {uri}]"
    return f"[блок {kind!r} не поддерживается]"


def _render_resource(resource: object) -> str:
    if not isinstance(resource, dict):
        return ""
    uri = resource.get("uri") or ""
    text = resource.get("text")
    if isinstance(text, str):
        return f"[ресурс {uri}]\n{text}"
    mime = resource.get("mimeType") or "двоичные данные"
    return f"[ресурс {uri}: {mime} — модель его не видит]"
