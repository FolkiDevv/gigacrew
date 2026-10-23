"""Тесты генерации JSON-schema инструментов под формат API.

API не принимает union-типы (anyOf/oneOf), которые pydantic генерирует
для Optional-полей. Проверяем, что все схемы инструментов нормализованы:
каждое свойство имеет одиночный ``type`` и не содержит anyOf/oneOf/$defs.
"""

from __future__ import annotations

from typing import Any

from gigacrew.tools.base import build_default_registry


def test_all_tool_schemas_are_api_compatible() -> None:
    reg = build_default_registry()
    specs = reg.specs()
    assert specs, "реестр пуст"
    for spec in specs:
        params = spec.parameters
        assert params.get("type") == "object", spec.name
        assert "$defs" not in params, f"{spec.name}: остались $defs"
        for prop_name, prop in params.get("properties", {}).items():
            where = f"{spec.name}.{prop_name}"
            assert "anyOf" not in prop, f"{where}: остался anyOf"
            assert "oneOf" not in prop, f"{where}: остался oneOf"
            assert "type" in prop, f"{where}: нет одиночного type"
            assert prop["type"] != "null", where


_FORBIDDEN_KEYS = ("anyOf", "oneOf", "allOf", "$ref", "$defs", "title", "default")


def _walk_schema(schema: dict[str, Any], where: str) -> None:
    """Проверяет каждый уровень схемы: свойства и элементы массивов.

    Без union-типов, ссылок и служебных ключей; у объекта — type object и required
    только из его свойств.
    """
    for key in _FORBIDDEN_KEYS:
        assert key not in schema, f"{where}: остался {key}"
    no_single_type = f"{where}: нет одиночного type"
    assert "type" in schema, no_single_type
    assert schema["type"] != "null", no_single_type
    if "properties" in schema:
        _walk_properties(schema, where)
    if isinstance(schema.get("items"), dict):
        _walk_schema(schema["items"], f"{where}[]")


def _walk_properties(schema: dict[str, Any], where: str) -> None:
    """Объект: type object, required только из его свойств, каждое свойство — рекурсивно."""
    assert schema["type"] == "object", where
    props = schema["properties"]
    assert set(schema.get("required", [])) <= set(props), f"{where}: required вне properties"
    for name, prop in props.items():
        _walk_schema(prop, f"{where}.{name}")


def test_all_tool_schemas_are_normalized_at_every_depth() -> None:
    for spec in build_default_registry().specs():
        _walk_schema(spec.parameters, spec.name)


def test_optional_field_collapsed_to_type() -> None:
    reg = build_default_registry()
    read = reg.get("read_file")
    assert read is not None
    schema = read.spec().parameters
    # Optional[int] -> просто integer, без null
    assert schema["properties"]["start_line"]["type"] == "integer"
    # обязательное поле осталось в required, необязательные — нет
    assert "path" in schema["required"]
    assert "start_line" not in schema.get("required", [])


def test_array_field_schema() -> None:
    reg = build_default_registry()
    git = reg.get("git")
    assert git is not None
    args_schema = git.spec().parameters["properties"]["args"]
    assert args_schema["type"] == "array"
    assert args_schema["items"]["type"] == "string"


def test_nested_models_are_inlined() -> None:
    ask_user = build_default_registry().get("ask_user")
    assert ask_user is not None
    schema = ask_user.spec().parameters
    assert "$ref" not in str(schema)
    assert "$defs" not in str(schema)
    question = schema["properties"]["questions"]["items"]
    assert question["type"] == "object"
    assert question["required"] == ["question", "options"]
    option = question["properties"]["options"]["items"]
    assert option["properties"]["label"] == {
        "description": "The answer option, short (1–5 words).",
        "type": "string",
    }
    assert "title" not in str(schema)
