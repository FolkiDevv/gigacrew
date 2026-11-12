"""MCP ↔ function calling: имена инструментов, свёртка схем, текст результатов."""

from __future__ import annotations

import json
from typing import Any

import pytest

from gigacrew.tools.mcp_schema import (
    MAX_RESULT_CHARS,
    MCP_PREFIX,
    render_result,
    tool_name,
    tool_schema,
)

_FORBIDDEN_KEYS = frozenset((
    "anyOf",
    "oneOf",
    "allOf",
    "$ref",
    "$defs",
    "definitions",
    "title",
    "default",
    "additionalProperties",
    "$schema",
))


def _check(schema: dict[str, Any], where: str = "схема") -> None:
    """Каждый уровень: одиночный type, без union-типов и ссылок, required — из свойств."""
    assert not _FORBIDDEN_KEYS & set(schema), where
    assert isinstance(schema["type"], str), where
    assert schema["type"] != "null", where
    props = schema.get("properties", {})
    assert set(schema.get("required", [])) <= set(props), where
    for name, prop in props.items():
        _check(prop, f"{where}.{name}")
    element = schema.get("items")
    if element is not None:
        _check(element, f"{where}[]")


def test_tool_name_sanitized() -> None:
    assert tool_name("github", "create_issue") == "mcp__github__create_issue"
    assert tool_name("fs", "read.file/v2") == "mcp__fs__read_file_v2"


def test_long_tool_names_stay_unique_and_short() -> None:
    stem = "x" * 80
    first = tool_name("server", f"{stem}a")
    second = tool_name("server", f"{stem}b")
    assert first != second
    assert len(first) == len(second) == 64
    assert first.startswith(MCP_PREFIX)


def test_schema_non_object_becomes_empty_object() -> None:
    assert tool_schema(None) == {"type": "object", "properties": {}}
    assert tool_schema({"type": "string"}) == {"type": "object", "properties": {}}


def test_schema_unions_refs_and_all_of_are_flattened() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "maybe": {"anyOf": [{"type": "null"}, {"type": "integer"}], "default": None},
            "kinds": {"type": ["null", "string"], "title": "Kinds"},
            "point": {"$ref": "#/$defs/Point", "description": "точка"},
            "legacy": {"$ref": "#/definitions/Legacy"},
            "tags": {"type": "array", "items": {"oneOf": [{"type": "string"}, {"type": "number"}]}},
            "mode": {"enum": ["a", "b", None]},
            "merged": {"allOf": [{"properties": {"x": {"type": "number"}}}, {"required": ["x"]}]},
            "missing": {"$ref": "#/$defs/Nope"},
        },
        "required": ["point", "ghost"],
        "$defs": {
            "Point": {
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x"],
                "description": "описание модели",
            },
        },
        "definitions": {"Legacy": {"type": "boolean"}},
    }
    converted = tool_schema(schema)
    _check(converted)
    props = converted["properties"]
    assert props["maybe"] == {"type": "integer"}
    assert props["kinds"] == {"type": "string"}
    assert props["point"]["description"] == "точка"
    assert props["point"]["required"] == ["x"]
    assert props["legacy"] == {"type": "boolean"}
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert props["mode"] == {"type": "string", "enum": ["a", "b"]}
    assert props["merged"]["required"] == ["x"]
    assert props["missing"] == {"type": "string"}
    assert converted["required"] == ["point"]


def test_recursive_schema_terminates() -> None:
    schema = {
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
                    "self": {"$ref": "#/$defs/Node"},
                },
            },
        },
    }
    _check(tool_schema(schema))
    looped = {
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/Loop"}},
        "$defs": {"Loop": {"$ref": "#/$defs/Loop"}},
    }
    assert tool_schema(looped)["properties"]["x"] == {"type": "string"}


def test_render_text_and_error() -> None:
    reply = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert render_result(reply) == ("a\n\nb", False)
    failed = {"content": [{"type": "text", "text": "сломано"}], "isError": True}
    assert render_result(failed) == ("сломано", True)


@pytest.mark.parametrize(
    ("block", "fragment"),
    [
        ({"type": "image", "data": "AAAA", "mimeType": "image/png"}, "image/png, около 3 байт"),
        ({"type": "audio", "data": "AAAA", "mimeType": "audio/wav"}, "audio/wav"),
        ({"type": "resource", "resource": {"uri": "file:///a", "text": "тело"}}, "тело"),
        ({"type": "resource", "resource": {"uri": "file:///b", "blob": "AA"}}, "file:///b"),
        ({"type": "resource_link", "uri": "file:///c", "name": "c"}, "[ресурс file:///c: c]"),
        ({"type": "video"}, "не поддерживается"),
    ],
)
def test_render_non_text_blocks(block: dict[str, Any], fragment: str) -> None:
    text, failed = render_result({"content": [block]})
    assert fragment in text
    assert not failed


def test_render_structured_content_and_empty() -> None:
    reply = {"content": [], "structuredContent": {"temperature": 21}}
    assert '"temperature": 21' in render_result(reply)[0]
    assert render_result({}) == ("(пустой ответ)", False)


def test_render_truncates_long_output() -> None:
    reply = {"content": [{"type": "text", "text": "x" * (MAX_RESULT_CHARS * 2)}]}
    text, _ = render_result(reply)
    assert len(text) < MAX_RESULT_CHARS + 200


def test_all_of_with_several_refs_keeps_every_part() -> None:
    schema = {
        "type": "object",
        "properties": {"item": {"allOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/B"}]}},
        "$defs": {
            "A": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
            "B": {"type": "object", "properties": {"b": {"type": "integer"}}, "required": ["b"]},
        },
    }
    nested = tool_schema(schema)["properties"]["item"]
    _check(nested)
    assert set(nested["properties"]) == {"a", "b"}
    assert sorted(nested["required"]) == ["a", "b"]


def test_recursive_refs_expand_once_per_path() -> None:
    node_props = {f"link{index}": {"$ref": "#/$defs/Node"} for index in range(6)}
    schema = {
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/Node"}},
        "$defs": {"Node": {"type": "object", "properties": node_props, "description": "узел"}},
    }
    converted = tool_schema(schema)
    _check(converted)
    root = converted["properties"]["root"]
    assert set(root["properties"]) == set(node_props)
    assert root["properties"]["link0"] == {"type": "object", "description": "узел"}
    assert len(json.dumps(converted)) < 2000


def test_json_pointer_refs_into_properties() -> None:
    # zod-to-json-schema ссылается на уже описанное свойство: #/properties/<имя>.
    schema = {
        "type": "object",
        "properties": {
            "source": {"type": "object", "properties": {"path": {"type": "string"}}},
            "target": {"$ref": "#/properties/source"},
            "escaped": {"$ref": "#/properties/a~1b"},
            "a/b": {"type": "integer"},
            "external": {"$ref": "https://example.com/schema.json"},
        },
    }
    props = tool_schema(schema)["properties"]
    assert props["target"]["properties"] == {"path": {"type": "string"}}
    assert props["escaped"] == {"type": "integer"}
    assert props["external"] == {"type": "string"}
