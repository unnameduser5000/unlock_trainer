#!/usr/bin/env python3
"""Shared protocol helpers for tool-calling datasets and runners."""

from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JSON_CALLS_SYSTEM_PROMPT = (
    "You are an on-device Android function-calling model.\n"
    "Return only one JSON object with this exact shape:\n"
    '{"calls":[{"name":"function_name","arguments":{}}]}\n'
    "Use only the available functions.\n"
    "Preserve the user-requested action order.\n"
    "If a later call depends on the return value of an earlier call, "
    'reference it as a string placeholder like "#0", "#1", and so on.\n'
    "Do not add explanations or Markdown."
)

CODE_SHORT_SYSTEM_PROMPT = "You are an expert in composing functions."


@dataclass(frozen=True)
class ToolCallExample:
    split: str
    prompt: str
    target: str
    user: str
    expected_calls: list[dict[str, Any]]
    target_format: str
    example_id: str = ""
    protocol: str = ""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def extract_calls_from_openai_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    assistant = row["messages"][-1]
    calls = []
    for call in assistant.get("tool_calls") or []:
        function = call.get("function") or {}
        calls.append(
            {
                "name": function.get("name", ""),
                "arguments": function.get("arguments") or {},
            }
        )
    return calls


def normalize_tool_spec(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function":
        function = tool.get("function") or {}
        parameters = function.get("parameters") or {}
        return {
            "name": function.get("name", ""),
            "description": function.get("description", ""),
            "arguments": parameters.get("properties") or {},
            "required": parameters.get("required") or [],
        }
    arguments = tool.get("arguments") or {}
    required = [name for name, spec in arguments.items() if bool((spec or {}).get("required"))]
    return {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "arguments": arguments,
        "required": required,
    }


def compact_tools_for_json_calls(tools: list[dict[str, Any]]) -> str:
    lines = []
    for tool in tools:
        normalized = normalize_tool_spec(tool)
        arguments = normalized.get("arguments") or {}
        required = normalized.get("required") or []
        args = []
        for name, spec in sorted(arguments.items()):
            if "type" in spec:
                spec_type = spec.get("type", "ANY")
                desc = spec.get("description", "")
            else:
                spec_type = "ANY"
                desc = str(spec)
            args.append(f"{name}: {spec_type} - {desc}")
        args_text = "; ".join(args) if args else "none"
        lines.append(
            f"- {normalized.get('name', '')}: {normalized.get('description', '')} "
            f"required={required}; args={args_text}"
        )
    return "\n".join(lines)


def make_json_calls_prompt(*, user: str, tools: list[dict[str, Any]], developer: str = "") -> str:
    return (
        "You are an on-device Android function-calling model.\n"
        "Return only one JSON object with this exact shape:\n"
        '{"calls":[{"name":"function_name","arguments":{}}]}\n'
        "Use only the available functions. Preserve the user-requested action order. "
        'If a later call depends on the return value of an earlier call, reference it as "#0", "#1", and so on. '
        "Do not add explanations or Markdown.\n\n"
        f"{developer.strip()}\n\n"
        "Available functions:\n"
        f"{compact_tools_for_json_calls(tools)}\n\n"
        f"User request: {user}\n"
        "JSON:"
    )


def format_code_short_tool(tool: dict[str, Any]) -> str:
    normalized = normalize_tool_spec(tool)
    lines = [
        "Name:",
        f"    {normalized.get('name', '')}",
        "Description:",
        f"    {normalized.get('description', '').strip()}",
    ]
    arguments = normalized.get("arguments") or {}
    if arguments:
        lines.append("")
        lines.append("Args:")
        for arg_name, spec in arguments.items():
            spec = spec or {}
            arg_type = spec.get("type", "Any")
            desc = str(spec.get("description", "")).strip()
            default = spec.get("default")
            if default is not None:
                desc = f"{desc} Default is {default!r}.".strip()
            lines.append(f"    {arg_name} ({arg_type}): {desc}")
    lines.append("Returns:")
    returns = tool.get("returns") or {}
    if isinstance(returns, dict) and returns:
        return_type = returns.get("type", "Any")
        return_desc = str(returns.get("description", "")).strip()
        lines.append(f"    {return_type}: {return_desc}")
    else:
        lines.append("    None")
    examples = tool.get("examples") or []
    if examples:
        lines.append("Example:")
        for example in examples[:2]:
            lines.append(f"    {example}")
    return "\n".join(lines)


def make_code_short_prompt(*, user: str, tools: list[dict[str, Any]]) -> str:
    rendered_tools = "\n==================================================\n".join(
        format_code_short_tool(tool) for tool in tools
    )
    return (
        f"{CODE_SHORT_SYSTEM_PROMPT}\n\n"
        "Here is a list of functions:\n"
        f"{rendered_tools}\n\n"
        f"Now my query is: {user}\n"
    )


def calls_to_code_short(calls: list[dict[str, Any]]) -> str:
    lines = []
    for idx, call in enumerate(calls):
        args = call.get("arguments") or {}
        rendered_args = ", ".join(f"{name}={repr(value)}" for name, value in args.items())
        lines.append(f"$result{idx} = {call.get('name', '')}({rendered_args})" if idx == 0 else f"result{idx} = {call.get('name', '')}({rendered_args})")
    if lines:
        lines[-1] = f"{lines[-1]}$"
    return "\n".join(lines)


def parse_json_object(text: str) -> Any | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : idx + 1])
                except json.JSONDecodeError:
                    return None
    return None


def normalize_json_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    calls = value.get("calls")
    if not isinstance(calls, list):
        return []
    normalized = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        args = call.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        normalized.append({"name": str(call.get("name", "")), "arguments": args})
    return normalized


def _literal_from_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_literal_from_ast(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return [_literal_from_ast(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal_from_ast(key): _literal_from_ast(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal_from_ast(node.operand)
    if isinstance(node, ast.Name):
        if node.id == "None":
            return None
        if node.id == "True":
            return True
        if node.id == "False":
            return False
    raise ValueError(f"Unsupported literal node: {ast.dump(node)}")


def parse_code_short_calls(text: str) -> list[dict[str, Any]] | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    cleaned = cleaned.replace("```python", "").replace("```json", "").replace("```", "")
    cleaned = cleaned.replace("\r\n", "\n")
    candidate_lines: list[str] = []
    for raw_line in cleaned.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("$"):
            line = line[1:].strip()
        if line.endswith("$"):
            line = line[:-1].strip()
        if not line or "Explanation:" in line:
            continue
        if "=" in line and "(" in line and ")" in line:
            candidate_lines.append(line)
        elif line.endswith(")") and "(" in line:
            candidate_lines.append(line)
    calls: list[dict[str, Any]] = []
    for line in candidate_lines:
        expression = line.split("=", 1)[1].strip() if "=" in line else line.strip()
        try:
            parsed = ast.parse(expression, mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(parsed, ast.Call):
            continue
        if not isinstance(parsed.func, ast.Name):
            continue
        if parsed.args:
            return None
        arguments: dict[str, Any] = {}
        for keyword in parsed.keywords:
            if keyword.arg is None:
                return None
            arguments[keyword.arg] = _literal_from_ast(keyword.value)
        calls.append({"name": parsed.func.id, "arguments": arguments})
    return calls or None


def parse_prediction_text(text: str, target_format: str) -> tuple[Any | None, list[dict[str, Any]], bool, str]:
    if target_format == "code_short":
        calls = parse_code_short_calls(text)
        if calls is None:
            return None, [], False, "code_short"
        return {"calls": calls}, calls, True, "code_short"
    parsed = parse_json_object(text)
    calls = normalize_json_calls(parsed)
    return parsed, calls, parsed is not None, "json_calls"


def infer_target_format_from_messages(messages: list[dict[str, Any]]) -> str:
    assistant = messages[-1] if messages else {}
    if assistant.get("tool_calls"):
        return "json_calls"
    content = str(assistant.get("content", "") or "")
    if "result0 =" in content or content.startswith("$result"):
        return "code_short"
    return "json_calls"


def rows_to_examples(rows: list[dict[str, Any]]) -> list[ToolCallExample]:
    examples: list[ToolCallExample] = []
    for index, row in enumerate(rows):
        if "prompt" in row and "target" in row:
            target_format = str(row.get("target_format", "json_calls"))
            expected_calls = row.get("expected_calls")
            if not isinstance(expected_calls, list):
                _, expected_calls, _, _ = parse_prediction_text(str(row["target"]), target_format)
            examples.append(
                ToolCallExample(
                    split=str(row.get("split", "train")),
                    prompt=str(row.get("prompt", "")),
                    target=str(row.get("target", "")),
                    user=str(row.get("user", "")),
                    expected_calls=expected_calls,
                    target_format=target_format,
                    example_id=str(row.get("id", f"row{index}")),
                    protocol=str(row.get("protocol", "")),
                )
            )
            continue
        if "messages" not in row:
            raise ValueError(f"Unsupported row schema at index {index}: keys={sorted(row.keys())}")
        messages = row["messages"]
        user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        target_format = str(row.get("target_format") or infer_target_format_from_messages(messages))
        if target_format == "code_short":
            prompt = str(row.get("prompt") or "\n\n".join(str(m.get("content", "") or "") for m in messages[:-1]))
            target = str(messages[-1].get("content", "") or "")
            _, expected_calls, _, _ = parse_prediction_text(target, "code_short")
        else:
            prompt = make_json_calls_prompt(
                user=user,
                tools=row.get("tools") or [],
                developer=next((m.get("content", "") for m in messages if m.get("role") == "developer"), ""),
            )
            expected_calls = extract_calls_from_openai_row(row)
            target = canonical_json({"calls": expected_calls})
        examples.append(
            ToolCallExample(
                split=str(row.get("split") or row.get("metadata", "train")),
                prompt=prompt,
                target=target,
                user=str(user),
                expected_calls=expected_calls,
                target_format=target_format,
                example_id=str(row.get("id", f"row{index}")),
                protocol=str(row.get("protocol", "")),
            )
        )
    return examples


def normalize_json_type(raw_type: str | None) -> tuple[str, dict[str, Any] | None]:
    if not raw_type:
        return "string", None
    value = raw_type.strip().lower()
    if value in {"int", "integer"}:
        return "integer", None
    if value in {"float", "double", "number"}:
        return "number", None
    if value in {"bool", "boolean"}:
        return "boolean", None
    if value in {"dict", "object"}:
        return "object", None
    if "list[" in value or value.startswith("list") or value.startswith("array"):
        item_type = "string"
        if "int" in value:
            item_type = "integer"
        elif "float" in value or "number" in value:
            item_type = "number"
        elif "bool" in value:
            item_type = "boolean"
        return "array", {"type": item_type}
    return "string", None


def convert_raw_tool_to_openai_function(tool: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for arg_name, spec in (tool.get("arguments") or {}).items():
        spec = spec or {}
        json_type, items = normalize_json_type(spec.get("type"))
        prop: dict[str, Any] = {
            "type": json_type,
            "description": spec.get("description", ""),
        }
        if items is not None:
            prop["items"] = items
        if "default" in spec and spec.get("default") is not None:
            prop["default"] = spec.get("default")
        properties[arg_name] = prop
        if bool(spec.get("required")):
            required.append(arg_name)
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def convert_answer_calls_to_openai(answer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []
    for idx, answer in enumerate(answer_rows):
        calls.append(
            {
                "id": f"call_{idx}",
                "type": "function",
                "function": {
                    "name": answer.get("name", ""),
                    "arguments": answer.get("arguments") or {},
                },
            }
        )
    return calls


def add_raw_tool_distractors(
    used_tools: list[dict[str, Any]],
    api_catalog: list[dict[str, Any]],
    *,
    n_api: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    tools = [dict(tool) for tool in used_tools]
    used_names = {tool.get("name", "") for tool in tools}
    if len(tools) < n_api:
        candidates = [tool for tool in api_catalog if tool.get("name", "") not in used_names]
        if candidates:
            extra = rng.sample(candidates, k=min(n_api - len(tools), len(candidates)))
            tools.extend(extra)
    rng.shuffle(tools)
    return tools
