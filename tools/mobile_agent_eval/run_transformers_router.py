#!/usr/bin/env python3
"""Run a Hugging Face causal LM as a Chinese mobile tool router."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any


ALLOWED_ROUTES = {"local_tool", "clarify", "no_tool", "cloud"}


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(rows) >= limit:
                break
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def load_tool_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_tool_list(schema: dict[str, Any]) -> str:
    lines = []
    for tool in schema["tools"]:
        params = tool.get("parameters") or {}
        required = ",".join(tool.get("required") or [])
        arg_names = ",".join(params.keys())
        lines.append(f"- {tool['name']}: required=[{required}], args=[{arg_names}]")
    return "\n".join(lines)


def build_system_prompt(schema: dict[str, Any]) -> str:
    return f"""你是手机端本地 Agent 的工具路由器。你只能输出一个 JSON 对象，不能输出解释文字。

任务：根据用户中文指令选择路线。

route 必须严格等于下面四个字符串之一：
- "local_tool": 可以直接调用一个或多个本地手机工具。
- "clarify": 工具意图存在但缺少必要参数，或指代/歧义需要问用户确认。
- "no_tool": 用户请求不属于任何手机本地工具。
- "cloud": 请求复杂、高风险、需要联网最新信息、大范围搜索、长文档处理或多步规划，应升级云端。

输出 JSON schema:
{{"id":"样本 id","route":"四选一","calls":[{{"tool":"工具名","args":{{}}}}],"clarification":""}}

规则：
1. 只输出 JSON，不要 Markdown。
2. route 不允许写工具名，不允许写 "local_tool|clarify|no_tool|cloud"。
3. local_tool 时 calls 必须按用户动作顺序排列，只输出必要工具，不要额外添加闹钟/提醒。
4. 不要编造具体日期、联系人、地点；缺少 required 参数时 route=clarify。
5. 参数值尽量规范化，例如 7 点半写成 07:30，百分比写成 0-100 的数字字符串。
6. 不要输出空字符串参数；未知可选参数直接省略。
7. 不支持的闲聊、建议、知识问答 route=no_tool。
8. 医疗、法律、金融、旅行规划、联网比价、长文档总结等复杂任务 route=cloud。

例子：
用户: 打开微信
输出: {{"id":"demo","route":"local_tool","calls":[{{"tool":"app.open","args":{{"app_name":"微信"}}}}],"clarification":""}}

用户: 把蓝牙打开
输出: {{"id":"demo","route":"local_tool","calls":[{{"tool":"settings.toggle","args":{{"setting":"bluetooth","state":"on"}}}}],"clarification":""}}

用户: 帮我设个闹钟
输出: {{"id":"demo","route":"clarify","calls":[],"clarification":"请问要设置几点的闹钟？"}}

用户: 讲个笑话
输出: {{"id":"demo","route":"no_tool","calls":[],"clarification":""}}

用户: 帮我规划下个月去日本七天的详细行程并比较价格
输出: {{"id":"demo","route":"cloud","calls":[],"clarification":""}}

可用工具：
{compact_tool_list(schema)}
"""


def build_user_prompt(row: dict[str, Any]) -> str:
    return f"""样本 id: {row['id']}
用户指令: {row['text']}

请输出 JSON："""


def format_prompt(tokenizer: Any, system_prompt: str, user_prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{system_prompt}\n\n{user_prompt}\n"


def find_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
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
                return text[start : index + 1]
    return None


def normalize_prediction(parsed: Any, sample_id: str) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {"id": sample_id, "route": "parse_error", "calls": [], "clarification": "model did not return an object"}
    route = str(parsed.get("route", "parse_error"))
    calls = parsed.get("calls") or []
    if not isinstance(calls, list):
        calls = []
    normalized_calls = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        args = call.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        normalized_calls.append({"tool": str(call.get("tool", "")), "args": args})
    return {
        "id": str(parsed.get("id") or sample_id),
        "route": route if route in ALLOWED_ROUTES else route,
        "calls": normalized_calls,
        "clarification": str(parsed.get("clarification", "")),
    }


def parse_model_json(text: str, sample_id: str) -> tuple[dict[str, Any], str | None]:
    candidate = find_json_object(text)
    if candidate is None:
        return normalize_prediction(None, sample_id), "no_json_object"
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return normalize_prediction(None, sample_id), f"json_decode_error: {exc}"
    return normalize_prediction(parsed, sample_id), None


def parse_torch_dtype(raw: str) -> Any:
    import torch

    normalized = raw.lower()
    if normalized == "auto":
        return "auto"
    if normalized == "float32":
        return torch.float32
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported --torch_dtype: {raw}")


def process_memory_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def generate_streamed(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: float,
) -> tuple[str, float, float, int]:
    import torch
    from transformers import TextIteratorStreamer

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=timeout_s)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p,
        "pad_token_id": tokenizer.eos_token_id,
    }
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

    errors: list[BaseException] = []

    def run_generate() -> None:
        try:
            with torch.no_grad():
                model.generate(**generation_kwargs)
        except BaseException as exc:  # propagate from worker thread
            errors.append(exc)

    started = time.perf_counter()
    thread = threading.Thread(target=run_generate, daemon=True)
    thread.start()
    chunks: list[str] = []
    ttft_ms = 0.0
    for chunk in streamer:
        if chunk and ttft_ms == 0.0:
            ttft_ms = (time.perf_counter() - started) * 1000.0
        chunks.append(chunk)
    thread.join()
    if errors:
        raise RuntimeError(f"model.generate failed: {errors[0]}") from errors[0]
    e2e_ms = (time.perf_counter() - started) * 1000.0
    text = "".join(chunks)
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    return text, ttft_ms, e2e_ms, token_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("seed_dataset_zh.jsonl"))
    parser.add_argument("--tool_schema", type=Path, default=Path(__file__).with_name("tool_schemas.json"))
    parser.add_argument("--output", type=Path, required=True, help="Prediction JSONL output.")
    parser.add_argument("--runtime_output", type=Path, default=None, help="Per-request runtime JSONL output.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0")
    parser.add_argument("--torch_dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--disable_chat_template", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--timeout_s", type=float, default=120.0)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_jsonl(args.dataset, args.limit)
    schema = load_tool_schema(args.tool_schema)
    system_prompt = build_system_prompt(schema)

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=parse_torch_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device)
    model.eval()
    load_ms = (time.perf_counter() - load_started) * 1000.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    runtime_path = args.runtime_output or args.output.with_name(args.output.stem + "_runtime.jsonl")
    runtime_path.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as pred_handle, runtime_path.open("w", encoding="utf-8") as runtime_handle:
        for index, row in enumerate(rows, start=1):
            prompt = format_prompt(
                tokenizer,
                system_prompt,
                build_user_prompt(row),
                use_chat_template=not args.disable_chat_template,
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            generated, ttft_ms, e2e_ms, output_tokens = generate_streamed(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout_s=args.timeout_s,
            )
            parse_error = None
            prediction, parse_error = parse_model_json(generated, str(row["id"]))
            prediction["raw_text"] = generated
            pred_handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            pred_handle.flush()

            cuda_peak_mb = 0.0
            if device.type == "cuda":
                cuda_peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            runtime = {
                "id": row["id"],
                "index": index,
                "model_name_or_path": args.model_name_or_path,
                "device": str(device),
                "load_ms_first_row": load_ms if index == 1 else 0.0,
                "ttft_ms": ttft_ms,
                "e2e_ms": e2e_ms,
                "output_tokens": output_tokens,
                "tokens_per_s": output_tokens / (e2e_ms / 1000.0) if e2e_ms > 0 else 0.0,
                "process_rss_mb": process_memory_mb(),
                "cuda_peak_alloc_mb": cuda_peak_mb,
                "parse_error": parse_error or "",
                "route": prediction.get("route", ""),
                "category": row.get("category", ""),
            }
            runtime_handle.write(json.dumps(runtime, ensure_ascii=False) + "\n")
            runtime_handle.flush()
            print(
                f"[{index}/{len(rows)}] {row['id']} route={prediction.get('route')} "
                f"ttft={ttft_ms:.1f}ms e2e={e2e_ms:.1f}ms parse_error={parse_error or '-'}",
                flush=True,
            )

    print(f"Wrote predictions: {args.output}")
    print(f"Wrote runtime: {runtime_path}")


if __name__ == "__main__":
    main()
