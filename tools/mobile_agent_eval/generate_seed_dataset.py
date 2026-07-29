#!/usr/bin/env python3
"""Generate a deterministic Chinese seed set for mobile tool-routing evals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def expected_local(*calls: dict[str, Any]) -> dict[str, Any]:
    return {"route": "local_tool", "calls": list(calls)}


def call(tool: str, **args: Any) -> dict[str, Any]:
    return {"tool": tool, "args": args}


def expected_clarify(reason: str, missing: list[str] | None = None, candidates: list[str] | None = None) -> dict[str, Any]:
    return {
        "route": "clarify",
        "calls": [],
        "reason": reason,
        "missing": missing or [],
        "candidate_tools": candidates or [],
    }


def expected_none(reason: str) -> dict[str, Any]:
    return {"route": "no_tool", "calls": [], "reason": reason}


def expected_cloud(reason: str) -> dict[str, Any]:
    return {"route": "cloud", "calls": [], "reason": reason}


def add_case(rows: list[dict[str, Any]], category: str, text: str, expected: dict[str, Any]) -> None:
    rows.append(
        {
            "id": f"zh_{len(rows) + 1:04d}",
            "locale": "zh-CN",
            "category": category,
            "text": text,
            "expected": expected,
        }
    )


def build_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    single_cases = [
        ("明天早上七点半叫我起床", call("alarm.create", time="07:30", date="明天", label="起床", repeat="none")),
        ("工作日早上 8 点设个闹钟", call("alarm.create", time="08:00", repeat="weekdays")),
        ("取消今晚 11 点的闹钟", call("alarm.delete", time="23:00", date="今天")),
        ("帮我计时 15 分钟", call("timer.start", duration="15m")),
        ("十分钟后提醒我喝水", call("reminder.create", title="喝水", time="10 分钟后")),
        ("提醒我明天下午三点给王老师回电话", call("reminder.create", title="给王老师回电话", date="明天", time="15:00")),
        ("下周一上午十点加一个产品评审会", call("calendar.create_event", title="产品评审会", date="下周一", start_time="10:00")),
        ("查一下我明天有哪些会议", call("calendar.find_event", query="会议", date="明天")),
        ("打开微信", call("app.open", app_name="微信")),
        ("把蓝牙打开", call("settings.toggle", setting="bluetooth", state="on")),
        ("关掉飞行模式", call("settings.toggle", setting="airplane_mode", state="off")),
        ("把媒体音量调到 30", call("settings.set_volume", stream="media", level="30")),
        ("屏幕亮度调到一半", call("settings.set_brightness", level="50")),
        ("找一下李雷的电话", call("contacts.search", name="李雷")),
        ("给妈妈打电话", call("phone.call", contact="妈妈", number_type="unknown")),
        ("发短信给小张说我晚十分钟到", call("sms.send", recipient="小张", content="我晚十分钟到")),
        ("给王总写封邮件，主题是周报", call("email.compose", recipient="王总", subject="周报")),
        ("记一下买牛奶和鸡蛋", call("notes.create", content="买牛奶和鸡蛋")),
        ("查一下北京明天的天气", call("weather.query", location="北京", date="明天")),
        ("导航去上海虹桥站", call("navigation.start", destination="上海虹桥站", mode="driving")),
        ("放一首周杰伦的歌", call("music.play", query="周杰伦")),
        ("打开相机扫码", call("camera.open", mode="scan")),
        ("把客厅灯关了", call("smart_home.control", device="客厅灯", action="turn_off")),
        ("帮我找一下身份证照片", call("file.search", query="身份证照片", file_type="image")),
        ("复制这句话：下午三点见", call("clipboard.copy", content="下午三点见")),
        ("打开支付宝扫码", call("payment.open_scan", app_name="Alipay", mode="scan")),
        ("把 good morning 翻译成中文", call("translation.translate", text="good morning", target_language="中文")),
    ]
    colloquial_prefixes = ["帮我", "麻烦", "顺手", "现在", "一会儿"]
    for text, tool_call in single_cases:
        add_case(rows, "single_tool_complete", text, expected_local(tool_call))
    for index, (text, tool_call) in enumerate(single_cases):
        prefix = colloquial_prefixes[index % len(colloquial_prefixes)]
        add_case(rows, "colloquial", f"{prefix}{text}", expected_local(tool_call))

    missing_cases = [
        ("帮我设个闹钟", "missing required alarm time", ["time"], ["alarm.create"]),
        ("提醒我一下", "missing reminder content and time", ["title"], ["reminder.create"]),
        ("给他打电话", "ambiguous contact pronoun", ["contact"], ["phone.call"]),
        ("发个短信说我到了", "missing recipient", ["recipient"], ["sms.send"]),
        ("加个会议到日历", "missing event title or date", ["title", "date"], ["calendar.create_event"]),
        ("导航过去", "missing destination", ["destination"], ["navigation.start"]),
        ("开一下灯", "missing smart-home device or room", ["device"], ["smart_home.control"]),
        ("把音量调低点", "missing stream or exact level", ["stream", "level"], ["settings.set_volume"]),
        ("帮我打开那个软件", "ambiguous app name", ["app_name"], ["app.open"]),
        ("给老板发邮件", "missing email subject", ["subject"], ["email.compose"]),
    ]
    for text, reason, missing, candidates in missing_cases:
        add_case(rows, "missing_parameter", text, expected_clarify(reason, missing, candidates))

    ambiguous_cases = [
        ("打开热点还是无线都行", "two possible settings requested without preference", [], ["settings.toggle"]),
        ("给张伟打电话", "multiple contacts may share this name", ["contact"], ["phone.call", "contacts.search"]),
        ("明天上午安排一下", "calendar intent lacks event title and exact time", ["title", "start_time"], ["calendar.create_event"]),
        ("把家里温度调舒服点", "smart-home target/value is subjective", ["device", "value"], ["smart_home.control"]),
        ("提醒我那个事", "deictic reminder content is unresolved", ["title"], ["reminder.create"]),
        ("帮我放那首歌", "song reference unresolved", ["query"], ["music.play"]),
        ("开一下省电", "state may mean enable battery saver but needs confirmation", [], ["settings.toggle"]),
        ("找一下昨天那个文件", "file query is too vague", ["query"], ["file.search"]),
    ]
    for text, reason, missing, candidates in ambiguous_cases:
        add_case(rows, "ambiguous", text, expected_clarify(reason, missing, candidates))

    multi_cases = [
        (
            "明早七点叫我起床，然后打开蓝牙",
            [
                call("alarm.create", time="07:00", date="明天", label="起床", repeat="none"),
                call("settings.toggle", setting="bluetooth", state="on"),
            ],
        ),
        (
            "给妈妈打电话，顺便提醒我晚上买菜",
            [
                call("phone.call", contact="妈妈", number_type="unknown"),
                call("reminder.create", title="买菜", date="今天", time="晚上"),
            ],
        ),
        (
            "打开微信并把 WiFi 关掉",
            [
                call("app.open", app_name="微信"),
                call("settings.toggle", setting="wifi", state="off"),
            ],
        ),
        (
            "查上海明天天气，再导航去虹桥机场",
            [
                call("weather.query", location="上海", date="明天"),
                call("navigation.start", destination="虹桥机场", mode="driving"),
            ],
        ),
        (
            "记一下买咖啡，同时十分钟后提醒我出门",
            [
                call("notes.create", content="买咖啡"),
                call("reminder.create", title="出门", time="10 分钟后"),
            ],
        ),
        (
            "打开支付宝扫码，然后把屏幕亮度调到 80",
            [
                call("payment.open_scan", app_name="Alipay", mode="scan"),
                call("settings.set_brightness", level="80"),
            ],
        ),
    ]
    for text, calls in multi_cases:
        add_case(rows, "multi_action", text, expected_local(*calls))

    no_tool_texts = [
        "你觉得我今天心情怎么样",
        "讲个冷笑话",
        "我应该换工作吗",
        "帮我写一首关于夏天的诗",
        "解释一下量子纠缠",
        "你是谁",
        "这个手机好不好看",
        "给我推荐一部电影",
        "我肚子疼怎么办",
        "股票明天会涨吗",
    ]
    for text in no_tool_texts:
        add_case(rows, "no_tool", text, expected_none("request is not a supported phone tool action"))

    cloud_texts = [
        "帮我规划下个月去日本七天的详细行程，顺便比较机票酒店价格",
        "分析我最近三个月的消费记录，给出预算建议",
        "把这篇长论文读完并总结贡献和缺陷",
        "帮我查一下现在最新的房贷政策并算一下我能贷多少",
        "根据我所有聊天记录判断谁最可能参加聚会",
        "帮我写一个完整的商业计划书并找竞品数据",
        "帮我诊断这个持续两周的胸痛可能是什么",
        "查全网最便宜的冰箱并下单前比较评价",
        "根据邮件往来自动谈判会议时间",
        "做一个三个月健身饮食计划并每天调整",
    ]
    for text in cloud_texts:
        add_case(rows, "cloud_route", text, expected_cloud("complex or high-stakes task should route to cloud agent"))

    # Add deterministic paraphrases until the seed set is large enough for a useful smoke benchmark.
    base_rows = list(rows)
    paraphrase_prefixes = ["请", "能不能", "现在帮我", "麻烦你", "我想让你"]
    for source in base_rows:
        if len(rows) >= 240:
            break
        if source["category"] in {"single_tool_complete", "colloquial", "multi_action"}:
            for prefix in paraphrase_prefixes:
                if len(rows) >= 240:
                    break
                add_case(rows, source["category"], f"{prefix}{source['text']}", source["expected"])

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("seed_dataset_zh.jsonl"))
    args = parser.parse_args()

    rows = build_cases()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
