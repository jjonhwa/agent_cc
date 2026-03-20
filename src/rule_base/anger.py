import ast
from typing import Dict, List

from .common import normalize_signals


def rule_anger(ts_rows: list) -> bool:
    """
    Pattern detection using normalize_signals + has_anger_pattern_in_window.
    Expects each row to contain '최종판단' (str) and '세부카테고리' (list) keys.
    """
    if not ts_rows:
        return False
    signals = normalize_signals(ts_rows)

    events = []

    for sig, row in zip(signals, ts_rows):
        raw_classes = row.get("세부카테고리", [])

        try:
            parsed_classes = (
                ast.literal_eval(raw_classes)
                if isinstance(raw_classes, str)
                else raw_classes
            )
        except (ValueError, SyntaxError):
            parsed_classes = []

        events.append({"signal": sig, "class": parsed_classes})

    return has_anger_pattern_in_window(events)


def has_anger_pattern_in_window(events: List[Dict]) -> bool:
    """
    multi-criteria 기반 분노 영역 패턴 판정
    각 event:
    {
        "signal": bool,
        "class": List[str]
    }
    """
    if not events:
        return False

    def count_class(target: str, window: list = None) -> int:
        src = window if window is not None else events
        return sum(1 for e in src if target in (e.get("class") or []))

    recent3 = events[-3:]

    # 1. "갈등사건"이 2회 이상 반복 등장 AND "공격행동"이 1회 이상 등장
    if count_class("갈등사건") >= 2 and count_class("공격행동") >= 1:
        return True

    # 2. "음주상태 & 문제행동"이 2회 이상 등장 AND "공격행동"이 1회 이상 등장
    if count_class("음주상태 & 문제행동") >= 2 and count_class("공격행동") >= 1:
        return True

    # 3. "공격행동"이 1회 이상 등장
    if count_class("공격행동") >= 1:
        return True

    # 4. "음주상태"가 1회 이상 등장 AND "공격행동"이 2회 이상 반복 등장
    if count_class("음주상태") >= 1 and count_class("공격행동") >= 2:
        return True

    # 5. "사회에 대한 불만"이 최근 3회 중 2회 이상 등장
    if count_class("사회에대한불만", recent3) >= 2:
        return True

    # 6. "문제행동"이 2회 이상 등장
    if count_class("문제행동") >= 2:
        return True

    return False
