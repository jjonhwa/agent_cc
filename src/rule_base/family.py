import ast
from typing import Dict, List

from .common import normalize_signals


def rule_family(ts_rows: list) -> bool:
    """
    Pattern detection using normalize_signals + has_family_pattern_in_window.
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

    return has_family_pattern_in_window(events)


def has_family_pattern_in_window(events: List[Dict]) -> bool:
    """
    multi-criteria 기반 가족관계 영역 패턴 판정
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
    last10 = events[-10:]

    # -------------------------
    # 패턴 없음 조건 (early exit)
    # -------------------------
    # "가족의정서적실질적지지"가 최근 3회 내 등장 AND "관계 개선을 위한 행동"이 최근 3회 내 등장
    if (
        count_class("가족의정서적실질적지지", recent3) >= 1
        and count_class("관계개선을위한행동", recent3) >= 1
    ):
        return False

    # -------------------------
    # 패턴 있음 조건
    # -------------------------
    # 1. "가족과의갈등이나연락단절"이 10회 중 3회 이상 등장 AND 그 중 1회 이상이 최근 3회 내에 등장
    if (
        count_class("가족과의갈등이나연락단절", last10) >= 3
        and count_class("가족과의갈등이나연락단절", recent3) >= 1
    ):
        return True

    # 2. "가족과의갈등이나연락단절"이 3회 이상 등장
    if count_class("가족과의갈등이나연락단절") >= 3:
        return True

    # 3. "상대방의거절의사"가 1회 이상 등장 AND "관계개선을위한행동"이 1회 이상 등장
    if count_class("상대방의거절의사") >= 1 and count_class("관계개선을위한행동") >= 1:
        return True

    # 4. "가족과의갈등이나연락단절"이 1회 이상 등장 AND "이탈행동"이 10회 중 2회 이상 등장
    if (
        count_class("가족과의갈등이나연락단절") >= 1
        and count_class("이탈행동", last10) >= 2
    ):
        return True

    return False
