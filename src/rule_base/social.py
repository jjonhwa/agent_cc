import ast
from typing import Dict, List

from .common import normalize_signals


def rule_social(ts_rows: list) -> bool:
    """
    Pattern detection for social domain.
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

    return has_social_pattern_in_window(events)


def has_social_pattern_in_window(events: List[Dict]) -> bool:
    """
    multi-criteria 기반 사회 영역 패턴 판정
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

    def has_consecutive(target: str, n: int) -> bool:
        count = 0
        for e in events:
            if target in (e.get("class") or []):
                count += 1
                if count >= n:
                    return True
            else:
                count = 0
        return False

    last10 = events[-10:]

    # -------------------------
    # 패턴 없음 조건 (early exit)
    # -------------------------
    # "교류회복"이 연속 3회 이상 등장
    if has_consecutive("교류회복", 3):
        return False

    # -------------------------
    # 패턴 있음 조건
    # -------------------------
    # 1. "친구지인애인과의교류부재"가 10회 중 3회 이상 등장
    if count_class("친구지인애인과의교류부재", last10) >= 3:
        return True

    # 2. "친구지인애인과의교류부재"가 1회 이상 등장 AND "새로운 인간관계 형성"이 등장하지 않음
    if (
        count_class("친구지인애인과의교류부재") >= 1
        and count_class("친구지인애인과의교류부재, 새로운관계형성") == 0
    ):
        return True

    return False
