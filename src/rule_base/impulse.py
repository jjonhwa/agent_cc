import ast
from typing import Dict, List

from .common import normalize_signals

_IMPULSE_CATEGORIES = [
    "규범위반적행동",
    "만취로인한문제행동",
    "온라인에서의성적언어적접촉시도",
    "갑작스러운직업변화",
    "통제되지않는지출행동",
]


def rule_impulse(ts_rows: list) -> bool:
    """
    Pattern detection for impulse domain.
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

    return has_impulse_pattern_in_window(events)


def has_impulse_pattern_in_window(events: List[Dict]) -> bool:
    """
    multi-criteria 기반 충동 영역 패턴 판정
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

    last10 = events[-10:]

    # 1. 충동 카테고리 중 하나가 10회 중 2회 이상 등장
    if any(count_class(cat, last10) >= 2 for cat in _IMPULSE_CATEGORIES):
        return True

    # 2. 충동 카테고리 중 하나가 1회 이상 등장 AND "위법 행동"이 1회 이상 등장
    if (
        any(count_class(cat) >= 1 for cat in _IMPULSE_CATEGORIES)
        and count_class("위법행동") >= 1
    ):
        return True

    return False
