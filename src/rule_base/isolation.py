import ast
from typing import Dict, List

from .common import normalize_signals


def rule_isolation(ts_rows: list) -> bool:
    """
    Pattern detection for isolation domain.
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

    return has_isolation_pattern_in_window(events)


def has_isolation_pattern_in_window(events: List[Dict]) -> bool:
    """
    multi-criteria 기반 고립 영역 패턴 판정
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
    # "일상기능저하"가 1회 이하 등장 AND "단기스트레스"가 1회 이상 등장
    if count_class("일상기능저하") <= 1 and count_class("단기스트레스") >= 1:
        return False

    # -------------------------
    # 패턴 있음 조건
    # -------------------------
    # 1. "생활에대한직접적무기력호소"가 1회 이상 등장 AND "일상기능저하"가 3회 이상 등장
    if (
        count_class("생활에대한직접적무기력호소") >= 1
        and count_class("일상기능저하") >= 3
    ):
        return True

    # 2. "중증신호"가 2회 이상 등장
    if count_class("중증 신호") >= 2:
        return True

    # 3. "일상기능저하"가 10회 중 3회 이상 등장 OR 최근 3회 중 1회 이상 등장
    if (
        count_class("일상기능저하", last10) >= 3
        or count_class("일상기능저하", recent3) >= 1
    ):
        return True

    # 4. "명령거부 또는 비순응"이 1회 이상 등장
    if count_class("명령거부또는비순응") >= 1:
        return True

    return False
