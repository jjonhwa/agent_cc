import ast
from typing import Dict, List

from .common import normalize_signals


def rule_economy(ts_rows: list) -> bool:
    """
    Pattern detection for economic domain.
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

    return has_economy_pattern_in_window(events)


def has_economy_pattern_in_window(events: List[Dict]) -> bool:
    """
    multi-criteria 기반 경제 영역 패턴 판정
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
    # "고용상태안정"이 최근 3회 중 3회 등장 AND "생활영위어려움호소"가 10회 중 2회 이하
    if (
        count_class("고용상태안정", recent3) >= 3
        and count_class("생활영위어려움호소", last10) <= 2
    ):
        return False

    # -------------------------
    # 패턴 있음 조건
    # -------------------------
    # 1. "직업변화"가 10회 중 2회 이상 등장
    if count_class("직업변화", last10) >= 2:
        return True

    # 2. "채무발생"이 2회 이상 등장
    if count_class("채무발생") >= 2:
        return True

    # 3. "유흥으로인한생활비부족"이 2회 이상 등장
    if count_class("유흥으로인한생활비부족") >= 2:
        return True

    # 4. "채무발생"이 발생하면서 AND "생활영위어려움호소"가 2회 이상 등장
    if count_class("채무발생") >= 1 and count_class("생활영위어려움호소") >= 2:
        return True

    # 5. "채무발생"이 1회 이상 등장 AND "경제적 어려움호소"가 2회 이상 등장
    if count_class("채무발생") >= 1 and count_class("생활영위어려움호소") >= 2:
        return True

    # 6. "직업변화"가 1회 이상 등장 AND "경제적 어려움호소"가 2회 이상 등장
    if count_class("직업변화") >= 1 and count_class("생활영위어려움호소") >= 2:
        return True

    # 7. "생계유지가 어려운사건"이 최근 3회 중 1회 이상 등장
    if count_class("생계유지가 어려운사건", recent3) >= 1:
        return True

    # 8. "신상정보공개 또는 고지"가 1회 이상 등장 AND "직업변화"가 1회 이상 등장
    if count_class("신상정보공개 또는 고지") >= 1 and count_class("직업변화") >= 1:
        return True

    # 9. "채무발생"이 등장하지 않음 AND "생활영위어려움호소"가 최근 3회 중 2회 이상 등장
    if count_class("채무발생") == 0 and count_class("생활영위어려움호소", recent3) >= 2:
        return True

    return False
