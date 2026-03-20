from typing import Dict, List, Optional


def normalize_signals(rows: List[Dict]) -> List[bool]:
    """
    Neutral을 이전 상태로 치환하여 True/False 시퀀스로 변환
    """
    normalized: List[bool] = []
    last_state: Optional[bool] = None

    for row in rows:
        raw_state = str(row["최종판단"]).lower()

        if raw_state == "true":
            state: Optional[bool] = True
        elif raw_state == "false":
            state = False
        else:
            state = None  # neutral

        if state is None:
            normalized.append(last_state if last_state is not None else False)
        else:
            normalized.append(state)
            last_state = state

    return normalized
