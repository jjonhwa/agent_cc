import ast
import uuid as _uuid_lib
from typing import Dict, List

import pandas as pd

from .common import normalize_signals

# Predefined subcategories that escalate risk to level 3.
# Update this list to match your domain requirements.
HIGH_RISK_CATEGORIES: List[str] = [
    "이성 또는 특정 커뮤니티 활동에 집착",
    "목적 없는 이동",
]

# Moderate-risk condition definitions — edit only these lists and thresholds.
_MODERATE_RISK_A_CATEGORIES: List[str] = [
    "허위 진술 및 사실 은폐 시도",
]
_MODERATE_RISK_A_THRESHOLD: int = 3

_MODERATE_RISK_B_CATEGORIES: List[str] = [
    "이성과의 일회성 만남 및 중첩적 교제",
]
_MODERATE_RISK_B_THRESHOLD: int = 1

MODERATE_RISK_CATEGORIES: List[str] = (
    _MODERATE_RISK_A_CATEGORIES + _MODERATE_RISK_B_CATEGORIES
)

# ---------------------------------------------------------------------------
# Serious behavior case definitions — edit only these lists.
# ---------------------------------------------------------------------------

# Case A: any item >= 1 in last 3 sessions
_CASE_A_CATEGORIES: List[str] = [
    "생계 유지가 어려운 사건",
]

# Case B: any item in TRIGGER >= 1  AND  any item in AND >= 1
_CASE_B_TRIGGER_CATEGORIES: List[str] = [
    "신상 정보 공개 또는 고지",
]
_CASE_B_AND_CATEGORIES: List[str] = [
    "직업 변화",
]

# Case C: any item >= 1
_CASE_C_CATEGORIES: List[str] = [
    "공격 행동",
]

# Case D: any item in TRIGGER >= 1  AND  any item in AND >= 1
_CASE_D_TRIGGER_CATEGORIES: List[str] = [
    "규범 위반적 행동",  # D1
    "만취로 인한 문제 행동",  # D2
    "온라인에서의 성적 언어적 접촉 시도",  # D3
    "갑작스러운 직업 변화",  # D4
    "통제되지 않는 지출 행동",  # D5
]
_CASE_D_AND_CATEGORIES: List[str] = [
    "위법 행동",
]

# Only items in this list are retained in 세부카테고리 during normalization.
SERIOUS_ACCIDENTS: List[str] = (
    _CASE_A_CATEGORIES
    + _CASE_B_TRIGGER_CATEGORIES
    + _CASE_B_AND_CATEGORIES
    + _CASE_C_CATEGORIES
    + _CASE_D_TRIGGER_CATEGORIES
    + _CASE_D_AND_CATEGORIES
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_list(value) -> List[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        return [value] if value.strip() else []
    return []


def _is_true(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate required columns, normalize '최종판단' to bool,
    and union all subcategory columns into a single '세부카테고리' list column.
    All subcategory items are kept here; SERIOUS_ACCIDENTS filtering is applied
    later in rule_risk() so items from every file survive the merge union.
    """
    missing = {"id", "date", "최종판단"} - set(df.columns)
    if missing:
        raise ValueError(f"File is missing required columns: {missing}")
    subcat_cols = ["세부카테고리"]

    df = df.copy()
    df["최종판단_bool"] = df["최종판단"].apply(_is_true)

    if subcat_cols:
        df["세부카테고리"] = df[subcat_cols].apply(
            lambda row: [item for col in subcat_cols for item in _parse_list(row[col])],
            axis=1,
        )
    else:
        df["세부카테고리"] = [[] for _ in range(len(df))]

    return df


def _merge_normalized(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge a list of normalized frames, combining 세부카테고리 from corresponding rows.

    All frames must have the same length and the same id distribution (including
    duplicate dates for the same id). Sorting by (id, date) guarantees that
    positionally equivalent rows across frames represent the same observation.
    A shared UUID per position is used as a third groupby key so that duplicate-date
    rows are kept as separate output rows — each duplicate date row gets its own
    세부카테고리 union from the matching positional rows across all frames.

    - 최종판단: True if any source row is True (OR)
    - 세부카테고리: union of all source rows at the same position
    """
    lengths = [len(f) for f in frames]
    if len(set(lengths)) != 1:
        raise ValueError(f"Frames have different lengths: {lengths}")

    id_lists = [sorted(f["id"].tolist()) for f in frames]
    if len({tuple(ids) for ids in id_lists}) != 1:
        raise ValueError("Frames have different id distributions")

    n = lengths[0]

    # Sort each frame identically so positional index is aligned across frames.
    sorted_frames = [
        f.sort_values(["id", "date"]).reset_index(drop=True).copy() for f in frames
    ]

    # Assign the same UUID list to all frames so rows at the same position share a key.
    uuids = [str(_uuid_lib.uuid4()) for _ in range(n)]
    for f in sorted_frames:
        f["_uuid"] = uuids

    combined = pd.concat(sorted_frames, ignore_index=True)

    merged_rows = []
    for (_, _, _), group in combined.groupby(["id", "date", "_uuid"], sort=False):
        row = group.iloc[0].to_dict()
        row["최종판단"] = bool(group["최종판단_bool"].any())
        row["세부카테고리"] = [
            item
            for cats in group["세부카테고리"]
            for item in (cats if isinstance(cats, list) else _parse_list(cats))
        ]
        merged_rows.append(row)

    result = pd.DataFrame(merged_rows).drop(columns=["_uuid"], errors="ignore")
    return result.sort_values(["id", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public merging API
# ---------------------------------------------------------------------------


def merge_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge a list of DataFrames (already loaded) by (id, date).
    Accepts raw DataFrames — validation and normalization happen internally.
    Returns a DataFrame with columns: id, date, 최종판단 (bool), 세부카테고리 (list).
    """
    if not dfs:
        return pd.DataFrame(columns=["id", "date", "최종판단", "세부카테고리"])
    frames = [_normalize_df(df) for df in dfs]
    return _merge_normalized(frames)


# def load_and_merge_files(file_paths: List[str]) -> pd.DataFrame:
#     """
#     Load multiple CSV / Excel files from disk and merge by (id, date).
#     Returns a DataFrame with columns: id, date, 최종판단 (bool), 세부카테고리 (list).
#     """
#     dfs = []
#     for path in file_paths:
#         if path.endswith((".xlsx", ".xls")):
#             dfs.append(pd.read_excel(path))
#         else:
#             dfs.append(pd.read_csv(path))
#     return merge_dataframes(dfs)


def get_ts_rows_by_id(merged_df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Convert merged DataFrame to {id: [ts_row, ...]} sorted by date.
    Each ts_row has '최종판단' (str) and '세부카테고리' (list) keys,
    matching the format expected by rule_risk().
    """
    result = {}
    for id_val, group in merged_df.groupby("id"):
        group = group.sort_values("date")
        result[id_val] = [
            {
                "최종판단": str(row["최종판단"]),
                "세부카테고리": row["세부카테고리"],
            }
            for _, row in group.iterrows()
        ]
    return result


# ---------------------------------------------------------------------------
# Category count helpers
# ---------------------------------------------------------------------------


def is_high_risk(events: List[Dict]) -> bool:
    """Return True if any HIGH_RISK_CATEGORIES item appears >= 3 times in the window."""

    def count_cat(target: str) -> int:
        return sum(1 for e in events if target in (e.get("class") or []))

    return any(count_cat(cat) >= 3 for cat in HIGH_RISK_CATEGORIES)


def is_moderate_risk(events: List[Dict]) -> bool:
    """
    Return True if any moderate-risk condition is satisfied in the window.
    Conditions and thresholds are controlled by the lists defined at the top:
    - _MODERATE_RISK_A_CATEGORIES / _MODERATE_RISK_A_THRESHOLD
    - _MODERATE_RISK_B_CATEGORIES / _MODERATE_RISK_B_THRESHOLD
    """

    def count_cat(target: str) -> int:
        return sum(1 for e in events if target in (e.get("class") or []))

    if any(
        count_cat(cat) >= _MODERATE_RISK_A_THRESHOLD
        for cat in _MODERATE_RISK_A_CATEGORIES
    ):
        return True
    if any(
        count_cat(cat) >= _MODERATE_RISK_B_THRESHOLD
        for cat in _MODERATE_RISK_B_CATEGORIES
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# Rule function (returns int 0–3, not bool)
# ---------------------------------------------------------------------------


def rule_risk(ts_rows: list) -> int:
    """
    Risk scoring from time-series rows for a single subject.
    Expects each row to contain '최종판단' (str/bool) and '세부카테고리' (list) keys.
    Returns risk level: 0, 1, 2, or 3.
    """
    if not ts_rows:
        return 0

    signals = normalize_signals(ts_rows)

    alive_categories = set(
        SERIOUS_ACCIDENTS + HIGH_RISK_CATEGORIES + MODERATE_RISK_CATEGORIES
    )

    events = []
    for sig, row in zip(signals, ts_rows):
        raw_cats = row.get("세부카테고리", [])
        all_cats = raw_cats if isinstance(raw_cats, list) else _parse_list(raw_cats)
        cats = [c for c in all_cats if c in alive_categories]
        events.append({"signal": sig, "class": cats})

    return has_risk_level_in_window(events)


def _count_serious_behaviors(events: List[Dict]) -> int:
    """
    Count how many serious_behavior cases (A–D) are met in the window.
    Each satisfied case contributes 1 to the count.

    Case A: "생계 유지가 어려운 사건" >= 1 in last 3
    Case B: "신상 정보 공개 또는 고지" >= 1 AND "직업 변화" >= 1
    Case C: "공격 행동" >= 1
    Case D: any of _CASE_D_TRIGGER_CATEGORIES (D1–D5) >= 1 AND any of _CASE_D_AND_CATEGORIES >= 1
    """

    def count_class(target: str, window: list = None) -> int:
        src = window if window is not None else events
        return sum(1 for e in src if target in (e.get("class") or []))

    recent3 = events[-3:]
    count = 0

    # Case A
    if any(count_class(cat, recent3) >= 1 for cat in _CASE_A_CATEGORIES):
        count += 1

    # Case B
    if any(count_class(cat) >= 1 for cat in _CASE_B_TRIGGER_CATEGORIES) and any(
        count_class(cat) >= 1 for cat in _CASE_B_AND_CATEGORIES
    ):
        count += 1

    # Case C
    if any(count_class(cat) >= 1 for cat in _CASE_C_CATEGORIES):
        count += 1

    # Case D
    if any(count_class(cat) >= 1 for cat in _CASE_D_TRIGGER_CATEGORIES) and any(
        count_class(cat) >= 1 for cat in _CASE_D_AND_CATEGORIES
    ):
        count += 1

    return count


def is_serious_behaviors(events: List[Dict]) -> int:
    """
    Count how many serious_behavior cases (A–D) are met in the window.
    Each satisfied case contributes 1 to the count.

    Case A: "생계 유지가 어려운 사건" >= 1 in last 3
    Case B: "신상 정보 공개 또는 고지" >= 1 AND "직업 변화" >= 1
    Case C: "공격 행동" >= 1
    Case D: any of _CASE_D_TRIGGER_CATEGORIES (D1–D5) >= 1 AND any of _CASE_D_AND_CATEGORIES >= 1
    """

    def count_class(target: str, window: list = None) -> int:
        src = window if window is not None else events
        return sum(1 for e in src if target in (e.get("class") or []))

    # Case A
    if any(count_class(cat, events) >= 1 for cat in _CASE_A_CATEGORIES):
        return True

    # Case B
    if any(count_class(cat) >= 1 for cat in _CASE_B_TRIGGER_CATEGORIES) and any(
        count_class(cat) >= 1 for cat in _CASE_B_AND_CATEGORIES
    ):
        return True

    # Case C
    if any(count_class(cat) >= 1 for cat in _CASE_C_CATEGORIES):
        return True

    # Case D
    if any(count_class(cat) >= 1 for cat in _CASE_D_TRIGGER_CATEGORIES) and any(
        count_class(cat) >= 1 for cat in _CASE_D_AND_CATEGORIES
    ):
        return True

    return False


def has_risk_level_in_window(events: List[Dict]) -> int:
    """
    Returns risk level 0–3 based on a window of events.

    Risk 0: no True in 최종판단
    Risk 1: any True in 최종판단 (but not Risk 2/3 criteria)
    Risk 2: >= 3 Trues AND last 3 has >= 1 True
    Risk 3 (independent, either condition triggers):
      - serious_behavior count > 0
      - HIGH_RISK_CATEGORIES appear > 3 times total
    """
    if not events:
        return 0

    signals = [e.get("signal", False) for e in events]
    total_true = sum(signals)
    recent3_true = sum(signals[-3:])

    if total_true == 0:
        return 0

    # risk 3
    # 심각한 사건 1회 발생
    # 목적 없는 이동 3회 이상 발생
    # 온라인 이성 또는 특정 커뮤니티 활동 집착 3회 이상 발생
    # if _count_serious_behaviors(events) > 0 or is_high_risk(events) >= 3:
    #     return 3

    if is_serious_behaviors(events) or is_high_risk(events):
        return 3

    # risk 2
    # 시계열 패턴 있음
    if total_true >= 3 and recent3_true >= 1:
        return 2

    if is_moderate_risk(events):
        return 2

    # 시계열 패턴 없음. 단일 패턴만 존재
    return 1
