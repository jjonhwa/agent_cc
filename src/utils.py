import re
import ast
import json
import base64
import mimetypes
from typing import Union, List, Iterable


def encode_media(file) -> str:
    """
    Encode media file to base64 string with data URI scheme.
    Supports images, audio, video.
    """
    if file is None:
        return ""

    try:
        # Streamlit UploadedFile has .read() and .type
        bytes_data = file.getvalue()
        mime_type = getattr(file, "type", None)
        filename = getattr(file, "name", None)

        if not mime_type and filename:
            mime_type, _ = mimetypes.guess_type(filename)

    except AttributeError:
        # Fallback for bytes or other file-like objects
        if isinstance(file, bytes):
            bytes_data = file
            mime_type = "application/octet-stream"
        else:
            return ""

    if not mime_type:
        mime_type = "application/octet-stream"

    base64_str = base64.b64encode(bytes_data).decode("utf-8")
    return f"data:{mime_type};base64,{base64_str}"


# def clean_message(msg: str) -> str:
#     """
#     지정된 항목을 메시지에서 제거합니다.

#     :param msg: 원본 메시지 문자열
#     :return: 제거된 항목이 없는 클린 메시지
#     """
#     remove_keys = [
#         "동거인유무:",
#         "정서상태:",
#         "경제상태:",
#         "근무형태:",
#         "생업:",
#         "사회생활현황:",
#         "신상정보에 대한 보호관찰관 의견:",
#         "생업종사:",
#         "면담시간 준수여부:",
#         "재범존재여부:",
#         "재범 및 준수사항 위반 사실:",
#         "장치점검 부착:",
#         "장치점검 휴대:",
#         "장치점검 재택감독:면담태도:",
#         "동거인관계:",
#         "면담내용:",
#         "주거지:",
#         "면담내용:",
#         "면담 내용:",
#         "재범 및 수사기관 조사사실:",
#         "출장개요:",
#         "정신적:",
#         "재범, 준수사항 위반 여부:",
#         "출장목적:",
#         "출장지:",
#         "경보명:",
#         "지도대상:",
#         "기타상세:",
#         "가족관계:",
#         "생업관련:",
#     ]

#     lines = msg.split("\n")
#     cleaned_lines = [
#         line for line in lines if not any(key in line for key in remove_keys)
#     ]
#     return "\n".join(cleaned_lines)


def highlight_substrings(
    text: str,
    substrings: Union[str, List[str], Iterable],
    color: str = "#FFD966",
) -> str:
    """
    지정된 substring들을 텍스트 내에서 하이라이트합니다.

    - list/iterable: 각 요소를 substring으로 처리
    - string: 단일 substring으로 처리
    - 기타 타입: string으로 변환 후 단일 substring으로 처리
    """

    if substrings is None:
        return text

    # --- substrings 타입 정규화 ---
    if isinstance(substrings, str):
        substrings = [substrings]
    elif isinstance(substrings, Iterable):
        substrings = [str(s) for s in substrings if s]
    else:
        substrings = [str(substrings)]

    if not substrings:
        return text

    # Remove duplicates & sort by length (longer first prevents partial overlaps)
    substrings = sorted(set(substrings), key=len, reverse=True)

    for sub in substrings:
        if not sub or sub not in text:
            continue
        escaped = re.escape(sub)
        text = re.sub(
            escaped,
            f"<span style='background-color:{color}; font-weight:bold;'>{sub}</span>",
            text,
        )

    return text


def format_hms(seconds: Union[int, float]) -> str:
    """
    초 단위를 HH:MM:SS 형식으로 변환합니다.

    :param seconds: 경과 시간 (초)
    :return: 문자열 형식 HH:MM:SS
    """
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def post_process_llm_result(output_str: str) -> dict:
    """
    LLM 결과 문자열을 dict로 변환합니다.
    JSON 또는 파이썬 dict 문자열 모두 지원하며, 마크다운 코드 블록(```json)을 자동으로 제거합니다.

    :param output_str: LLM 결과 문자열
    :return: dict
    :raises ValueError: 문자열을 파싱할 수 없는 경우
    """
    # 마크다운 코드 블록 제거
    if output_str.strip().startswith("```"):
        import re

        # 시작 부분의 ```json 또는 ``` 제거
        output_str = re.sub(
            r"^```[a-z]*\n?", "", output_str.strip(), flags=re.IGNORECASE
        )
        # 끝 부분의 ``` 제거
        output_str = re.sub(r"\n?```$", "", output_str)

    output_str = output_str.strip()

    try:
        # print("출력 예시:\n", output_str) # 디버깅용 (필요시 주석 해제)
        return json.loads(output_str)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(output_str)
        except (ValueError, SyntaxError):
            import re as _re

            # Truncated JSON: try to recover by removing the last incomplete field
            try:
                fixed = _re.sub(
                    r',?\s*"[^"]*":\s*"[^"]*$', "", output_str, flags=_re.DOTALL
                )
                fixed = fixed.rstrip(",").rstrip()
                if not fixed.endswith("}"):
                    fixed = fixed + "\n}"
                if fixed.startswith("{"):
                    recovered = json.loads(fixed)
                    recovered["__truncated__"] = True
                    print("⚠ 잘린 응답 복구:", output_str[:80], "...")
                    return recovered
            except Exception:
                pass
            # Fallback: return as raw text so pipeline doesn't crash
            print("파싱 실패 데이터:", output_str)
            return {"raw_response": output_str, "error": "parse_failed"}


def get_page_window(
    current: int, total_pages: int, window: int = 2
) -> List[Union[int, str]]:
    """
    페이지 번호 리스트를 생성, 많은 페이지일 경우 슬라이딩 윈도우 적용.

    예시: [1, '...', 43, 44, 45, 46, 47, '...', 90]

    :param current: 현재 페이지
    :param total_pages: 총 페이지 수
    :param window: 현재 페이지 주변에 표시할 페이지 수
    :return: 페이지 번호와 '...'를 포함한 리스트
    """
    pages = []

    if total_pages <= (window * 2 + 5):
        return list(range(1, total_pages + 1))

    pages.append(1)

    if current > window + 3:
        pages.append("...")

    start = max(2, current - window)
    end = min(total_pages - 1, current + window)

    pages.extend(range(start, end + 1))

    if current < total_pages - (window + 2):
        pages.append("...")

    pages.append(total_pages)

    return pages


def get_page_df(df, page: int, page_size: int):
    """
    지정된 페이지에 해당하는 DataFrame 슬라이스를 반환합니다.

    :param df: 전체 DataFrame
    :param page: 페이지 번호 (1부터 시작)
    :param page_size: 페이지당 행 개수
    :return: 해당 페이지의 DataFrame
    """
    start = (page - 1) * page_size
    end = start + page_size
    return df.iloc[start:end]


# def get_page_window(current, total_pages, window=2):
#     """
#     Returns a list like:
#     [1, '...', 43, 44, 45, 46, 47, '...', 90]
#     """
#     pages = []

#     if total_pages <= (window * 2 + 5):
#         return list(range(1, total_pages + 1))

#     pages.append(1)

#     if current > window + 3:
#         pages.append("...")

#     start = max(2, current - window)
#     end = min(total_pages - 1, current + window)

#     pages.extend(range(start, end + 1))

#     if current < total_pages - (window + 2):
#         pages.append("...")

#     pages.append(total_pages)

#     return pages


# def get_page_df(df, page, page_size):
#     start = (page - 1) * page_size
#     end = start + page_size
#     return df.iloc[start:end]


def preprocess_time_series(
    df: "pd.DataFrame", window_size: int = 10, stride: int = 1
) -> "pd.DataFrame":
    """
    Pre-process DataFrame for time series mode.
    Groups by 'id', sorts each group by 'date', then applies a sliding window
    (window_size=10, stride=1) to produce one row per window per id.
    Each result row holds 'id', 'window_index', and 'time_series_rows' (list of dicts).
    """
    import pandas as pd

    if "id" not in df.columns:
        raise ValueError("Time series mode requires a column named 'id'.")
    if "date" not in df.columns:
        raise ValueError("Time series mode requires a column named 'date'.")

    rows = []
    for id_val, group in df.groupby("id", sort=False):
        group_sorted = group.sort_values("date").reset_index(drop=True)
        records = group_sorted.to_dict("records")
        n = len(records)

        id_str = str(id_val)
        if n <= window_size:
            rows.append({
                "id": id_str,
                "window_index": 0,
                "time_series_rows": records,
            })
        else:
            for start in range(0, n - window_size + 1, stride):
                rows.append({
                    "id": id_str,
                    "window_index": start,
                    "time_series_rows": records[start : start + window_size],
                })

    return pd.DataFrame(rows)


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"
