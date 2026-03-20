import streamlit as st
import pandas as pd
import yaml
from configs.config import (
    TEXT_APP_NAME,
    MULTIMODAL_APP_NAME,
    TIME_SERIES_APP_NAME,
    RISK_APP_NAME,
    FIXED_FIELDS,
    PAGE_SIZE,
    VLLM_MODEL_NAME,
    VLLM_API_URL,
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    CHUNK_SIZE,
)
from src.utils import get_page_df, get_page_window, preprocess_time_series
from src.llm_manager import get_clients


def render_sidebar():
    with st.sidebar:
        st.title("🔧 Configuration")
        st.session_state.app_mode = st.radio(
            "Select Labeling Mode",
            [TEXT_APP_NAME, MULTIMODAL_APP_NAME, TIME_SERIES_APP_NAME, RISK_APP_NAME],
        )
        if st.session_state.app_mode == TIME_SERIES_APP_NAME:
            from src.time_series_rule_base_logic import RULE_REGISTRY

            rule_names = [k for k in RULE_REGISTRY.keys() if k != "Risk"]
            current = st.session_state.get("ts_rule_method") or rule_names[0]
            if current not in rule_names:
                current = rule_names[0]
            st.session_state.ts_rule_method = st.radio(
                "Rule Method",
                rule_names,
                index=rule_names.index(current),
                horizontal=True,
            )
        elif st.session_state.app_mode == RISK_APP_NAME:
            st.session_state.ts_rule_method = "Risk"
        st.divider()
        st.subheader("Model Settings")

        api_provider = st.selectbox(
            "API Provider",
            ["OpenAI-Compatible (vLLM)", "OpenAI", "Google Gemini"],
            index=["OpenAI-Compatible (vLLM)", "OpenAI", "Google Gemini"].index(
                st.session_state.get("api_provider", "Google Gemini")
            ),
        )
        st.session_state.api_provider = api_provider

        if api_provider == "OpenAI-Compatible (vLLM)":
            api_key = OPENAI_API_KEY if OPENAI_API_KEY else "EMPTY"
            # api_url = st.text_input("Base URL", value=VLLM_API_URL)
            api_url = "http://175.115.52.16:8003/v1"
            model_name = st.text_input("Model Name", value=VLLM_MODEL_NAME)
        elif api_provider == "OpenAI":
            api_key = st.text_input(
                "OpenAI API Key", value=OPENAI_API_KEY, type="password"
            )
            api_url = "https://api.openai.com/v1"
            from configs.config import OPENAI_MODELS

            model_name = st.selectbox("Model Name", options=OPENAI_MODELS)
        else:  # Google Gemini
            api_key = st.text_input(
                "Gemini API Key", value=GEMINI_API_KEY, type="password"
            )
            from configs.config import GEMINI_MODELS

            model_name = st.selectbox("Model Name", options=GEMINI_MODELS)
            api_url = None

        if st.button("🚀 Load Model"):
            if api_provider != "OpenAI-Compatible (vLLM)" and (
                not api_key or api_key == "EMPTY"
            ):
                st.error(f"Please enter a valid API Key for {api_provider}")
            else:
                try:
                    st.session_state.model_name = model_name
                    st.session_state.clients = get_clients(
                        api_key, api_url, provider=api_provider
                    )
                    st.session_state.api_provider = (
                        api_provider  # Ensure session state matches
                    )
                    st.success(f"{api_provider} client initialized")
                except Exception as e:
                    st.error(f"Failed to initialize {api_provider} client: {e}")

        # ── History ────────────────────────────────────────────────────────────
        import os as _os
        import json as _json_h

        history_dir = "history"
        if _os.path.exists(history_dir):
            history_files = sorted(
                [f for f in _os.listdir(history_dir) if f.endswith(".csv")],
                reverse=True,
            )
            if history_files:
                st.divider()
                st.subheader("📜 History")

                if "history_editing" not in st.session_state:
                    st.session_state.history_editing = None

                for fname in history_files:
                    fpath = _os.path.join(history_dir, fname)
                    prompt_fpath = fpath.replace(".csv", ".json")

                    # Resolve display name: custom "name" in JSON or fallback to timestamp
                    display_name = fname[:-4]
                    if _os.path.exists(prompt_fpath):
                        with open(prompt_fpath, "r", encoding="utf-8") as _pf:
                            _meta = _json_h.load(_pf)
                        display_name = _meta.get("name", display_name)

                    if st.session_state.history_editing == fname:
                        # ── Edit mode ──────────────────────────────────────
                        new_name = st.text_input(
                            "Rename",
                            value=display_name,
                            key=f"history_rename_{fname}",
                            label_visibility="collapsed",
                        )
                        col_save, col_cancel = st.columns([1, 1])
                        if col_save.button(
                            "✅",
                            key=f"history_save_{fname}",
                            help="Save name",
                            use_container_width=True,
                        ):
                            if _os.path.exists(prompt_fpath):
                                with open(prompt_fpath, "r", encoding="utf-8") as _pf:
                                    _meta = _json_h.load(_pf)
                            else:
                                _meta = {}
                            _meta["name"] = new_name
                            with open(prompt_fpath, "w", encoding="utf-8") as _pf:
                                _json_h.dump(_meta, _pf, ensure_ascii=False, indent=2)
                            st.session_state.history_editing = None
                            st.rerun()
                        if col_cancel.button(
                            "❌",
                            key=f"history_cancel_{fname}",
                            help="Cancel",
                            use_container_width=True,
                        ):
                            st.session_state.history_editing = None
                            st.rerun()
                    else:
                        # ── Normal mode ────────────────────────────────────
                        col_load, col_edit, col_del = st.columns([4, 1, 1])
                        if col_load.button(
                            display_name,
                            key=f"history_{fname}",
                            use_container_width=True,
                        ):
                            st.session_state.results_df = pd.read_csv(fpath).fillna("")
                            st.session_state.page = 1
                            # Restore prompts from companion JSON
                            if _os.path.exists(prompt_fpath):
                                with open(prompt_fpath, "r", encoding="utf-8") as _pf:
                                    prompt_data = _json_h.load(_pf)
                                st.session_state.user_prompt = prompt_data.get(
                                    "user_prompt", ""
                                )
                                # Update prompts.yaml so the text areas reflect the loaded prompts
                                _os.makedirs("prompts", exist_ok=True)
                                with open(
                                    "prompts/prompts.yaml", "w", encoding="utf-8"
                                ) as _yf:
                                    yaml.dump(
                                        {"combined": prompt_data},
                                        _yf,
                                        allow_unicode=True,
                                    )
                            st.rerun()
                        if col_edit.button(
                            "✏️", key=f"history_edit_{fname}", help="Rename this entry"
                        ):
                            st.session_state.history_editing = fname
                            st.rerun()
                        if col_del.button(
                            "🗑", key=f"history_del_{fname}", help="Delete this entry"
                        ):
                            _os.remove(fpath)
                            if _os.path.exists(prompt_fpath):
                                _os.remove(prompt_fpath)
                            st.rerun()


def render_upload_section():
    st.header(f"📂 Load Data: {st.session_state.app_mode}")
    data_context = {"ready": False}
    col1, col2 = st.columns([3, 1])

    import os

    prompts_data = {}
    if os.path.exists("prompts/prompts.yaml"):
        with open("prompts/prompts.yaml", "r", encoding="utf-8") as f:
            prompts_data = yaml.safe_load(f) or {}

    system_prompt = prompts_data.get("combined", {}).get("system_prompt", "")
    user_prompt = prompts_data.get("combined", {}).get("user_prompt", "")

    with col1:
        prompt = st.text_area("System Prompt", height=500, value=system_prompt)
        user_prompt = st.text_area("User Prompt", height=250, value=user_prompt)

        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_txt = f"[System Prompt]\n{prompt}\n\n[User Prompt]\n{user_prompt}"
        st.download_button(
            label="💾 Download Prompt",
            data=prompt_txt.encode("utf-8"),
            file_name=f"prompt_{ts}.txt",
            mime="text/plain",
        )

    with col2:
        if st.session_state.app_mode == TEXT_APP_NAME:
            uploaded = st.file_uploader(
                "Upload Data (CSV or XLSX)", type=["csv", "xlsx"]
            )
            if uploaded:
                if uploaded.name.endswith(".csv"):
                    df_raw = pd.read_csv(uploaded)
                else:
                    df_raw = pd.read_excel(uploaded)

                total_rows = len(df_raw)
                df_chunked = (
                    df_raw[i : i + CHUNK_SIZE] for i in range(0, total_rows, CHUNK_SIZE)
                )
                data_context.update(
                    {
                        "df": df_chunked,
                        "total_rows": total_rows,
                        "prompt": prompt,
                        "user_prompt": user_prompt,
                        "ready": True,
                    }
                )

        elif st.session_state.app_mode == TIME_SERIES_APP_NAME:
            uploaded = st.file_uploader(
                "Upload Data (CSV or XLSX)", type=["csv", "xlsx"]
            )
            if uploaded:
                if uploaded.name.endswith(".csv"):
                    df_raw = pd.read_csv(uploaded)
                else:
                    df_raw = pd.read_excel(uploaded)

                try:
                    df_ts = preprocess_time_series(df_raw)
                    st.info(
                        f"📈 Time series: {len(df_raw)} rows → "
                        f"{len(df_ts)} windows across {df_ts['id'].nunique()} unique IDs"
                    )
                    total_rows = len(df_ts)
                    df_chunked = (
                        df_ts[i : i + CHUNK_SIZE]
                        for i in range(0, total_rows, CHUNK_SIZE)
                    )
                    data_context.update(
                        {
                            "df": df_chunked,
                            "total_rows": total_rows,
                            "prompt": prompt,
                            "user_prompt": user_prompt,
                            "ready": True,
                        }
                    )
                except ValueError as e:
                    st.error(str(e))

        elif st.session_state.app_mode == RISK_APP_NAME:
            uploaded_files = st.file_uploader(
                "Upload Data Files (CSV or XLSX) — multiple files will be merged by (id, date)",
                type=["csv", "xlsx"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                try:
                    from src.rule_base.risk import (
                        merge_dataframes,
                    )

                    # dfs = load_and_merge_files(uploaded_files)
                    dfs = []
                    for f in uploaded_files:
                        if f.name.endswith(".csv"):
                            dfs.append(pd.read_csv(f))
                        else:
                            dfs.append(pd.read_excel(f))

                    df_raw = merge_dataframes(dfs)
                    st.info(
                        f"📂 Merged {len(uploaded_files)} file(s): "
                        f"{len(df_raw)} rows across {df_raw['id'].nunique()} unique IDs"
                    )
                    df_ts = preprocess_time_series(df_raw)
                    st.info(
                        f"📈 Time series: {len(df_raw)} rows → "
                        f"{len(df_ts)} windows across {df_ts['id'].nunique()} unique IDs"
                    )
                    total_rows = len(df_ts)
                    df_chunked = (
                        df_ts[i : i + CHUNK_SIZE]
                        for i in range(0, total_rows, CHUNK_SIZE)
                    )
                    data_context.update(
                        {
                            "df": df_chunked,
                            "total_rows": total_rows,
                            "prompt": prompt,
                            "user_prompt": user_prompt,
                            "ready": True,
                        }
                    )
                except (ValueError, KeyError) as e:
                    st.error(str(e))

        else:
            media_method = st.radio(
                "Media Loading Method",
                ["Upload Files", "Local Directory Path"],
                horizontal=True,
                help="Upload files is good for small sets. Local Directory Path is better for thousands of files.",
            )

            media_root_path = None
            if media_method == "Upload Files":
                tab_img, tab_aud, tab_vid = st.tabs(
                    ["🖼️ Images", "🎵 Audio", "🎥 Video"]
                )
                with tab_img:
                    imgs = st.file_uploader(
                        "Upload Images",
                        type=["png", "jpg", "jpeg", "webp"],
                        accept_multiple_files=True,
                        key="uploader_img",
                    )
                with tab_aud:
                    auds = st.file_uploader(
                        "Upload Audio",
                        type=["mp3", "wav", "ogg", "flac"],
                        accept_multiple_files=True,
                        key="uploader_aud",
                    )
                with tab_vid:
                    vids = st.file_uploader(
                        "Upload Video",
                        type=["mp4", "mov", "avi", "mkv"],
                        accept_multiple_files=True,
                        key="uploader_vid",
                    )

                # ── Stem-based grouping: build one row per unique stem ──
                # Collect all uploaded files grouped by stem (filename without extension)
                EXTENSIONS = {
                    "image_file": {".jpg", ".jpeg", ".png", ".webp"},
                    "audio_file": {".mp3", ".wav", ".ogg", ".flac"},
                    "video_file": {".mp4", ".mov", ".avi", ".mkv"},
                }
                stem_buckets: dict[
                    str, dict
                ] = {}  # stem -> {image_file: ..., audio_file: ..., ...}

                for files_list in [imgs, auds, vids]:
                    for f in files_list or []:
                        stem = os.path.splitext(f.name)[0]
                        ext = os.path.splitext(f.name)[1].lower()
                        col = next(
                            (c for c, exts in EXTENSIONS.items() if ext in exts), None
                        )
                        if stem not in stem_buckets:
                            stem_buckets[stem] = {"raw_filename": stem}
                        if col:
                            stem_buckets[stem][col] = f.name
                        # Keep media_map: full_filename -> file object
                        st.session_state.media_map[f.name] = f

                # Cross-modality consistency check
                if stem_buckets:
                    mod_cols_used = set()
                    for row in stem_buckets.values():
                        mod_cols_used |= set(row.keys()) - {"raw_filename"}

                    if len(mod_cols_used) >= 2:
                        incomplete = [
                            stem
                            for stem, row in stem_buckets.items()
                            if len(set(row.keys()) - {"raw_filename"})
                            < len(mod_cols_used)
                        ]
                        if incomplete:
                            st.warning(
                                f"⚠️ {len(incomplete)}개 스템에 모든 모달리티 파일이 없습니다: "
                                f"`{', '.join(sorted(incomplete)[:5])}{'...' if len(incomplete) > 5 else ''}`"
                            )
                        else:
                            st.success(
                                f"✅ {len(stem_buckets)}개 멀티모달 세트 구성 완료 (스템 기준 매칭)"
                            )
                    else:
                        st.success(f"✅ {len(stem_buckets)}개 미디어 세트 확인 완료")

                active_mods = stem_buckets  # used below for df_base

            else:
                media_root_path = st.text_input(
                    "Enter Local Media Directory Path",
                    placeholder="/home/user/data/media/",
                    help="The app will look for media files named in your CSV within this folder.",
                )
                if media_root_path and not os.path.isdir(media_root_path):
                    st.warning("⚠️ The path provided is not a valid directory.")

            st.divider()
            csv_file = st.file_uploader(
                "Upload Metadata (CSV or XLSX)", type=["csv", "xlsx"]
            )

            if csv_file:
                if csv_file.name.endswith(".csv"):
                    df_meta = pd.read_csv(csv_file)
                else:
                    df_meta = pd.read_excel(csv_file)

                # ── Media-First: build execution DataFrame from stem_buckets ──
                if media_method == "Upload Files" and stem_buckets:
                    # Convert stem_buckets to DataFrame (one row per stem)
                    df_base = pd.DataFrame(
                        sorted(stem_buckets.values(), key=lambda r: r["raw_filename"])
                    )

                    if "raw_filename" in df_meta.columns:
                        df_meta["_stem"] = df_meta["raw_filename"].apply(
                            lambda x: os.path.splitext(str(x))[0]
                        )
                        # Left-join: stem rows drive count, metadata fills in extra columns
                        df_base["_stem"] = df_base["raw_filename"]
                        df_joined = df_base.merge(
                            df_meta.drop(columns=["raw_filename"]),
                            on="_stem",
                            how="left",
                        ).drop(columns=["_stem"])
                        missing = df_joined["raw_filename"].isna().sum()
                        if missing:
                            st.warning(
                                f"⚠️ {missing}개 스템에 매핑되는 메타데이터가 없습니다."
                            )
                        else:
                            st.success(
                                f"✅ {len(df_joined)}개 멀티모달 세트-메타데이터 매핑 완료"
                            )
                    else:
                        st.warning(
                            "⚠️ 메타데이터에 `raw_filename` 컬럼이 없습니다. 미디어 파일 목록만으로 실행합니다."
                        )
                        df_joined = df_base

                    total_rows = len(df_joined)
                    df_chunked = (
                        df_joined[i : i + CHUNK_SIZE]
                        for i in range(0, total_rows, CHUNK_SIZE)
                    )

                elif media_root_path and os.path.isdir(media_root_path):
                    # Local directory: group all files by stem
                    EXTENSIONS_LOCAL = {
                        "image_file": {".jpg", ".jpeg", ".png", ".webp"},
                        "audio_file": {".mp3", ".wav", ".ogg", ".flac"},
                        "video_file": {".mp4", ".mov", ".avi", ".mkv"},
                    }
                    dir_stem_buckets: dict[str, dict] = {}
                    for fname in sorted(os.listdir(media_root_path)):
                        if not os.path.isfile(os.path.join(media_root_path, fname)):
                            continue
                        stem = os.path.splitext(fname)[0]
                        ext = os.path.splitext(fname)[1].lower()
                        col = next(
                            (c for c, exts in EXTENSIONS_LOCAL.items() if ext in exts),
                            None,
                        )
                        if stem not in dir_stem_buckets:
                            dir_stem_buckets[stem] = {"raw_filename": stem}
                        if col:
                            dir_stem_buckets[stem][col] = fname

                    df_base = pd.DataFrame(
                        sorted(
                            dir_stem_buckets.values(), key=lambda r: r["raw_filename"]
                        )
                    )

                    if "raw_filename" in df_meta.columns:
                        df_meta["_stem"] = df_meta["raw_filename"].apply(
                            lambda x: os.path.splitext(str(x))[0]
                        )
                        df_base["_stem"] = df_base["raw_filename"]
                        df_joined = df_base.merge(
                            df_meta.drop(columns=["raw_filename"]),
                            on="_stem",
                            how="left",
                        ).drop(columns=["_stem"])
                    else:
                        df_joined = df_base

                    total_rows = len(df_joined)
                    df_chunked = (
                        df_joined[i : i + CHUNK_SIZE]
                        for i in range(0, total_rows, CHUNK_SIZE)
                    )

                else:
                    # Fallback: use metadata as primary (no media uploaded)
                    total_rows = len(df_meta)
                    df_chunked = (
                        df_meta[i : i + CHUNK_SIZE]
                        for i in range(0, total_rows, CHUNK_SIZE)
                    )

                data_context.update(
                    {
                        "df": df_chunked,
                        "total_rows": total_rows,
                        "prompt": prompt,
                        "user_prompt": user_prompt,
                        "media_root_path": media_root_path,
                        "ready": True,
                    }
                )

    return data_context


def render_pagination_controls(total_rows, key_prefix="top"):
    if total_rows <= PAGE_SIZE:
        return
    total_pages = (total_rows - 1) // PAGE_SIZE + 1
    current = st.session_state.page
    pages = get_page_window(current, total_pages)
    cols = st.columns(len(pages) + 2)

    if cols[0].button("◀", disabled=current == 1, key=f"{key_prefix}_prev"):
        st.session_state.page -= 1
        st.rerun()

    for i, p in enumerate(pages, 1):
        with cols[i]:
            if p == "...":
                st.write("...")
            elif p == current:
                st.write(f"**{p}**")
            else:
                if st.button(str(p), key=f"{key_prefix}_page_{p}"):
                    st.session_state.page = p
                    st.rerun()

    if cols[-1].button("▶", disabled=current == total_pages, key=f"{key_prefix}_next"):
        st.session_state.page += 1
        st.rerun()


def _render_col_width_controls(ordered_cols, defaults):
    """Show per-column visibility checkboxes + width controls.

    Returns (visible_cols, weights) with hidden columns excluded from both.
    """
    with st.expander("⚙️ Column Widths", expanded=False):
        ctrl_cols = st.columns(len(ordered_cols))
        visible_cols = []
        weights = []
        for i, (col_name, default) in enumerate(zip(ordered_cols, defaults)):
            with ctrl_cols[i]:
                shown = st.checkbox(
                    col_name,
                    value=st.session_state.get(f"col_vis_{col_name}", True),
                    key=f"col_vis_{col_name}",
                )
                w = st.number_input(
                    "width",
                    min_value=1,
                    max_value=20,
                    value=st.session_state.get(f"col_w_{col_name}", default),
                    step=1,
                    key=f"col_w_{col_name}",
                    disabled=not shown,
                    label_visibility="collapsed",
                )
                if shown:
                    visible_cols.append(col_name)
                    weights.append(int(w))
    return visible_cols, weights


def render_results_section():
    if st.session_state.results_df is None:
        return
    df = st.session_state.results_df
    if df.empty:
        st.warning("⚠️ Inference returned no results.")
        return
    st.divider()
    st.subheader("📊 Labeling Interface")
    render_pagination_controls(len(df), key_prefix="top")
    display_df = get_page_df(df, st.session_state.page, PAGE_SIZE)
    st.divider()

    import re

    # ── Determine input vs generated columns ──────────────────────────────────
    all_cols = list(df.columns)

    _is_ts_mode = st.session_state.app_mode in (TIME_SERIES_APP_NAME, RISK_APP_NAME)

    if _is_ts_mode:
        ts_meta = {"id", "window_index", "ts_input"}
        input_cols = [c for c in all_cols if c in ts_meta]
        generated_cols = [c for c in all_cols if c not in ts_meta]
    else:
        user_prompt_template = st.session_state.get("user_prompt", "")
        template_vars = set(re.findall(r"\{\{\s*(\w+)\s*\}\}", user_prompt_template))
        input_cols = [c for c in all_cols if c in template_vars]
        generated_cols = [c for c in all_cols if c not in template_vars]

    if _is_ts_mode:
        ordered_cols = input_cols + generated_cols
        defaults = [3] * len(input_cols) + [2] * len(generated_cols)
        if ordered_cols:
            ordered_cols, col_weights = _render_col_width_controls(
                ordered_cols, defaults
            )
        else:
            col_weights = defaults
        if not col_weights:
            st.dataframe(display_df, use_container_width=True)
        else:
            header_cols = st.columns(col_weights)
            for hcol, name in zip(header_cols, ordered_cols):
                hcol.markdown(f"**{name}**")
            st.markdown("---")
            for _, row in display_df.iterrows():
                row_cols = st.columns(col_weights)
                for rcol, col_name in zip(row_cols, ordered_cols):
                    cell_val = row.get(col_name, "")
                    if isinstance(cell_val, str):
                        cell_val = cell_val.replace("\n", "\n\n")
                    rcol.write(cell_val)
                st.markdown("---")
        render_pagination_controls(len(df), key_prefix="bottom")
        return

    elif st.session_state.app_mode == TEXT_APP_NAME:
        # ── Highlight column controls ──────────────────────────────────────────
        import json as _json
        import ast as _ast

        def _highlight_text(text: str, phrases) -> str:
            """Wrap each phrase found in *text* with a <mark> tag."""
            if not text or not phrases:
                return text
            if isinstance(phrases, str):
                try:
                    parsed = _json.loads(phrases)
                    if isinstance(parsed, list):
                        phrases = [str(p) for p in parsed if p]
                    else:
                        phrases = [phrases]
                except (ValueError, TypeError):
                    # Fallback: handle Python-style lists (single quotes) from LLM output
                    try:
                        parsed = _ast.literal_eval(phrases)
                        if isinstance(parsed, list):
                            phrases = [str(p) for p in parsed if p]
                        else:
                            phrases = [phrases]
                    except (ValueError, SyntaxError):
                        phrases = [phrases]
            result = text
            for phrase in phrases:
                phrase = str(phrase).strip()
                if not phrase:
                    continue
                escaped = re.escape(phrase)
                result = re.sub(
                    escaped,
                    lambda m: (
                        f'<mark style="background:#FFD700;border-radius:3px;'
                        f'padding:1px 3px">{m.group()}</mark>'
                    ),
                    result,
                    flags=re.IGNORECASE,
                )
            return result

        text_cols = [
            c
            for c in all_cols
            if c
            not in {
                "image_file",
                "picture_name",
                "audio_file",
                "audio_name",
                "video_file",
                "video_name",
            }
        ]

        base_opts = ["(none)"] + text_cols
        hl_opts = ["(none)"] + text_cols

        if "highlight_active" not in st.session_state:
            st.session_state["highlight_active"] = False
        if "hl_base_col" not in st.session_state:
            st.session_state["hl_base_col"] = "(none)"
        if "hl_value_col" not in st.session_state:
            st.session_state["hl_value_col"] = "(none)"

        base_idx = (
            base_opts.index(st.session_state["hl_base_col"])
            if st.session_state["hl_base_col"] in base_opts
            else 0
        )
        hl_idx = (
            hl_opts.index(st.session_state["hl_value_col"])
            if st.session_state["hl_value_col"] in hl_opts
            else 0
        )

        hl_ctrl_cols = st.columns([2, 2, 1, 3])
        st.session_state["hl_base_col"] = hl_ctrl_cols[0].selectbox(
            "🔍 Base column (text to search in)",
            options=base_opts,
            index=base_idx,
        )
        st.session_state["hl_value_col"] = hl_ctrl_cols[1].selectbox(
            "✏️ Highlight column (value to mark)",
            options=hl_opts,
            index=hl_idx,
        )
        st.divider()
        btn_label = (
            "🟡 Highlight ON"
            if st.session_state["highlight_active"]
            else "⚪ Highlight OFF"
        )
        if hl_ctrl_cols[2].button(btn_label, key="hl_toggle_btn"):
            st.session_state["highlight_active"] = not st.session_state[
                "highlight_active"
            ]
            st.rerun()
        st.divider()
        highlight_active = (
            st.session_state["highlight_active"]
            and st.session_state["hl_base_col"] != "(none)"
            and st.session_state["hl_value_col"] != "(none)"
        )

        if not input_cols and not generated_cols:
            st.dataframe(display_df, use_container_width=True)
            render_pagination_controls(len(df), key_prefix="bottom")
            return

        # ── Build column layout: input cols | generated cols ──────────────────
        ordered_cols = input_cols + generated_cols
        defaults = [3] * len(input_cols) + [2] * len(generated_cols)
        if ordered_cols:
            ordered_cols, col_weights = _render_col_width_controls(
                ordered_cols, defaults
            )
        else:
            col_weights = defaults

        if not col_weights:
            st.dataframe(display_df, use_container_width=True)
        else:
            # Header row
            header_cols = st.columns(col_weights)
            for hcol, name in zip(header_cols, ordered_cols):
                hcol.markdown(f"**{name}**")
            st.markdown("---")

            for idx, row in display_df.iterrows():
                row_cols = st.columns(col_weights)
                for rcol, col_name in zip(row_cols, ordered_cols):
                    cell_val = row.get(col_name, "")
                    if isinstance(cell_val, str):
                        cell_val = cell_val.replace("\n", "\n\n")
                    if highlight_active and col_name == st.session_state["hl_base_col"]:
                        phrases = row.get(st.session_state["hl_value_col"], "")
                        highlighted = _highlight_text(str(cell_val), phrases)
                        rcol.markdown(highlighted, unsafe_allow_html=True)
                    else:
                        rcol.write(cell_val)
                st.markdown("---")

    else:
        # ── Multimodal: show media columns first, then generated cols ─────────
        media_col_map = {
            "image_file": st.image,
            "picture_name": st.image,
            "audio_file": st.audio,
            "audio_name": st.audio,
            "video_file": st.video,
            "video_name": st.video,
        }
        media_cols = [c for c in all_cols if c in media_col_map]
        non_media_input = [c for c in input_cols if c not in media_col_map]
        # Combine: media | other input | generated
        ordered_cols = media_cols + non_media_input + generated_cols
        defaults = (
            [3] * len(media_cols)
            + [2] * len(non_media_input)
            + [2] * len(generated_cols)
        )
        if ordered_cols:
            ordered_cols, col_weights = _render_col_width_controls(
                ordered_cols, defaults
            )
        else:
            col_weights = defaults

        if not col_weights:
            st.dataframe(display_df, use_container_width=True)
        else:
            header_cols = st.columns(col_weights)
            for hcol, name in zip(header_cols, ordered_cols):
                hcol.markdown(f"**{name}**")

            for idx, row in display_df.iterrows():
                st.markdown("---")
                row_cols = st.columns(col_weights)
                for rcol, col_name in zip(row_cols, ordered_cols):
                    val = row.get(col_name, "")
                    render_fn = media_col_map.get(col_name)
                    if render_fn and val:
                        if val in st.session_state.media_map:
                            with rcol:
                                render_fn(st.session_state.media_map[val])
                        else:
                            rcol.write(val)
                    else:
                        rcol.write(val)

    render_pagination_controls(len(df), key_prefix="bottom")
