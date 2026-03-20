import sys
import os

# Add project root to sys.path to allow absolute imports when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import pandas as pd
import streamlit as st

from configs.config import TEXT_APP_NAME, MULTIMODAL_APP_NAME, TIME_SERIES_APP_NAME, RISK_APP_NAME, GEMINI_MODELS
from src.llm_manager import run_text_inference, run_multimodal_batch, run_time_series_inference
from src.ui_components import (
    render_sidebar,
    render_upload_section,
    render_results_section,
)


def init_state():
    """Initialize session state for both modes"""
    defaults = {
        "app_mode": TEXT_APP_NAME,
        "df": None,
        "results_df": None,
        "page": 1,
        "clients": None,
        "model_name": GEMINI_MODELS[0],
        "mapping": {},
        "media_map": {},
        "api_provider": "Google Gemini",
        "hl_base_col": "(none)",
        "hl_value_col": "(none)",
        "ts_rule_method": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    st.set_page_config(layout="wide", page_title="Unified Labeling App")
    init_state()
    render_sidebar()

    st.title(f"🚀 {st.session_state.app_mode}")
    if st.session_state.clients is None:
        st.write("Load the model first")
        return

    data_ctx = render_upload_section()

    if data_ctx["ready"]:
        if st.button("▶ Run Inference", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            results = []

            import time

            async def runner():
                total_rows = data_ctx.get("total_rows", 0)
                start_time = time.time()
                total_processed = 0

                if st.session_state.app_mode in (TIME_SERIES_APP_NAME, RISK_APP_NAME):
                    inference_gen = run_time_series_inference(
                        data_ctx["df"],
                        data_ctx["prompt"],
                        data_ctx["user_prompt"],
                        st.session_state.clients,
                        st.session_state.model_name,
                        st.session_state.api_provider,
                        rule_method=st.session_state.get("ts_rule_method"),
                    )
                elif st.session_state.app_mode == TEXT_APP_NAME:
                    inference_gen = run_text_inference(
                        data_ctx["df"],
                        data_ctx["prompt"],
                        data_ctx["user_prompt"],
                        st.session_state.clients,
                        st.session_state.model_name,
                        st.session_state.api_provider,
                        st.session_state.mapping,
                    )
                else:
                    # In multimodal mode, df might be a single DF or a generator
                    inference_gen = run_multimodal_batch(
                        data_ctx["df"],
                        data_ctx["prompt"],
                        data_ctx["user_prompt"],
                        st.session_state.clients,
                        st.session_state.model_name,
                        st.session_state.api_provider,
                        st.session_state.media_map,
                        media_root_path=data_ctx.get("media_root_path"),
                    )

                async for done, total, res in inference_gen:
                    results.append(res)
                    total_processed += 1

                    elapsed = time.time() - start_time
                    avg_time = elapsed / total_processed
                    remaining = total_rows - total_processed
                    etr = avg_time * remaining if total_rows > 0 else 0

                    # Convert ETR to minutes and seconds
                    mins, secs = divmod(int(etr), 60)
                    etr_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

                    pct = total_processed / total_rows if total_rows > 0 else 0
                    progress_bar.progress(min(pct, 1.0))
                    status_text.text(
                        f"Overall Progress: {total_processed}/{total_rows} ({pct:.1%}) | "
                        f"ETR: {etr_str}"
                    )

            asyncio.run(runner())
            results_df = pd.DataFrame(results).fillna("")
            if st.session_state.app_mode in (TIME_SERIES_APP_NAME, RISK_APP_NAME):
                results_df = results_df.sort_values(
                    ["id", "window_index"]
                ).reset_index(drop=True)
            st.session_state.results_df = results_df
            st.session_state.user_prompt = data_ctx.get("user_prompt", "")

            # Save result to history
            import json
            from datetime import datetime

            history_dir = "history"
            os.makedirs(history_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            st.session_state.results_df.to_csv(
                os.path.join(history_dir, f"{ts}.csv"),
                index=False,
                encoding="utf-8-sig",
            )
            with open(
                os.path.join(history_dir, f"{ts}.json"), "w", encoding="utf-8"
            ) as _f:
                json.dump(
                    {
                        "system_prompt": data_ctx.get("prompt", ""),
                        "user_prompt": data_ctx.get("user_prompt", ""),
                    },
                    _f,
                    ensure_ascii=False,
                    indent=2,
                )

            st.success("Analysis Complete!")
            st.session_state.page = 1
            st.rerun()

    if st.session_state.results_df is not None:
        render_results_section()
        st.divider()
        csv = st.session_state.results_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "💾 Download Results CSV", csv, "labeled_data.csv", "text/csv"
        )


if __name__ == "__main__":
    main()
