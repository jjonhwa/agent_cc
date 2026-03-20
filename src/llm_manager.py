import streamlit as st
import os
import pandas as pd
from google import genai
from google.genai import types
from openai import AsyncOpenAI
from jinja2 import Template, Environment, meta
from src.utils import (
    post_process_llm_result,
    # clean_message,
    encode_media,
)
from configs.config import MAX_RETRY, MAX_LEN, TEMP
from src.time_series_rule_base_logic import RULE_REGISTRY
import asyncio


@st.cache_resource
def get_clients(api_key: str, base_url: str = None, provider: str = "Google Gemini"):
    if provider == "Google Gemini":
        return genai.Client(api_key=api_key)
    else:
        return AsyncOpenAI(api_key=api_key, base_url=base_url)


async def call_llm(system_instruction, user_content, model_name, api_provider, client):
    last_error = None
    for _ in range(MAX_RETRY):
        try:
            if api_provider == "Google Gemini":
                contents = (
                    [types.Part.from_text(text=user_content)]
                    if isinstance(user_content, str)
                    else user_content
                )
                res = await client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=TEMP,
                        max_output_tokens=MAX_LEN,
                    ),
                )
                raw = res.text
            else:
                res = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=TEMP,
                    seed=42,
                    max_completion_tokens=MAX_LEN,
                )
                raw = res.choices[0].message.content

            parsed = post_process_llm_result(raw)
            return parsed if isinstance(parsed, dict) else {"result": parsed}

        except Exception as e:
            print(f"LLM Error ({api_provider}): {e}")
            last_error = e

    return {"error": str(last_error)}


async def process_row(
    row,
    prompt,
    user_prompt,
    client,
    model_name,
    api_provider,
    mapping,
):
    text = user_prompt
    text = Template(text).render(**row.to_dict())

    try:
        res_dict = await call_llm(
            Template(prompt).render(**row.to_dict()),
            text,
            model_name,
            api_provider,
            client,
        )

        # Merge original data with LLM results
        out = {**row.to_dict(), **res_dict}

        return out

    except Exception as e:
        curr = row.to_dict()
        curr.update({"error": str(e)})
        return curr


async def run_text_inference(
    df_or_chunks,
    prompt,
    input_column,
    client,
    model_name,
    api_provider,
    mapping,
    max_concurrency=16,
):
    semaphore = asyncio.Semaphore(max_concurrency)

    async def sem_task(row):
        async with semaphore:
            return await process_row(
                row,
                prompt,
                input_column,
                client,
                model_name,
                api_provider,
                mapping,
            )

    # Convert single DF to a list of one DF if needed for uniform iteration
    chunks = [df_or_chunks] if isinstance(df_or_chunks, pd.DataFrame) else df_or_chunks
    for chunk in chunks:
        tasks = [asyncio.create_task(sem_task(row)) for _, row in chunk.iterrows()]
        completed = 0
        total = len(tasks)
        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1
            yield completed, total, result


async def process_time_series_row(
    row,
    prompt,
    user_prompt,
    client,
    model_name,
    api_provider,
    rule_method=None,
):
    row_dict = row.to_dict()
    id_val = row_dict.get("id")
    ts_rows = row_dict.get("time_series_rows", [])

    # Render user prompt against each data row, then concatenate
    rendered_parts = [Template(user_prompt).render(**ts_row) for ts_row in ts_rows]
    final_text = "\n\n".join(rendered_parts)

    method_name = rule_method or st.session_state.get(
        "ts_rule_method", next(iter(RULE_REGISTRY))
    )
    rule_fn = RULE_REGISTRY.get(method_name, next(iter(RULE_REGISTRY.values())))
    rule_result = rule_fn(ts_rows)

    try:
        res_dict = await call_llm(
            Template(prompt).render(id=id_val),
            final_text,
            model_name,
            api_provider,
            client,
        )

        return {
            "id": id_val,
            "window_index": row_dict.get("window_index"),
            "ts_input": "\n\n----------\n\n".join(rendered_parts),
            "rule_result": rule_result,
            **res_dict,
        }
    except Exception as e:
        return {
            "id": id_val,
            "window_index": row_dict.get("window_index"),
            "ts_input": "\n\n----------\n\n".join(rendered_parts),
            "rule_result": rule_result,
            "error": str(e),
        }


async def run_time_series_inference(
    df_or_chunks,
    prompt,
    user_prompt,
    client,
    model_name,
    api_provider,
    rule_method=None,
    max_concurrency=16,
):
    semaphore = asyncio.Semaphore(max_concurrency)

    async def sem_task(row):
        async with semaphore:
            return await process_time_series_row(
                row,
                prompt,
                user_prompt,
                client,
                model_name,
                api_provider,
                rule_method=rule_method,
            )

    chunks = [df_or_chunks] if isinstance(df_or_chunks, pd.DataFrame) else df_or_chunks
    for chunk in chunks:
        tasks = [asyncio.create_task(sem_task(row)) for _, row in chunk.iterrows()]
        completed = 0
        total = len(tasks)
        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1
            yield completed, total, result


async def run_multimodal_inference(
    row,
    system_prompt,
    user_prompt,
    client,
    model_name,
    api_provider,
    media_map,
    media_root_path=None,
):
    data_dict = row.to_dict()
    # Replace NaN with empty string for Jinja2 rendering
    data_dict = {k: ("" if pd.isna(v) else v) for k, v in data_dict.items()}

    def _render(template: str, data: dict) -> str:
        """
        Safely render a prompt template:
        - {key}   -> replaced with data[key]
        - {{ }}   -> kept as literal { }  (for JSON output format in the prompt)
        Avoids Jinja2's TemplateSyntaxError when prompts contain JSON structures.
        """
        import re

        # Replace {key} (single braces, word chars only) with value, skipping {{ }}
        def replacer(m):
            key = m.group(1)
            return str(data.get(key, m.group(0)))  # keep original if key not found

        rendered = re.sub(r"(?<!\{)\{([a-zA-Z_]\w*)\}(?!\})", replacer, template)
        # Unescape {{ -> { and }} -> }
        rendered = rendered.replace("{{", "{").replace("}}", "}")
        return rendered

    rendered_system = _render(system_prompt, data_dict)
    rendered_user = _render(user_prompt, data_dict)

    used_modalities = ["Text"]

    if api_provider == "Google Gemini":
        user_content = [
            types.Part.from_text(text=rendered_user),
        ]

        # Scan all columns for media filenames (e.g., raw_filename)
        for col, val in data_dict.items():
            if not val:
                continue
            val_str = str(val)

            media_file = None
            if val_str in media_map:
                media_file = media_map[val_str]
            elif media_root_path:
                full_path = os.path.join(media_root_path, val_str)
                if os.path.exists(full_path):
                    import mimetypes

                    mime, _ = mimetypes.guess_type(full_path)
                    with open(full_path, "rb") as f:
                        media_file = type(
                            "MediaFile",
                            (),
                            {
                                "getvalue": lambda self=None: f.read(),
                                "type": mime or "application/octet-stream",
                                "name": val_str,
                            },
                        )()

            if media_file:
                mime_type = getattr(media_file, "type", "application/octet-stream")
                user_content.append(
                    types.Part.from_bytes(
                        data=media_file.getvalue(),
                        mime_type=mime_type,
                    )
                )

                if "image" in mime_type:
                    used_modalities.append("Image")
                elif "video" in mime_type:
                    used_modalities.append("Video")
                elif "audio" in mime_type:
                    used_modalities.append("Audio")
    else:
        user_content = [{"type": "text", "text": rendered_user}]
        for col, val in data_dict.items():
            if not val:
                continue
            val_str = str(val)

            media_file = None
            if val_str in media_map:
                media_file = media_map[val_str]
            elif media_root_path:
                full_path = os.path.join(media_root_path, val_str)
                if os.path.exists(full_path):
                    import mimetypes

                    mime, _ = mimetypes.guess_type(full_path)
                    with open(full_path, "rb") as f:
                        content = f.read()
                        media_file = type(
                            "MediaFile",
                            (),
                            {
                                "getvalue": lambda self=None: content,
                                "type": mime or "application/octet-stream",
                                "name": val_str,
                            },
                        )()

            if media_file:
                data_uri = encode_media(media_file)
                mime = data_uri.split(";")[0].split(":")[-1]
                if "image" in mime:
                    user_content.append(
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    )
                    used_modalities.append("Image")
                elif "video" in mime:
                    user_content.append(
                        {"type": "video_url", "video_url": {"url": data_uri}}
                    )
                    used_modalities.append("Video")
                elif "audio" in mime:
                    user_content.append(
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": data_uri.split(",")[-1],
                                "format": mime.split("/")[-1],
                            },
                        }
                    )
                    used_modalities.append("Audio")

    used_modalities_str = ", ".join(sorted(list(set(used_modalities))))

    res = await call_llm(
        rendered_system, user_content, model_name, api_provider, client
    )
    if isinstance(res, dict):
        res["used_modalities"] = used_modalities_str

    return res


async def run_multimodal_batch(
    df_or_chunks,
    prompt,
    user_prompt,
    client,
    model_name,
    api_provider,
    media_map,
    media_root_path=None,
    max_concurrency=4,
):
    semaphore = asyncio.Semaphore(max_concurrency)

    async def sem_task(row):
        async with semaphore:
            res = await run_multimodal_inference(
                row,
                prompt,
                user_prompt,
                client,
                model_name,
                api_provider,
                media_map,
                media_root_path,
            )
            # Merge original row with LLM response
            out = {**row.to_dict(), **res}
            if "label" not in out:
                out["label"] = None
            return out

    # Convert single DF to a list of one DF if needed for uniform iteration
    chunks = [df_or_chunks] if isinstance(df_or_chunks, pd.DataFrame) else df_or_chunks

    for chunk in chunks:
        tasks = [asyncio.create_task(sem_task(row)) for _, row in chunk.iterrows()]
        completed = 0
        total = len(tasks)
        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1
            yield completed, total, result
