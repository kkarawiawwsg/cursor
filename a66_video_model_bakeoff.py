"""Streamlit app: upload one bodycam clip and compare Gemini video models."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

from bakeoff_core import (
    DEFAULT_PROMPT,
    count_events,
    parse_json_safe,
    summary_row,
)

st.set_page_config(page_title="A66 Video Model Bake-Off", layout="wide")

FILE_READY_TIMEOUT_SEC = 300
DEFAULT_MODELS = "gemini-3.8-flash\ngemini-3.7-flash"


def _file_state_name(remote_file) -> Optional[str]:
    state = getattr(remote_file, "state", None)
    if state is None:
        return None
    name = getattr(state, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(state, str):
        return state
    return str(state)


def upload_video(client: genai.Client, uploaded_file, timeout_sec: int = FILE_READY_TIMEOUT_SEC):
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        remote_file = client.files.upload(file=tmp_path)
        status_box = st.empty()
        deadline = time.monotonic() + timeout_sec

        while True:
            remote_file = client.files.get(name=remote_file.name)
            state = _file_state_name(remote_file)
            status_box.info(f"Google file processing state: {state or 'UNKNOWN'}")
            if state == "ACTIVE":
                status_box.success("Video processed and ready.")
                return remote_file
            if state in {"FAILED", "ERROR"}:
                raise RuntimeError(f"Video processing failed with state: {state}")
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for video processing after {timeout_sec}s "
                    f"(last state: {state or 'UNKNOWN'})."
                )
            time.sleep(2)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def run_model(
    client: genai.Client,
    model: str,
    remote_file,
    prompt: str,
    *,
    temperature: float,
    force_json: bool,
) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        config_kwargs: Dict[str, Any] = {"temperature": temperature}
        if force_json:
            config_kwargs["response_mime_type"] = "application/json"
        response = client.models.generate_content(
            model=model,
            contents=[remote_file, prompt],
            config=types.GenerateContentConfig(**config_kwargs),
        )
        latency = time.perf_counter() - started
        text = response.text or ""
        parsed, valid_json = parse_json_safe(text)

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
        total_tokens = getattr(usage, "total_token_count", None) if usage else None

        return {
            "model": model,
            "status": "OK",
            "latency_sec": round(latency, 2),
            "json_valid": valid_json,
            "events_detected": count_events(parsed),
            "incident_detected": parsed.get("incident_detected") if isinstance(parsed, dict) else None,
            "ir_should_exist": parsed.get("incident_report_should_exist")
            if isinstance(parsed, dict)
            else None,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "raw_output": text,
            "parsed_output": parsed,
        }
    except Exception as e:
        latency = time.perf_counter() - started
        return {
            "model": model,
            "status": f"ERROR: {e}",
            "latency_sec": round(latency, 2),
            "json_valid": False,
            "events_detected": None,
            "incident_detected": None,
            "ir_should_exist": None,
            "prompt_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "raw_output": "",
            "parsed_output": None,
        }


def render_results(results: List[Dict[str, Any]]) -> None:
    st.subheader("3. Comparison")
    df = pd.DataFrame([summary_row(r) for r in results])
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_col, json_col = st.columns(2)
    with csv_col:
        st.download_button(
            "Download comparison CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="a66_model_comparison.csv",
            mime="text/csv",
            key="download_csv",
        )
    with json_col:
        st.download_button(
            "Download full JSON results",
            data=json.dumps(results, indent=2, default=str).encode("utf-8"),
            file_name="a66_model_comparison.json",
            mime="application/json",
            key="download_json",
        )

    st.subheader("4. Outputs")
    tabs = st.tabs([r["model"] for r in results])
    for tab, result in zip(tabs, results):
        with tab:
            st.write(f"**Status:** {result['status']}")
            st.write(f"**Latency:** {result['latency_sec']} sec")
            st.write(f"**Valid JSON:** {result['json_valid']}")

            if result["parsed_output"] is not None:
                st.json(result["parsed_output"], expanded=True)
                with st.expander("Raw model output"):
                    st.code(result["raw_output"] or "(No output)", language="json")
            else:
                st.code(result["raw_output"] or "(No output)", language="text")

    st.info(
        "For a real benchmark, hand-label the clip first. Score each model on missed incidents, "
        "false positives, timestamp error, event classification, and whether it correctly decides "
        "that an incident report should exist."
    )


st.title("A66 Video Model Bake-Off")
st.caption(
    "Upload one bodycam clip once, run the same prompt against multiple Gemini models, "
    "and compare outputs side-by-side."
)

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Stored only in this Streamlit session unless you set GEMINI_API_KEY in your environment.",
    )
    model_text = st.text_area(
        "Models (one per line)",
        value=DEFAULT_MODELS,
        height=100,
    )
    st.caption("You can replace/add any Gemini model ID available to your API account.")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    force_json = st.checkbox(
        "Force JSON response",
        value=True,
        help="Asks the API to return application/json. Turn off if a model rejects that config.",
    )

uploaded_file = st.file_uploader(
    "Upload bodycam video",
    type=["mp4", "mov", "m4v", "webm", "mpeg", "mpg", "avi"],
)

prompt = st.text_area("Analysis prompt", value=DEFAULT_PROMPT, height=360)

run = st.button("Run model comparison", type="primary", use_container_width=True)

if run:
    st.session_state.pop("bakeoff_results", None)

    if not api_key:
        st.error("Enter your Gemini API key.")
        st.stop()
    if uploaded_file is None:
        st.error("Upload a video first.")
        st.stop()

    models = [m.strip() for m in model_text.splitlines() if m.strip()]
    if not models:
        st.error("Enter at least one model ID.")
        st.stop()

    client = genai.Client(api_key=api_key)

    st.subheader("1. Uploading video")
    try:
        remote_file = upload_video(client, uploaded_file)
    except Exception as e:
        st.error(f"Upload/processing failed: {e}")
        st.stop()

    st.subheader("2. Running models")
    results: List[Dict[str, Any]] = []
    progress = st.progress(0)

    for idx, model in enumerate(models, start=1):
        with st.spinner(f"Running {model}..."):
            results.append(
                run_model(
                    client,
                    model,
                    remote_file,
                    prompt,
                    temperature=temperature,
                    force_json=force_json,
                )
            )
        progress.progress(idx / len(models))

    try:
        client.files.delete(name=remote_file.name)
    except Exception:
        pass

    st.session_state["bakeoff_results"] = results

if st.session_state.get("bakeoff_results"):
    render_results(st.session_state["bakeoff_results"])
