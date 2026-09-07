import json
import mimetypes
import os
import tempfile
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from google import genai

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB local upload cap

DEFAULT_PROMPT = """Analyze this body-worn camera video as a security incident reviewer.

Return JSON only.

Detect these event types:
- physical_altercation
- verbal_confrontation
- threat
- weapon_visible
- suspected_theft
- fall_or_medical_event
- use_of_force
- guard_intervention
- police_or_ems_present
- other_security_incident

For each detected event return:
- event_type
- start_timestamp
- end_timestamp
- confidence (0-100)
- factual_description
- severity (0-5)
- evidence_basis: visible, audible, or both

Also return:
- overall_summary
- incident_detected (boolean)
- highest_severity (0-5)
- incident_report_should_exist (boolean)
- recommended_action
- uncertain_or_ambiguous_observations (array)
- events (array; empty if nothing is detected)

Rules:
- Do not infer facts that are not visible or audible.
- Separate observed facts from interpretations.
- If uncertain, lower confidence and explain the ambiguity.
- Use timestamps whenever possible.
- Do not identify people by real-world identity.
- Return valid JSON only. No markdown fences.
"""


def clean_json_text(text):
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def try_parse_json(text):
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned), True
    except Exception:
        return None, False


def state_name(file_obj):
    state = getattr(file_obj, "state", None)
    return getattr(state, "name", None) or str(state or "")


def wait_for_file(client, uploaded):
    deadline = time.time() + 15 * 60
    while time.time() < deadline:
        current = client.files.get(name=uploaded.name)
        state = state_name(current).upper()
        if "ACTIVE" in state:
            return current
        if "FAILED" in state or "ERROR" in state:
            raise RuntimeError(f"Google video processing failed: {state}")
        time.sleep(2)
    raise TimeoutError("Timed out waiting for Google to process the video.")


def upload_once(client, local_path, mime_type):
    uploaded = client.files.upload(
        file=local_path,
        config={"mime_type": mime_type} if mime_type else None,
    )
    return wait_for_file(client, uploaded)


def usage_tokens(interaction):
    usage = getattr(interaction, "usage", None) or getattr(interaction, "usage_metadata", None)
    if usage is None:
        return None
    return getattr(usage, "total_tokens", None) or getattr(usage, "total_token_count", None)


def run_interactions(client, model, video_file, prompt, processing):
    video_input = {
        "type": "video",
        "uri": video_file.uri,
        "mime_type": getattr(video_file, "mime_type", None) or "video/mp4",
    }
    if processing == "agentic":
        video_input["processing"] = "agentic"
    else:
        video_input["processing"] = "static"

    interaction = client.interactions.create(
        model=model,
        input=[
            video_input,
            {"type": "text", "text": prompt},
        ],
    )
    return getattr(interaction, "output_text", None) or "", usage_tokens(interaction)


def run_model(client, model, video_file, prompt, processing):
    started = time.perf_counter()
    try:
        output, total_tokens = run_interactions(client, model, video_file, prompt, processing)
        elapsed = round(time.perf_counter() - started, 2)
        parsed, valid = try_parse_json(output)

        events = None
        incident = None
        severity = None
        ir_needed = None
        if isinstance(parsed, dict):
            if isinstance(parsed.get("events"), list):
                events = len(parsed["events"])
            incident = parsed.get("incident_detected")
            severity = parsed.get("highest_severity")
            ir_needed = parsed.get("incident_report_should_exist")

        return {
            "model": model,
            "processing": processing,
            "status": "OK",
            "latency_sec": elapsed,
            "json_valid": valid,
            "events_detected": events,
            "incident_detected": incident,
            "highest_severity": severity,
            "ir_should_exist": ir_needed,
            "total_tokens": total_tokens,
            "parsed": parsed,
            "raw": output,
        }
    except Exception as exc:
        return {
            "model": model,
            "processing": processing,
            "status": f"ERROR: {exc}",
            "latency_sec": round(time.perf_counter() - started, 2),
            "json_valid": False,
            "events_detected": None,
            "incident_detected": None,
            "highest_severity": None,
            "ir_should_exist": None,
            "total_tokens": None,
            "parsed": None,
            "raw": "",
        }


@app.get("/")
def index():
    return render_template("index.html", default_prompt=DEFAULT_PROMPT)


@app.post("/api/analyze")
def analyze():
    api_key = request.form.get("api_key", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return jsonify({"error": "Gemini API key is required."}), 400

    uploaded = request.files.get("video")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "Upload a video."}), 400

    models_raw = request.form.get("models", "")
    models = [m.strip() for m in models_raw.replace(",", "\n").splitlines() if m.strip()]
    if not models:
        return jsonify({"error": "Enter at least one model ID."}), 400

    prompt = request.form.get("prompt", "").strip() or DEFAULT_PROMPT
    processing = request.form.get("processing", "agentic").strip()
    if processing not in {"agentic", "static"}:
        processing = "agentic"

    suffix = Path(uploaded.filename).suffix or ".mp4"
    mime_type = uploaded.mimetype or mimetypes.guess_type(uploaded.filename)[0] or "video/mp4"

    tmp_path = None
    remote_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            uploaded.save(tmp_path)

        client = genai.Client(api_key=api_key)
        remote_file = upload_once(client, tmp_path, mime_type)

        results = [
            run_model(client, model, remote_file, prompt, processing)
            for model in models
        ]

        return jsonify({
            "filename": uploaded.filename,
            "processing": processing,
            "results": results,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        # File cleanup is intentionally best-effort. Uploaded Gemini Files are temporary,
        # but deleting immediately can make debugging harder. Add client.files.delete(...)
        # here if you want strict immediate cleanup.


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=True)
