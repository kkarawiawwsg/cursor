# A66 Video Model Bake-Off

Upload one body-worn camera clip, run the same analysis prompt against multiple Gemini models, and compare outputs side-by-side.

The Flask app (`app.py`) is the main bake-off UI. It uploads the video once via the Gemini Files API, waits until the file is `ACTIVE`, then runs each model through the Interactions API with **agentic** or **static** video processing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set a Gemini API key (optional if you paste it in the UI):

```bash
export GEMINI_API_KEY=your_key
```

## Run (Flask)

```bash
python app.py
```

Open http://127.0.0.1:8080

1. Paste or confirm your Gemini API key.
2. List one model ID per line (defaults: `gemini-3.8-flash`, `gemini-3.7-flash`).
3. Choose **agentic** (default) or **static** video processing.
4. Upload a bodycam clip.
5. Click **Run model comparison**.

Download the comparison as CSV or JSON after the run finishes.

## Alternate UI (Streamlit)

```bash
streamlit run a66_video_model_bakeoff.py
```

## Tests

```bash
pytest
```
