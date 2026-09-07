# A66 Video Model Bake-Off

Upload one body-worn camera clip, run the same analysis prompt against multiple Gemini models, and compare outputs side-by-side.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set a Gemini API key (optional if you paste it in the sidebar):

```bash
export GEMINI_API_KEY=your_key
```

## Run

```bash
streamlit run a66_video_model_bakeoff.py
```

1. Paste or confirm your Gemini API key in the sidebar.
2. List one model ID per line (defaults: `gemini-3.8-flash`, `gemini-3.7-flash`).
3. Upload a bodycam clip.
4. Click **Run model comparison**.

The app uploads the video once, waits until Google marks the file `ACTIVE`, then runs each model with the same prompt. Results stay on screen across reruns (including CSV/JSON downloads).

## Tests

```bash
pytest
```
