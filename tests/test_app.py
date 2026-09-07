import io

from app import app, clean_json_text, try_parse_json


def test_clean_json_text_strips_fences():
    raw = """```json
{"events": []}
```"""
    assert clean_json_text(raw) == '{"events": []}'


def test_try_parse_json_valid_and_invalid():
    parsed, ok = try_parse_json('{"incident_detected": true}')
    assert ok is True
    assert parsed["incident_detected"] is True

    parsed, ok = try_parse_json("not-json")
    assert ok is False
    assert parsed is None


def test_index_renders_prompt():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "A66 Video Model Bake-Off" in html
    assert "incident_report_should_exist" in html
    assert 'name="processing"' in html


def test_analyze_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = app.test_client()
    response = client.post("/api/analyze", data={"models": "gemini-3.8-flash"})
    assert response.status_code == 400
    assert "API key" in response.get_json()["error"]


def test_analyze_requires_video():
    client = app.test_client()
    response = client.post(
        "/api/analyze",
        data={"api_key": "test-key", "models": "gemini-3.8-flash"},
    )
    assert response.status_code == 400
    assert "video" in response.get_json()["error"].lower()


def test_analyze_requires_model(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = app.test_client()
    data = {
        "api_key": "test-key",
        "models": "   ",
        "video": (io.BytesIO(b"fake-bytes"), "clip.mp4"),
    }
    response = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "model" in response.get_json()["error"].lower()
