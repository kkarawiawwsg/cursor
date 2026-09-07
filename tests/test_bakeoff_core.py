import json

from bakeoff_core import clean_json_text, count_events, parse_json_safe, summary_row


def test_clean_json_text_strips_markdown_fences():
    raw = """```json
{"events": []}
```"""
    assert clean_json_text(raw) == '{"events": []}'


def test_parse_json_safe_valid():
    parsed, ok = parse_json_safe('{"events": [{"event_type": "threat"}]}')
    assert ok is True
    assert parsed["events"][0]["event_type"] == "threat"


def test_parse_json_safe_invalid():
    parsed, ok = parse_json_safe("not json")
    assert ok is False
    assert parsed is None


def test_count_events_from_dict():
    assert count_events({"events": [{}, {}]}) == 2
    assert count_events({"events": []}) == 0
    assert count_events({"overall_summary": "none"}) is None
    assert count_events(["not", "a", "dict"]) is None


def test_summary_row_maps_fields():
    row = summary_row(
        {
            "model": "gemini-test",
            "status": "OK",
            "latency_sec": 1.23,
            "json_valid": True,
            "events_detected": 2,
            "incident_detected": True,
            "ir_should_exist": False,
            "prompt_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        }
    )
    assert row["Model"] == "gemini-test"
    assert row["Events"] == 2
    assert row["IR should exist?"] is False
    assert json.dumps(row)
