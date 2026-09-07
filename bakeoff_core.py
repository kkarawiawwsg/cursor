"""Helpers for the A66 video model bake-off (JSON parsing and result rows)."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

DEFAULT_PROMPT = """Analyze this body-worn camera video.

Return JSON only. Use this schema:

{
  "overall_summary": "string",
  "incident_detected": true,
  "incident_report_should_exist": true,
  "recommended_action": "string",
  "events": [
    {
      "event_type": "physical_altercation | verbal_confrontation | threat | weapon | suspected_theft | fall_or_medical_event | use_of_force | guard_intervention | police_ems_presence | other",
      "start_timestamp": "MM:SS",
      "end_timestamp": "MM:SS",
      "confidence": 0,
      "factual_description": "string",
      "severity": 0
    }
  ]
}

Detect:
- physical altercation
- verbal confrontation
- threats
- weapons
- suspected theft
- fall or medical event
- use of force
- guard intervention
- police/EMS presence
- whether an incident report should exist

For every detected event return:
- event_type
- start_timestamp
- end_timestamp
- confidence (0-100)
- factual_description
- severity (0-5)

Rules:
- Do not infer facts that are not visible or audible.
- Distinguish observed facts from uncertain interpretations.
- If an event is uncertain, lower confidence instead of asserting it as fact.
- Use timestamps whenever possible.
- Return valid JSON only, with no markdown fences.
- Always include an "events" array (empty if nothing is detected).
"""


def clean_json_text(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        else:
            text = text.replace("```json", "", 1).replace("```", "", 1)
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def parse_json_safe(text: str):
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned), True
    except Exception:
        return None, False


def count_events(parsed: Any) -> Optional[int]:
    if isinstance(parsed, dict):
        events = parsed.get("events")
        if isinstance(events, list):
            return len(events)
    return None


def summary_row(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Model": result["model"],
        "Status": result["status"],
        "Latency (sec)": result["latency_sec"],
        "JSON valid": result["json_valid"],
        "Events": result["events_detected"],
        "Incident?": result["incident_detected"],
        "IR should exist?": result["ir_should_exist"],
        "Prompt tokens": result["prompt_tokens"],
        "Output tokens": result["output_tokens"],
        "Total tokens": result["total_tokens"],
    }
