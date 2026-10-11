"""Clinical reference measurements (e.g. lab base-walk speeds) for real-data fixtures."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

REFERENCE_SPEEDS_PATH = Path(__file__).with_name("lab_walks_reference_speeds.json")


@lru_cache(maxsize=1)
def load_lab_walks_reference_speeds() -> dict:
    with REFERENCE_SPEEDS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def lab_walks_base_velocity(participant_id: str) -> Optional[float]:
    """Return clinical base-walk reference speed [m/s] for a participant, if known."""
    payload = load_lab_walks_reference_speeds()
    normalized = participant_id.strip().upper()
    value = payload["participants"].get(normalized)
    if value is None:
        return None
    return float(value)


def lab_walks_reference_participants() -> Dict[str, float]:
    payload = load_lab_walks_reference_speeds()
    return {str(key): float(value) for key, value in payload["participants"].items()}


def default_gait_speed_tolerance_relative() -> float:
    payload = load_lab_walks_reference_speeds()
    return float(payload.get("default_tolerance_relative", 0.25))
