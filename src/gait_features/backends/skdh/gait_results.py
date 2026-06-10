from typing import Any, Dict


class GaitResults:
    """Legacy wrapper for SKDH gait dict output."""

    def __init__(self, results: Dict[str, Any]):
        self.data: Dict[str, Any] = results
