from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier


@dataclass
class IntervalOverlay:
    start_index: int
    end_index: int
    label: str = ""
    style: str = "default"


@dataclass
class SelectionState:
    selected_imu_id: Optional[IMUDataIdentifier] = None
    start_index: int = 0
    end_index: int = 0
    overlays: List[IntervalOverlay] = field(default_factory=list)


class SelectionStateStore:
    """Shared selection state with a small publish/subscribe API."""

    def __init__(self) -> None:
        self._state = SelectionState()
        self._subscribers: Dict[str, Callable[[SelectionState], None]] = {}

    @property
    def state(self) -> SelectionState:
        return self._state

    def subscribe(self, subscriber_id: str, callback: Callable[[SelectionState], None]) -> None:
        self._subscribers[subscriber_id] = callback

    def unsubscribe(self, subscriber_id: str) -> None:
        if subscriber_id in self._subscribers:
            del self._subscribers[subscriber_id]

    def set_selected_imu(self, imu_id: Optional[IMUDataIdentifier]) -> None:
        self._state.selected_imu_id = imu_id
        self._notify()

    def set_time_window(self, start_index: int, end_index: int) -> None:
        self._state.start_index = max(0, start_index)
        self._state.end_index = max(start_index, end_index)
        self._notify()

    def set_overlays(self, overlays: List[IntervalOverlay]) -> None:
        self._state.overlays = overlays
        self._notify()

    def _notify(self) -> None:
        for callback in self._subscribers.values():
            callback(self._state)
