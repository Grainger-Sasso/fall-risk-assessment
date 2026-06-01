from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.figure import Figure
from src.data_model.data.imu.sensor_data import SensorData
from src.data_visualization.suite.state.selection_state_store import IntervalOverlay
from src.util.mechanics.coordinates.system.anatomical.anatomical_coordinate_system import (
    AnatomicalCoordinateSystem,
)


AXIS_MAP: Dict[str, AnatomicalCoordinateSystem] = {
    "vertical": AnatomicalCoordinateSystem.VERTICAL,
    "mediolateral": AnatomicalCoordinateSystem.MEDIOLATERAL,
    "anteroposterior": AnatomicalCoordinateSystem.ANTEROPOSTERIOR,
}

COLOR_MAP = {
    "vertical": "#1f77b4",
    "mediolateral": "#ff7f0e",
    "anteroposterior": "#2ca02c",
}


@dataclass
class IMUPlotEngine:
    """Render IMU plots onto a shared matplotlib figure."""

    figure: Figure

    def render(
        self,
        sensor_data: SensorData,
        enabled_axes: Dict[str, bool],
        start_index: int,
        end_index: int,
        overlays: Optional[List[IntervalOverlay]] = None,
    ) -> None:
        self.figure.clear()
        ax_time = self.figure.add_subplot(211)
        ax_mag = self.figure.add_subplot(212, sharex=ax_time)

        time_values = sensor_data.time
        if len(time_values) == 0:
            ax_time.set_title("No IMU samples")
            self.figure.tight_layout()
            return

        start_index = max(0, min(start_index, len(time_values) - 1))
        end_index = max(start_index + 1, min(end_index, len(time_values)))
        window = slice(start_index, end_index)
        t = time_values[window]

        stacked_components = []
        for axis_key, axis_enum in AXIS_MAP.items():
            if not enabled_axes.get(axis_key, False):
                continue
            component = sensor_data.get_data_by_anatomical_axis(axis_enum).data[window]
            stacked_components.append(component)
            ax_time.plot(t, component, label=axis_key, color=COLOR_MAP[axis_key], linewidth=0.8)

        if stacked_components:
            magnitude = np.sqrt(np.sum(np.square(np.vstack(stacked_components)), axis=0))
            ax_mag.plot(t, magnitude, color="#9467bd", linewidth=0.9, label="magnitude")
            ax_mag.legend(loc="upper right")

        ax_time.set_title(f"{sensor_data.metadata.sensor_type.value} triaxial stream")
        ax_time.set_ylabel(f"Value ({sensor_data.metadata.unit})")
        ax_mag.set_ylabel("Magnitude")
        ax_mag.set_xlabel("Time")
        ax_time.grid(alpha=0.2)
        ax_mag.grid(alpha=0.2)
        if stacked_components:
            ax_time.legend(loc="upper right")

        if overlays:
            for overlay in overlays:
                self._draw_overlay(ax_time, ax_mag, time_values, overlay, start_index, end_index)

        self.figure.tight_layout()

    @staticmethod
    def _draw_overlay(
        ax_time,
        ax_mag,
        time_values: np.ndarray,
        overlay: IntervalOverlay,
        start_index: int,
        end_index: int,
    ) -> None:
        clamped_start = max(start_index, min(overlay.start_index, len(time_values) - 1))
        clamped_end = max(clamped_start, min(overlay.end_index, len(time_values) - 1))
        if clamped_end <= start_index or clamped_start >= end_index:
            return
        x0, x1 = time_values[clamped_start], time_values[clamped_end]
        for axis in (ax_time, ax_mag):
            axis.axvspan(x0, x1, color="#d62728", alpha=0.12)
        if overlay.label:
            ax_time.text(x0, ax_time.get_ylim()[1], overlay.label, fontsize=8, va="top")
