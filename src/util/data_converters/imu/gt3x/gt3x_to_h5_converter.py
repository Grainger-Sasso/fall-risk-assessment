"""Convert ActiGraph ``.gt3x`` recordings to the pipeline's ``IMUData`` / ``imu_data.h5``.

Every file is checked for a misaligned ``log.bin`` record stream and repaired
in memory before parsing (see ``gt3x_log_repair``); without that, ``pygt3x``
can silently drop hours of data.

``info.txt`` carries subject identifiers (``Subject Name``, ``DateOfBirth``).
They are never read into ``GT3XInfo`` and never reach converted artifacts.
"""

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from pygt3x.reader import FileReader

from src.data_io.import_export.exporters.data.imu.imu_data_file_exporter import (
    IMUDataFileExporter,
)
from src.data_io.import_export.importers.data.imu.imu_data_importer import IMUDataImporter
from src.data_model.data.imu.imu_data import IMUData
from src.data_model.data.imu.metadata.imu_metadata import IMUMetadata
from src.data_model.data.imu.metadata.sensor_metadata import SensorMetadata
from src.data_model.data.imu.sensor_data import SensorData
from src.data_model.data.imu.uniaxial_sensor_data import UniaxialSensorData
from src.data_types.instrument.sensor_type import SensorType
from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument.instrument_identifier import InstrumentIdentifier
from src.identifiers.user.user_identifier import UserIdentifier
from src.util.data_converters.imu.gt3x.gt3x_log_repair import (
    LogRepairReport,
    repair_gt3x_bytes,
)
from src.util.mechanics.coordinates.system.anatomical.anatomical_axis import (
    AnatomicalAxis,
)
from src.util.mechanics.coordinates.system.anatomical.anatomical_coordinate_system import (
    AnatomicalCoordinateSystem,
)
from src.util.mechanics.coordinates.system.sensor.sensor_axis import SensorAxis
from src.util.mechanics.coordinates.system.sensor.sensor_coordinate_system import (
    SensorCoordinateSystem,
)

# info.txt keys that identify the subject; dropped at parse time.
IDENTIFYING_INFO_KEYS = frozenset({"Subject Name", "DateOfBirth"})
INSTRUMENT_NAME_PREFIX = "actigraph_"

# Device axes -> anatomical axes for an upright trunk-worn ActiGraph
# (Y along gravity, Z normal to the skin). Verified on the Upstate share:
# Z carries step-frequency energy (AP) and X stride-frequency energy (ML).
AXIS_MAP = (
    (SensorCoordinateSystem.X, AnatomicalCoordinateSystem.MEDIOLATERAL, 1),
    (SensorCoordinateSystem.Y, AnatomicalCoordinateSystem.VERTICAL, 2),
    (SensorCoordinateSystem.Z, AnatomicalCoordinateSystem.ANTEROPOSTERIOR, 3),
)


def _optional_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


@dataclass(frozen=True)
class GT3XInfo:
    """Non-identifying fields of a ``.gt3x`` ``info.txt``."""

    serial_number: str
    device_type: str
    firmware: Optional[str]
    sample_rate: int
    acceleration_scale: float
    acceleration_min: Optional[float]
    acceleration_max: Optional[float]
    timezone: Optional[str]
    sex: Optional[str]
    height_cm: Optional[float]
    mass_kg: Optional[float]
    age: Optional[float]
    limb: Optional[str]

    @classmethod
    def parse(cls, info_text: str) -> "GT3XInfo":
        fields: Dict[str, str] = {}
        for line in info_text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key in IDENTIFYING_INFO_KEYS:
                continue
            fields[key] = value.strip()
        missing = [k for k in ("Serial Number", "Device Type", "Sample Rate", "Acceleration Scale") if k not in fields]
        if missing:
            raise ValueError(f"info.txt is missing required fields: {missing}")
        return cls(
            serial_number=fields["Serial Number"],
            device_type=fields["Device Type"],
            firmware=fields.get("Firmware"),
            sample_rate=int(float(fields["Sample Rate"])),
            acceleration_scale=float(fields["Acceleration Scale"]),
            acceleration_min=_optional_float(fields.get("Acceleration Min")),
            acceleration_max=_optional_float(fields.get("Acceleration Max")),
            timezone=fields.get("TimeZone"),
            sex=fields.get("Sex") or None,
            height_cm=_optional_float(fields.get("Height")),
            mass_kg=_optional_float(fields.get("Mass")),
            age=_optional_float(fields.get("Age")),
            limb=fields.get("Limb") or None,
        )

    @property
    def instrument_name(self) -> str:
        return INSTRUMENT_NAME_PREFIX + self.device_type


@dataclass
class GT3XRecording:
    """A parsed recording: timestamps (s), acceleration (g, N x 3) and idle-sleep flags."""

    info: GT3XInfo
    time: np.ndarray
    acceleration_g: np.ndarray
    idle_mask: np.ndarray
    repair: LogRepairReport

    @property
    def n_samples(self) -> int:
        return int(self.time.shape[0])

    @property
    def duration_hours(self) -> float:
        return float(self.time[-1] - self.time[0]) / 3600.0 if self.n_samples > 1 else 0.0


def read_gt3x(source: Union[Path, str, bytes]) -> GT3XRecording:
    """Parse a ``.gt3x`` (path or raw bytes), repairing its record stream if needed."""
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    repaired, report = repair_gt3x_bytes(raw)
    del raw
    previous_level = logging.getLogger("pygt3x").level
    logging.getLogger("pygt3x").setLevel(logging.ERROR)
    try:
        with FileReader(io.BytesIO(repaired)) as reader:
            info = GT3XInfo.parse(reader.zipfile.read("info.txt").decode("utf-8", "replace"))
            acceleration = reader.acceleration
            if acceleration.shape[0] == 0:
                raise ValueError("No accelerometer samples could be decoded from log.bin.")
            if float(reader.info.acceleration_scale) != info.acceleration_scale:
                raise ValueError("Acceleration scale in info.txt disagrees with the parsed header.")
            time = np.ascontiguousarray(acceleration[:, 0], dtype=np.float64)
            # Raw counts / scale = g; counts are 12-bit integers, so float32 is exact.
            accel = (acceleration[:, 1:4] / info.acceleration_scale).astype(np.float32)
            idle = acceleration[:, 4].astype(np.uint8)
            del acceleration
    finally:
        logging.getLogger("pygt3x").setLevel(previous_level)
    return GT3XRecording(info=info, time=time, acceleration_g=accel, idle_mask=idle, repair=report)


class GT3XToH5Converter:
    """Builds ``IMUData`` from a ``GT3XRecording`` and exports it as ``imu_data.h5``."""

    def __init__(self) -> None:
        self.exporter = IMUDataFileExporter(as_arrays=True)
        self.importer = IMUDataImporter()

    @staticmethod
    def build_sensor_data(recording: GT3XRecording) -> SensorData:
        axes = [
            UniaxialSensorData(
                anatomical_axis=AnatomicalAxis(anatomical),
                sensor_axis=SensorAxis(sensor),
                data=np.ascontiguousarray(recording.acceleration_g[:, column - 1]),
            )
            for sensor, anatomical, column in AXIS_MAP
        ]
        return SensorData(
            data=axes,
            time=recording.time,
            idle_mask=recording.idle_mask,
            metadata=SensorMetadata(
                sensor_type=SensorType.ACCELEROMETER,
                sampling_rate=float(recording.info.sample_rate),
                unit="g",
            ),
        )

    def build_imu_data(
        self,
        recording: GT3XRecording,
        imu_id: IMUDataIdentifier,
        user_id: UserIdentifier,
    ) -> IMUData:
        sensor_data = self.build_sensor_data(recording)
        metadata = IMUMetadata(
            imu_data_identifier=imu_id,
            user_identifier=user_id,
            instrument_identifier=InstrumentIdentifier(
                recording.info.instrument_name, recording.info.serial_number
            ),
        )
        return IMUData(
            data=[sensor_data],
            metadata=metadata,
            start_time=float(recording.time[0]),
            end_time=float(recording.time[-1]),
        )

    def export(self, imu_data: IMUData, output_parent_dir: Path) -> Path:
        """Write ``imu_data.h5`` under ``output_parent_dir`` and confirm it reads back."""
        record_dir = self.exporter.export_data(Path(output_parent_dir), imu_data)
        header = self.importer.import_header(record_dir)
        if header.imu_data_identifier.value != imu_data.metadata.imu_data_identifier.value:
            raise IOError(f"IMU file in {record_dir} did not read back with the expected identifier.")
        return record_dir

    def convert(
        self,
        source: Union[Path, str, bytes],
        output_parent_dir: Path,
        imu_id: IMUDataIdentifier,
        user_id: UserIdentifier,
    ) -> Path:
        recording = read_gt3x(source)
        return self.export(self.build_imu_data(recording, imu_id, user_id), output_parent_dir)
