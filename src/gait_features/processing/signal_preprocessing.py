from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from src.gait_features.config.extraction_backend import GaitExtractionBackendId

GRAVITY_M_PER_S2 = 9.80665


class AccelOutputUnit(Enum):
    G = "g"
    MPS2 = "m/s2"


class GyroOutputUnit(Enum):
    RAD_S = "rad/s"
    DEG_S = "deg/s"


@dataclass(frozen=True)
class BackendSignalProfile:
    accel_unit: AccelOutputUnit
    gyro_unit: GyroOutputUnit

    @staticmethod
    def for_backend(backend: GaitExtractionBackendId) -> "BackendSignalProfile":
        if backend == GaitExtractionBackendId.SKDH:
            return BackendSignalProfile(
                accel_unit=AccelOutputUnit.G,
                gyro_unit=GyroOutputUnit.RAD_S,
            )
        if backend == GaitExtractionBackendId.MOBGAP:
            return BackendSignalProfile(
                accel_unit=AccelOutputUnit.MPS2,
                gyro_unit=GyroOutputUnit.DEG_S,
            )
        raise ValueError(f"No signal profile defined for backend '{backend.value}'.")


def canonicalize_accel_unit(unit_value: str) -> Optional[str]:
    normalized = str(unit_value).strip().lower().replace(" ", "")
    aliases = {
        "g": "g",
        "gravities": "g",
        "gravity": "g",
        "m/s^2": "m/s2",
        "m/s2": "m/s2",
        "ms^-2": "m/s2",
        "m/s/s": "m/s2",
    }
    return aliases.get(normalized)


def canonicalize_gyro_unit(unit_value: str) -> Optional[str]:
    normalized = str(unit_value).strip().lower().replace(" ", "")
    aliases = {
        "deg/s": "deg/s",
        "deg/s^2": "deg/s",
        "degrees/s": "deg/s",
        "°/s": "deg/s",
        "rad/s": "rad/s",
        "radians/s": "rad/s",
    }
    return aliases.get(normalized)


def infer_accel_unit(accel: np.ndarray) -> str:
    median_norm = float(np.nanmedian(np.linalg.norm(accel, axis=1)))
    if 0.25 <= median_norm <= 2.5:
        return AccelOutputUnit.G.value
    return AccelOutputUnit.MPS2.value


def convert_accel(
    accel: np.ndarray,
    source_unit: str,
    target_unit: AccelOutputUnit,
) -> np.ndarray:
    accel = accel.astype(float)
    if np.any(~np.isfinite(accel)):
        raise ValueError("Accelerometer array contains non-finite values.")

    target = target_unit.value
    if source_unit == target:
        return accel
    if source_unit == AccelOutputUnit.G.value and target == AccelOutputUnit.MPS2.value:
        print(
            "Converting accelerometer data from g to m/s^2 for backend consumption."
        )
        return accel * GRAVITY_M_PER_S2
    if source_unit == AccelOutputUnit.MPS2.value and target == AccelOutputUnit.G.value:
        print(
            "Converting accelerometer data from m/s^2 to g for backend consumption."
        )
        return accel / GRAVITY_M_PER_S2
    raise ValueError(
        f"Unsupported accelerometer unit conversion: {source_unit!r} -> {target!r}."
    )


def convert_gyro(
    gyro: np.ndarray,
    source_unit: str,
    target_unit: GyroOutputUnit,
) -> np.ndarray:
    gyro = gyro.astype(float)
    if np.any(~np.isfinite(gyro)):
        raise ValueError("Gyroscope array contains non-finite values.")

    target = target_unit.value
    if source_unit == target:
        return gyro
    if source_unit == GyroOutputUnit.DEG_S.value and target == GyroOutputUnit.RAD_S.value:
        print(
            "Converting gyroscope data from deg/s to rad/s for backend consumption."
        )
        return np.radians(gyro)
    if source_unit == GyroOutputUnit.RAD_S.value and target == GyroOutputUnit.DEG_S.value:
        print(
            "Converting gyroscope data from rad/s to deg/s for backend consumption."
        )
        return np.degrees(gyro)
    raise ValueError(
        f"Unsupported gyroscope unit conversion: {source_unit!r} -> {target!r}."
    )
