from dataclasses import dataclass

from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument.instrument_identifier import (
    InstrumentIdentifier,
)


@dataclass
class IMUMetadata:
    imu_data_identifier: IMUDataIdentifier
    instrument_identifier: InstrumentIdentifier
