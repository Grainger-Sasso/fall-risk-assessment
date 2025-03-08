from dataclasses import dataclass

from src.identifiers.imu.imu_data_identifier import IMUDataIdentifier
from src.identifiers.instrument.instrument_identifier import (
    InstrumentIdentifier,
)
from src.identifiers.user.user_identifier import UserIdentifier


@dataclass
class IMUMetadata:
    imu_data_identifier: IMUDataIdentifier
    user_identifier: UserIdentifier
    instrument_identifier: InstrumentIdentifier
