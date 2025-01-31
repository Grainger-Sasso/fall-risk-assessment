from dataclasses import dataclass
from datetime import datetime
from typing import List

from src.data_model.data.user.user_data import UserData
from src.data_model.data.imu.imu_data import IMUData


@dataclass
class Entry:
    date: datetime
    administrator: str
    user_data_id: str
    imu_data_id = str
 