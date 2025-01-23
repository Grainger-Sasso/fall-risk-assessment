from src.util.mechanics.coordinates.system.anatomical.anatomical_axis import AnatomicalAxis
from src.util.mechanics.coordinates.system.anatomical.anatomical_coordinate_system import AnatomicalCoordinateSystem
from src.util.mechanics.coordinates.system.sensor.sensor_axis import SensorAxis
from src.util.mechanics.coordinates.system.sensor.sensor_coordinate_system import SensorCoordinateSystem

def main():
    ml_axis = AnatomicalAxis(AnatomicalCoordinateSystem.MEDIOLATERAL)
    print(ml_axis.name)
    x_axis = SensorAxis(SensorCoordinateSystem.X)
    print(x_axis.name)

if __name__ == '__main__':
    main()
