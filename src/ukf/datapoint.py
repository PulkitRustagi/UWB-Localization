# coding=utf-8
class DataType(object):
    """
    A Enum type class meant to represent the sensor id
    """
    LIDAR = 1
    ODOMETRY = 2
    UWB = 3
    IMU = 4
    GROUND_TRUTH = 0


class DataPoint(object):
    """
    A class meant to represent a data point to pass into the position estimation system
    """

    def __init__(self, data_type, measurement_data, timestamp, extra=None):
        """
        Setups the DataPoint object
        @param data_type: the sensor DataType id
        @param measurement_data: the primary sensor data, ie.e for UWB the range measurement
        @param timestamp: the timestamp that the datapoint arrived
        @param extra: a variable for extra data to use, i.e. the anchor position for UWB sensors
        """
        self.extra = extra
        self.data_type = data_type
        self.measurement_data = measurement_data
        self.timestamp = timestamp

    def __repr__(self):
        """
        Generates a user readable string representation of the object
        @return:
        """
        return str(self.data_type) + ":" + str(self.timestamp) + ": " + str(self.measurement_data)
