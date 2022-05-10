# coding=utf-8
import numpy as np

from ukf.datapoint import DataType
from ukf.state import UKFState
from ukf.util import angle_diff


class MeasurementPredictor(object):
    """
    Class that helps calculating the estimated measurement vector
    """

    def __init__(self, sensor_std, N_SIGMA, WEIGHTS_M, WEIGHTS_C):
        """
        Setups MeasurementPredictor. Creates the R matrices given the sensor_std parameter as a diagonal matrix where
        the values are squared
        @param sensor_std: a dictionary with the sensor noise values. Each point in the dictorary is defined using its
        DataType id as its key, the number values the state contains (nz) and then a list of all the sensor noise
        values (std). An example is as follows

        >>> sensor_std = {
        ...   DataType.UWB: {
        ...       'std': [1],
        ...       'nz': 1
        ...   },
        ...   DataType.ODOMETRY: {
        ...     'std': [1, 1, 1, 1, 1, 1],
        ...     'nz': 6
        ...    },
        ... }

        @param N_SIGMA: the number of sigma points
        @param WEIGHTS_M: the weight distribution for the vector state calculations
        @param WEIGHTS_C: the weight distribution for the covariance matrix calculations
        """
        self.WEIGHTS_C = WEIGHTS_C
        self.sensor_std = sensor_std

        self.compute_R_matrix()

        self.WEIGHTS_M = WEIGHTS_M
        self.N_SIGMA = N_SIGMA

        self.z = None
        self.S = None
        self.sigma_z = None
        self.current_type = None

        self.R = None
        self.nz = None
        self.anchor_pos = None
        self.sensor_offset = None

    def initialize(self, data):
        """
        Initializes the parameters for the current sensor data type, such as the correct R matrices and the extra data
        information for the UWB sensors
        @param data: the current sensor measurement
        """
        sensor_type = data.data_type

        self.current_type = sensor_type

        self.R = self.sensor_std[sensor_type]["R"]
        self.nz = self.sensor_std[sensor_type]['nz']

        if sensor_type == DataType.UWB:
            self.anchor_pos = data.extra['anchor']
            self.sensor_offset = data.extra['sensor_offset']

    def rotation_matrix(self, angle):
        """
        Defines a 3D rotation matrix rotating the x and y by a certain angle/ angles.
        @param angle: the angle/angles to rotate by
        @return: a 3D rotation matrix as a numpy array of shape (3, 3, angle.size)
        """
        output = np.zeros((3, 3, angle.size))

        s = np.sin(angle)
        c = np.cos(angle)

        output[0, 0, :] = c
        output[1, 0, :] = s
        output[0, 1, :] = -s
        output[1, 1, :] = c
        output[2, 2, :] = 1

        return output

    def compute_sigma_z(self, sigma_x):
        """
        Calculates, for each of the sigma points, each of their associated estimated measurement value.
        @param sigma_x: the predicted state sigma points
        @return: a list of the calculated measurement values
        """
        sigma = np.zeros((self.nz, self.N_SIGMA))

        if self.current_type == DataType.LIDAR:
            sigma[UKFState.X] = sigma_x[UKFState.X]  # px
            sigma[UKFState.Y] = sigma_x[UKFState.Y]  # py
        elif self.current_type == DataType.UWB:
            sensor_pose = sigma_x[:UKFState.Z + 1]

            if self.sensor_offset is not None:
                angles = sigma_x[UKFState.YAW]
                rot = self.rotation_matrix(angles)

                offsets = np.einsum('ijn,j->in', rot, self.sensor_offset)

                sensor_pose = sensor_pose + offsets

            distances = np.linalg.norm(sensor_pose - self.anchor_pos.reshape((-1, 1)), axis=0)
            sigma[0] = distances
        elif self.current_type == DataType.ODOMETRY:
            sigma[UKFState.X] = sigma_x[UKFState.X]  # px
            sigma[UKFState.Y] = sigma_x[UKFState.Y]  # py
            sigma[UKFState.Z] = sigma_x[UKFState.Z]  # pz
            sigma[UKFState.V] = sigma_x[UKFState.V]  # v
            sigma[UKFState.YAW] = sigma_x[UKFState.YAW]  # theta
            sigma[UKFState.YAW_RATE] = sigma_x[UKFState.YAW_RATE]  # theta_yaw
        elif self.current_type == DataType.IMU:
            sigma[0] = sigma_x[UKFState.YAW]  # theta

        return sigma

    def compute_z(self, sigma):
        """
        Calculates the predicted sensor measurement by taking the weighted average of the sigma point's measurement
        estimations
        @param sigma: the sigma point's associated measurement estimation
        @return: the predicted sensor measurement
        """
        return np.dot(sigma, self.WEIGHTS_M)

    def compute_S(self, sigma, z):
        """
        Computes the sensor noise covariance matrix.
        @param sigma: the sigma point measurement estimates
        @param z: the predicted sensor measurement
        @return: the sensor noise covariance matrix
        """
        sub = np.subtract(sigma.T, z).T

        if self.current_type == DataType.ODOMETRY:
            sub[UKFState.YAW] = angle_diff(sigma[UKFState.YAW], z[UKFState.YAW])
        elif self.current_type == DataType.IMU:
            sub = angle_diff(sigma, z)

        return (np.matmul(self.WEIGHTS_C * sub, sub.T)) + self.R

    def process(self, sigma_x, data):
        """
        Calculates the estimated sensor measurement and sensor covariance based on the predicted sigma points
        @param sigma_x: the predicted sigma point matrix
        @param data: the sensor's measurement data
        """
        self.initialize(data)
        self.sigma_z = self.compute_sigma_z(sigma_x)
        self.z = self.compute_z(self.sigma_z)
        self.S = self.compute_S(self.sigma_z, self.z)

    def compute_R_matrix(self):
        """
        Creates and caches the sensor noise matrix R given the sensor_std[DataType]['std'] array.
        """
        for value in self.sensor_std:
            self.sensor_std[value]["R"] = np.diag(np.power(self.sensor_std[value]['std'], 2))
