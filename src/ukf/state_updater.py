# coding=utf-8
import numpy as np

from ukf.datapoint import DataType
from ukf.state import UKFState
from ukf.util import normalize, angle_diff


class StateUpdater(object):
    """
    Class that updates the current state vector using the predicted state and the actual measurement sensor value
    """

    def __init__(self, NX, N_SIGMA, WEIGHTS_M, WEIGHTS_C):
        """
        Setups the State Updater
        @param NX: the number of values the state vector has, 6 for a CTRV model. x, y, z, v, theta, theta rate
        @param N_SIGMA: the number of sigma points
        @param WEIGHTS_M: the weight distribution for the state calculation
        @param WEIGHTS_C: the weight distribution for the covariance matrix calculation
        """
        self.WEIGHTS_C = WEIGHTS_C
        self.N_SIGMA = N_SIGMA
        self.WEIGHTS_M = WEIGHTS_M
        self.NX = NX
        self.x = None
        self.P = None
        self.nis = None

    def compute_Tc(self, predicted_x, predicted_z, sigma_x, sigma_z, data_type):
        """
        Calculates the cross-correlation between sigma points in state space and measurement space
        @param predicted_x: the predicted state vector
        @param predicted_z: the predicted measurement vector
        @param sigma_x: the predicted sigma points
        @param sigma_z: the predicted measurement vector
        @param data_type: the datatype of the current measurement
        @return: the cross-correlation matrix
        """
        dx = np.subtract(sigma_x.T, predicted_x).T

        # normalize(dx, UKFState.YAW)
        dx[UKFState.YAW] = angle_diff(sigma_x[UKFState.YAW], predicted_x[UKFState.YAW])

        dz = np.subtract(sigma_z.T, predicted_z)

        if data_type == DataType.ODOMETRY:
            dz[:, UKFState.YAW] = angle_diff(sigma_z[UKFState.YAW], predicted_z[UKFState.YAW])
        elif data_type == DataType.IMU:
            dz[:, 0] = angle_diff(sigma_z[0], predicted_z[0])

        # normalize(dz, UKFState.YAW)

        return np.matmul(self.WEIGHTS_C * dx, dz)

    def update(self, z, S, Tc, predicted_z, predicted_x, predicted_P, data_type):
        """
        Updates the current position estimate given the actual sensor measurement
        @param z: the actual sensor value
        @param S: the predicted measurement covariance
        @param Tc: the cross-correlation between sigma points in state space and measurement space
        @param predicted_z: the predicted measurement vector
        @param predicted_x: the predicted state vector
        @param predicted_P: the predicted covariance matrix
        @param data_type: the current type of sensor
        """
        Si = np.linalg.inv(S)
        K = np.matmul(Tc, Si)

        dz = z - predicted_z

        if data_type == DataType.ODOMETRY:
            dz[UKFState.YAW] = angle_diff(np.atleast_1d(z[UKFState.YAW]), np.atleast_1d(predicted_z[UKFState.YAW]))
        elif data_type == DataType.IMU:
            dz[0] = angle_diff(np.atleast_1d(z[0]), np.atleast_1d(predicted_z[0]))

        # print(z[UKFState.YAW], predicted_z[UKFState.YAW], dz[UKFState.YAW])

        # Dm = np.sqrt(np.matmul(np.matmul(dz, Si), dz))

        self.x = predicted_x + np.matmul(K, dz)
        self.P = predicted_P - np.matmul(K, np.matmul(S, K.transpose()))
        self.nis = np.matmul(dz.transpose(), np.matmul(Si, dz))

    def process(self, predicted_x, predicted_z, z, S, predicted_P, sigma_x, sigma_z, data_type):
        """
        Process the current data to update the state vector and covariance matrix
        @param predicted_x: the predicted state vector
        @param predicted_z: the predicted measurement vector
        @param z: the actual sensor value
        @param S: the predicted measurement covariance
        @param predicted_P: the predicted covariance matrix
        @param sigma_x: the predicted state sigma points
        @param sigma_z: the predicted covariance matrix
        @param data_type: the current type of sensor
        """
        Tc = self.compute_Tc(predicted_x, predicted_z, sigma_x, sigma_z, data_type)
        self.update(z, S, Tc, predicted_z, predicted_x, predicted_P, data_type)

        normalize(self.x, UKFState.YAW)
