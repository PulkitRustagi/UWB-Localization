# coding=utf-8
import numpy as np

from ukf.state import UKFState
from ukf.util import angle_diff


class StatePredictor(object):
    """
    A class that handles the state prediction step of the UKF following a CTRV model
    """

    def __init__(self, NX, N_SIGMA, N_AUGMENTED, VAR_SPEED_NOISE, VAR_YAW_RATE_NOISE, SCALE, WEIGHTS_M, WEIGHTS_C):
        """
        Setups the State Predictor class
        @param NX: the number of states, 6 for the CTRV model. x, y, z, v, theta, theta rate
        @param N_SIGMA: the number of sigma points to generate
        @param N_AUGMENTED: the number of augmented states, 8 for the CTRV model. x, y, z, v, theta, theta rate,
        linear acceleration, yaw rate acceleration
        @param VAR_SPEED_NOISE: the expect linear acceleration defined as a Gaussian distribution
        @param VAR_YAW_RATE_NOISE: the expected angular acceleration defined as a Gaussian distribution
        @param SCALE: the sigma point scale
        @param WEIGHTS_M: the weights for the state calculation
        @param WEIGHTS_C: the weights for the covariance matrix calculation
        """
        self.WEIGHTS_M = WEIGHTS_M
        self.WEIGHTS_C = WEIGHTS_C
        self.NX = NX
        self.SCALE = SCALE
        self.N_SIGMA = N_SIGMA
        self.N_AUGMENTED = N_AUGMENTED
        self.VAR_SPEED_NOISE = VAR_SPEED_NOISE
        self.VAR_YAW_RATE_NOISE = VAR_YAW_RATE_NOISE

        self.sigma = np.zeros((NX, N_SIGMA))
        self.x = np.zeros(NX)
        self.P = np.zeros((NX, N_SIGMA))

        self.YAW_RATE_THRESHOLD = 0.001

    def compute_augmented_sigma(self, x, P):
        """
        Creates the augmented sigma point matrix
        @param x: the current state vector
        @param P: the current covariance matrix
        @return: a matrix with the size N_AUGMENTED x N_SIGMA_POINTS
        """
        augmented_x = np.zeros(self.N_AUGMENTED)
        augmented_P = np.zeros((self.N_AUGMENTED, self.N_AUGMENTED))

        augmented_x[:self.NX] = x
        augmented_P[:self.NX, :self.NX] = P
        augmented_P[self.NX, self.NX] = self.VAR_SPEED_NOISE
        augmented_P[self.NX + 1, self.NX + 1] = self.VAR_YAW_RATE_NOISE

        L = np.linalg.cholesky(augmented_P)
        augmented_sigma = np.repeat(augmented_x[None], self.N_SIGMA, axis=0).T

        scaled_L = self.SCALE * L

        augmented_sigma[:, 1:self.N_AUGMENTED + 1] += scaled_L
        augmented_sigma[:, self.N_AUGMENTED + 1:] -= scaled_L

        return augmented_sigma

    def predict_sigma(self, augmented_sigma, dt):
        """
        Coverts the current augmented sigma matrix to the next predicted state given dt time.
        @param augmented_sigma: the current sigma point state matrix
        @param dt: the time that has elapsed
        @return: the sigma point matrix given dt time has passed
        """
        predicted_sigma = np.zeros((self.NX, self.N_SIGMA))

        px = augmented_sigma[UKFState.X]
        py = augmented_sigma[UKFState.Y]
        pz = augmented_sigma[UKFState.Z]
        speed = augmented_sigma[UKFState.V]
        yaw = augmented_sigma[UKFState.YAW]
        yaw_rate = augmented_sigma[UKFState.YAW_RATE]
        speed_noise = augmented_sigma[UKFState.LIN_ACCEL]
        yaw_rate_noise = augmented_sigma[UKFState.YAW_ACCEL]

        # PREDICT NEXT STEP USING CTRV Model

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        dt_2 = dt * dt

        # Acceleration noise
        p_noise = 0.5 * speed_noise * dt_2
        y_noise = 0.5 * yaw_rate_noise * dt_2

        # Velocity change
        d_yaw = yaw_rate * dt
        d_speed = speed * dt

        # Predicted speed = constant speed + acceleration noise
        p_speed = speed + speed_noise * dt

        # Predicted yaw
        p_yaw = yaw + d_yaw + y_noise

        # Predicted yaw rate
        p_yaw_rate = yaw_rate + yaw_rate_noise * dt

        mask = abs(yaw_rate) <= self.YAW_RATE_THRESHOLD
        mask_n = np.logical_not(mask)

        p_px = np.empty(self.N_SIGMA)
        p_py = np.empty(self.N_SIGMA)

        p_px[mask] = px[mask] + d_speed[mask] * cos_yaw[mask] + p_noise[mask] * cos_yaw[mask]
        p_py[mask] = py[mask] + d_speed[mask] * sin_yaw[mask] + p_noise[mask] * sin_yaw[mask]

        k = speed[mask_n] / yaw_rate[mask_n]
        theta = yaw[mask_n] + d_yaw[mask_n]
        p_px[mask_n] = px[mask_n] + k * (np.sin(theta) - sin_yaw[mask_n]) + p_noise[mask_n] * cos_yaw[mask_n]
        p_py[mask_n] = py[mask_n] + k * (cos_yaw[mask_n] - np.cos(theta)) + p_noise[mask_n] * sin_yaw[mask_n]

        predicted_sigma[UKFState.X] = p_px
        predicted_sigma[UKFState.Y] = p_py
        predicted_sigma[UKFState.Z] = pz
        predicted_sigma[UKFState.V] = p_speed
        predicted_sigma[UKFState.YAW] = p_yaw
        predicted_sigma[UKFState.YAW_RATE] = p_yaw_rate

        # ------------------

        return predicted_sigma

    def predict_x(self, predicted_sigma):
        """
        Calculates the predicted state vector using the predicted sigma points and their associated weight distribution
        @param predicted_sigma: the predicted state sigma point matrix
        @return: the predicted state vector in the shape N_X
        """
        return np.dot(predicted_sigma, self.WEIGHTS_M)

    def predict_P(self, predicted_sigma, predicted_x):
        """
        Calculates the predicted covariance matrix using the predicted sigma points and their associated weight
        distribution.
        @param predicted_sigma: the predicted state sigma point matrix
        @param predicted_x: the predicted state vector
        @return: returns the predicted covariance state matrix in the shape N_X x N_X
        """
        sub = np.subtract(predicted_sigma.T, predicted_x).T

        sub[UKFState.YAW] = angle_diff(predicted_sigma[UKFState.YAW], predicted_x[UKFState.YAW])

        return np.matmul(self.WEIGHTS_C * sub, sub.T)

    def process(self, x, P, dt):
        """
        Processes the current state and estimates its new positions in dt time
        @param x: the current state vector
        @param P: the current covariance matrix
        @param dt: the elapsed time
        """
        augmented_sigma = self.compute_augmented_sigma(x, P)
        self.sigma = self.predict_sigma(augmented_sigma, dt)
        self.x = self.predict_x(self.sigma)

        self.P = self.predict_P(self.sigma, self.x)
