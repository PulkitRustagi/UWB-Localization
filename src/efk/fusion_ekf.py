# coding=utf-8
import time

import numpy as np

from kalman_filter import KalmanFilter


class FusionEKF(object):
    def __init__(self):
        self.kalman_filter = KalmanFilter()
        self.is_initialized = False
        self.previous_timestamp = 0

        self.noise_ax = 1
        self.noise_ay = 1
        self.noise_az = 1

        self.noise_vector = np.diag([self.noise_ax, self.noise_ay, self.noise_az])

        self.kalman_filter.P = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        self.kalman_filter.x = np.zeros((6, 1))

        # self.uwb_std = 23.5 / 1000
        self.uwb_std = 23.5 / 10

        self.R_UWB = np.array([
            [self.uwb_std ** 2]
        ])

        self.kalman_filter.R = self.R_UWB

    def process_measurement(self, anchor_pose, anchor_distance):
        if not self.is_initialized:
            self.is_initialized = True
            self.previous_timestamp = time.time()
            return

        current_time = time.time()
        dt = current_time - self.previous_timestamp
        self.previous_timestamp = current_time

        self.calulate_F_matrix(dt)
        self.calculate_Q_matrix(dt)

        self.kalman_filter.predict()

        self.kalman_filter.update(anchor_pose, anchor_distance)

        # print("X:", self.kalman_filter.x)
        # print("P:", self.kalman_filter.P)
        # print(self.kalman_filter.Q)

    def calulate_F_matrix(self, dt):
        self.kalman_filter.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

    def calculate_Q_matrix(self, dt):
        a = [
            [(dt ** 4) / 4, (dt ** 3) / 2],
            [(dt ** 3) / 2, dt ** 2]
        ]

        self.kalman_filter.Q = np.kron(a, self.noise_vector)


if __name__ == '__main__':
    anchor_pose = np.array([10, 10, 10], dtype=float)
    anchor_pose2 = np.array([0, 10, 10], dtype=float)

    robot_pose = np.array([0, 0, 0], dtype=float)

    fusion = FusionEKF()

    s = time.time()

    for i in range(100):
        for a in [anchor_pose, anchor_pose2]:
            distance = np.linalg.norm(a - robot_pose)

            fusion.process_measurement(a, distance)

        c = time.time()

        robot_pose[0] += (c - s) * .1

        s = c

        print(robot_pose[0])
