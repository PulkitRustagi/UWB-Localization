# coding=utf-8
from __future__ import print_function

import numpy as np


class KalmanFilter(object):
    def __init__(self):
        self.threshold = 100
        self.x = None
        self.P = None
        self.F = None
        self.H = None
        self.R = None
        self.Q = None
        self.S = None
        self.K = None
        self.I = np.identity(6)

    def init(self, x, P, F, H, R, Q):
        self.x = x
        self.P = P
        self.F = F
        self.H = H
        self.R = R
        self.Q = Q

    def predict(self):
        self.x = np.matmul(self.F, self.x)
        self.P = np.matmul(np.matmul(self.F, self.P), self.F.transpose()) + self.Q

    def update(self, anchor_p, anchor_range):
        d = anchor_range

        p = self.x[:3]

        anchor_p = anchor_p.reshape((3, -1))

        d_k = np.linalg.norm(p - anchor_p)

        self.H = np.zeros((1, 3 * 2))
        self.H[:, :3] = np.transpose(p - anchor_p) / d_k

        H_transpose = np.transpose(self.H)

        self.S = np.matmul(np.matmul(self.H, self.P), H_transpose) + self.R

        distance_sub = d - d_k
        inverse_S = np.linalg.inv(self.S)

        Dm = np.sqrt(distance_sub * inverse_S * distance_sub)

        print(Dm)

        if np.all(Dm < self.threshold):
            self.K = np.matmul(np.matmul(self.P, H_transpose), inverse_S)

            self.x = self.x + self.K * distance_sub
            self.P = np.matmul(self.I - np.matmul(self.K, self.H), self.P)
