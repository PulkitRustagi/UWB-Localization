# coding=utf-8
class UKFState(object):
    """
    A Enum like object meant to represent what each index of the state vector represents
    """
    X = 0
    Y = 1
    Z = 2
    V = 3
    YAW = 4
    YAW_RATE = 5
    LIN_ACCEL = 6
    YAW_ACCEL = 7
