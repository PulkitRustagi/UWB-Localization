# coding=utf-8
import numpy as np


def normalize(d, index):
    """
    In place normalizes angle values to be between -pi to pi, at a specific index at array
    @param d: The array to inplace normalize the angle values
    @param index: the index that the angle values are located
    """
    d[index] = (d[index] + np.pi) % (2 * np.pi) - np.pi


def angle_diff(start, end):
    """
    Calculates the bounded angle difference between the start and end value
    @param start: the starting angle, in radians
    @param end: the ending angle, in radians
    @return: the bounded angle difference between -180 and 180, in radians
    """
    diff = (end - start + np.pi) % (2 * np.pi) - np.pi

    diff[diff < -np.pi] += 2 * np.pi

    return diff

    # d[index] %= 2 * np.pi
    # mask = np.abs(d[index]) > np.pi
    # d[index, mask] -= (np.pi * 2)
