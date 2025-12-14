import numpy as np


def rot_x(theta: float) -> np.ndarray:
    m = np.array(
        [
            # fmt: off
            1,
            0,
            0,
            0,
            np.cos(theta),
            -np.sin(theta),
            0,
            np.sin(theta),
            np.cos(theta),
            # fmt: on
        ]
    ).reshape(3, 3)

    return m


def rot_y(theta: float) -> np.ndarray:
    m = np.array(
        [
            # fmt: off
            np.cos(theta),
            0,
            -np.sin(theta),
            0,
            1,
            0,
            np.sin(theta),
            0,
            np.cos(theta),
            # fmt: on
        ]
    ).reshape(3, 3)

    return m


def rot_z(theta: float) -> np.ndarray:
    # fmt: off
    m = np.array([
        np.cos(theta), -np.sin(theta), 0,
        np.sin(theta), np.cos(theta), 0,
        0, 0, 1,
        # fmt: on
    ]).reshape(3,3)

    return m
