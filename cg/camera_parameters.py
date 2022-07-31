import numpy as np


def fov2focal(fov: float, size: int) -> float:
    return size / (np.tan(float(fov) / 2)) / 2


def focal2fov(focal: float, size: int) -> float:
    return 2 * np.arctan2(float(size) / 2, focal)
