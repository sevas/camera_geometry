from typing import Tuple

import numpy as np


def make_img(img_shape: Tuple[int, int], u: np.ndarray, v: np.ndarray, pcl: np.ndarray) -> np.ndarray:
    h, w = img_shape
    im = np.ones(shape=img_shape) * 2
    uu = np.round(u).astype(int)
    vv = np.round(v).astype(int)

    inds = np.where((0 <= uu) * (uu < w) * (0 <= vv) * (vv < h))[0]
    uc = uu[inds]
    vc = vv[inds]

    im[vc, uc] = pcl[inds, 2]
    return im


# @njit
def make_img_zbuf(img_shape, u: np.ndarray, v: np.ndarray, pcl: np.ndarray) -> np.ndarray:
    h, w = img_shape
    im = np.ones(shape=img_shape) * 2
    uu = np.round(u).astype(int)
    vv = np.round(v).astype(int)

    inds = np.where((0 <= uu) * (uu < w) * (0 <= vv) * (vv < h))[0]

    for i in range(len(inds)):
        uc = uu[inds[i]]
        vc = vv[inds[i]]
        z = pcl[inds[i], 2]
        if im[vc, uc] >= z:
            im[vc, uc] = z

    return im
