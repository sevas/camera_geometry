import numpy as np
from numba import njit, cuda, prange


@njit
def project_points_nb(points, k, dist_coeffs):
    XYZ = points.reshape(-1, 3).T
    N = XYZ.shape[1]
    uv = np.empty(shape=(N, 2))

    fx = k[0, 0]
    fy = k[1, 1]
    cx = k[0, 2]
    cy = k[1, 2]

    for i in range(N):
        x = XYZ[0, i] / XYZ[2, i]
        y = XYZ[1, i] / XYZ[2, i]

        x2 = x**2
        y2 = y**2
        r2 = x2 + y2
        r4 = r2 * r2
        r6 = r4 * r2
        a1 = 2 * x * y
        a2 = r2 + 2 * x2
        a3 = r2 + 2 * y2
        k1, k2, p1, p2, k3 = dist_coeffs

        cdist = 1 + k1 * r2 + k2 * r4 + k3 * r6
        xd = x * cdist + p1 * a1 + p2 * a2
        yd = y * cdist + p1 * a3 + p2 * a1

        uv[i, 0] = xd * fx + cx
        uv[i, 1] = yd * fy + cy

    return uv


@njit(parallel=True)
def project_points_nb_parfor(points, k, dist_coeffs):
    XYZ = points.reshape(-1, 3).T
    N = XYZ.shape[1]
    uv = np.empty(shape=(N, 2))

    fx = k[0, 0]
    fy = k[1, 1]
    cx = k[0, 2]
    cy = k[1, 2]

    for i in prange(N):
        x = XYZ[0, i] / XYZ[2, i]
        y = XYZ[1, i] / XYZ[2, i]

        x2 = x**2
        y2 = y**2
        r2 = x2 + y2
        r4 = r2 * r2
        r6 = r4 * r2
        a1 = 2 * x * y
        a2 = r2 + 2 * x2
        a3 = r2 + 2 * y2
        k1, k2, p1, p2, k3 = dist_coeffs

        cdist = 1 + k1 * r2 + k2 * r4 + k3 * r6
        xd = x * cdist + p1 * a1 + p2 * a2
        yd = y * cdist + p1 * a3 + p2 * a1

        uv[i, 0] = xd * fx + cx
        uv[i, 1] = yd * fy + cy

    return uv


@cuda.jit
def project_points_cu(points, k, dist_coeffs, uv):
    i = cuda.grid(1)
    if i < points.shape[0]:
        x = points[i, 0] / points[i, 2]
        y = points[i, 1] / points[i, 2]

        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        a1 = 2 * x * y
        a2 = r2 + 2 * x * x
        a3 = r2 + 2 * y * y
        k1, k2, p1, p2, k3 = dist_coeffs

        cdist = 1 + k1 * r2 + k2 * r4 + k3 * r6
        xd = x * cdist + p1 * a1 + p2 * a2
        yd = y * cdist + p1 * a3 + p2 * a1

        fx = k[0, 0]
        fy = k[1, 1]
        cx = k[0, 2]
        cy = k[1, 2]
        uv[i, 0] = xd * fx + cx
        uv[i, 1] = yd * fy + cy
