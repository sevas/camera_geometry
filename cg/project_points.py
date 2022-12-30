import numpy as np
from numba import njit


def project_points_np_pinhole(points, k, dist_coeffs):
    uvw = k @ points.reshape(-1, 3).T
    u = uvw[0, :] / uvw[2, :]
    v = uvw[1, :] / uvw[2, :]

    return u, v


def project_points_np(points, k, dist_coeffs):
    XYZ = points.reshape(-1, 3).T
    x = XYZ[0, :] / XYZ[2, :]
    y = XYZ[1, :] / XYZ[2, :]

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
    u = xd * fx + cx
    v = yd * fy + cy

    return np.stack([u, v], axis=1)


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

        uv[i, 0] = xd * fx + cx
        uv[i, 1] = yd * fy + cy

    return uv


def project_points_cv(points, k, dist_coeffs):
    import cv2

    tvec = np.array([0.0, 0.0, 0.0])
    rvec = np.eye(3)
    cv_projected, _ = cv2.projectPoints(
        points, cameraMatrix=k, distCoeffs=dist_coeffs, rvec=rvec, tvec=tvec
    )
    uv = np.squeeze(cv_projected)
    return uv
