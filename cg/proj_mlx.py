import mlx.core as mx


def project_points_mlx(points, k, dist_coeffs):
    XYZ = points.reshape(-1, 3).T
    N = XYZ.shape[1]
    uv = mx.zeros(shape=(N, 2))

    fx = k[0, 0]
    fy = k[1, 1]
    cx = k[0, 2]
    cy = k[1, 2]

    x = XYZ[0, :] / XYZ[2, :]
    y = XYZ[1, :] / XYZ[2, :]

    x2 = x**2
    y2 = y**2
    r2 = x2 + y2
    r4 = r2**2
    r6 = r4 * r2
    a1 = 2 * x * y
    a2 = r2 + 2 * x2
    a3 = r2 + 2 * y2
    k1, k2, p1, p2, k3 = dist_coeffs

    cdist = 1 + k1 * r2 + k2 * r4 + k3 * r6
    xd = x * cdist + p1 * a1 + p2 * a2
    yd = y * cdist + p1 * a3 + p2 * a1

    u = xd * fx + cx
    v = yd * fy + cy

    uv[:, 0] = u
    uv[:, 1] = v
    return uv
