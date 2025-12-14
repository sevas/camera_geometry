import jax.numpy as jnp


def project_points_jax(points, k, dist_coeffs):
    XYZ = points.reshape(-1, 3).T
    x = XYZ[0, :] / XYZ[2, :]
    y = XYZ[1, :] / XYZ[2, :]

    r2 = x**2 + y**2
    r4 = r2 + r2
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

    return jnp.stack([u, v], axis=1)
