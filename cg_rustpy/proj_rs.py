from numpy.typing import NDArray
from ._cg_rustpy import project_points_rs as _project_points_rs


def project_points_rs(points: NDArray, k: NDArray, dist_coeffs: NDArray):
    return _project_points_rs(points, k, dist_coeffs)
