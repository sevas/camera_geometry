from pathlib import Path
import numpy as np
import plyfile
import pytest
import jax.numpy as jnp
from cg.math3d import rot_z, rot_y
from cg.proj_np import project_points_np
from cg.proj_nb import project_points_nb_parfor, project_points_nb, project_points_cu
from cg.proj_jax import project_points_jax
from cg.proj_cv import project_points_cv

try:
    import mlx.core as mx
    from cg.proj_mlx import project_points_mlx
except ImportError:
    mx = None
    project_points_mlx = None

from cg_rustpy.proj_rs import project_points_rs


DATA_DIR = Path(__file__).parent.parent / "data"


def load_ply(fpath: str | Path) -> plyfile.PlyData:
    with open(fpath, "rb") as f:
        return plyfile.PlyData.read(f)


@pytest.fixture()
def bunny_pcl():
    bunny = DATA_DIR / "bun_zipper.ply"

    bunny_ply = load_ply(bunny)
    bunny_pcl = np.stack(
        [
            bunny_ply["vertex"].data["x"],
            bunny_ply["vertex"].data["y"],
            bunny_ply["vertex"].data["z"],
        ],
        axis=-1,
    )

    # move things a bit so the model is in the fov and we don't need frustum culling
    theta = 1 * np.pi
    bunny_pcl_r = (rot_z(theta) @ bunny_pcl.T).T
    theta_y = np.pi
    bunny_pcl_r = (rot_y(theta_y) @ bunny_pcl_r.T).T

    bunny_pcl_r[:, 0] += 0.03
    bunny_pcl_r[:, 1] += 0.07
    bunny_pcl_r[:, 2] += 0.15

    print("Bunny pcl shape:", bunny_pcl_r.shape)
    return bunny_pcl_r


@pytest.fixture
def camera_params():
    h, w = 180, 240
    fx, fy = 70.0, 70.0
    cx, cy = w / 2, h / 2
    k1, k2, k3 = 0.02, -0.05, 0.09
    p1, p2 = 0.001, 0.002
    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
    dist_coeffs = np.array([k1, k2, p1, p2, k3])
    # tvec = np.array([0.0, 0.0, 0.0])
    # rvec = np.eye(3)

    return K, dist_coeffs


impl_under_test = [
    pytest.param((project_points_np, None), id="numpy"),
    pytest.param((project_points_cv, None), id="opencv"),
    pytest.param((project_points_nb, None), id="numba"),
    pytest.param((project_points_rs, None), id="rust"),
    pytest.param((project_points_nb_parfor, None), id="numba_parfor"),
    # pytest.param((project_points_cu, False), id="numba_cuda"),
    pytest.param((project_points_jax, jnp.asarray), id="jax"),
]

if mx is not None:
    impl_under_test.append(pytest.param((project_points_mlx, mx.array), id="mlx"))


@pytest.mark.parametrize("project_func", impl_under_test)
def test_benchmarkproject_points(benchmark, project_func, bunny_pcl, camera_params):
    k, dist = camera_params

    project_func, array_convert_func = project_func
    if array_convert_func:
        bunny_pcl = array_convert_func(bunny_pcl)
        k = array_convert_func(k)
        dist = array_convert_func(dist)

    _ = benchmark(project_func, bunny_pcl, k, dist)


def test_equivalence(bunny_pcl, camera_params):
    k, dist = camera_params
    uv_np = project_points_np(bunny_pcl, k, dist)
    uv_cv = project_points_cv(bunny_pcl, k, dist)
    uv_nb = project_points_nb(bunny_pcl, k, dist)
    # uv_nbcu = project_points_cu(bunny_pcl, k, dist)
    uv_rs = project_points_rs(bunny_pcl, k, dist)

    np.testing.assert_almost_equal(uv_np, uv_cv)
    np.testing.assert_almost_equal(uv_nb, uv_cv)
    # np.testing.assert_almost_equal(uv_nbcu, uv_cv)
    np.testing.assert_almost_equal(uv_rs, uv_cv)

    if mx is not None:
        uv_mx = project_points_mlx(bunny_pcl, k, dist)
        np.testing.assert_almost_equal(uv_mx, uv_cv, decimal=3)
