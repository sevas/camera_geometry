import numpy as np
import pytest
from cg.camera_parameters import focal2fov, fov2focal


@pytest.mark.parametrize(
    "target_im_shape", [(480, 640), (1200, 1600)], ids=lambda x: f"{x[0]}x{x[1]}"
)
def test_roundtrip(target_im_shape):
    hfovs = np.arange(10, 170, 0.5)
    h, w = target_im_shape
    r = h / w
    vfovs = hfovs * r

    vfov2focal = np.vectorize(fov2focal)
    vfocal2fov = np.vectorize(focal2fov)

    computed_hfovs = vfocal2fov(vfov2focal(np.deg2rad(hfovs), w), w)
    assert np.allclose(hfovs, np.rad2deg(computed_hfovs))

    computed_vfovs = vfocal2fov(vfov2focal(np.deg2rad(vfovs), h), h)
    assert np.allclose(vfovs, np.rad2deg(computed_vfovs))
