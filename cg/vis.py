from dataclasses import dataclass
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float = 0
    fy: float = 0
    cx: float = 0
    cy: float = 0
    k1: float = 0
    k2: float = 0
    k3: float = 0
    p1: float = 0
    p2: float = 0
    width: int = 0
    height: int = 0


def compute_pixel_rays(
    fx,
    fy,
    cx,
    cy,
    width,
    height,
    k1: float = 0,
    k2: float = 0,
    k3: float = 0,
    p1: float = 0,
    p2: float = 0,
    normalize=True,
):
    """
    Compute per-pixel ray origins and directions in camera coordinates.

    Args:
      fx, fy, cx, cy: camera intrinsics
      width, height: image size
      k1, k2, k3: radial distortion coefficients
      p1, p2: tangential distortion coefficients
      normalize: whether to return unit direction vectors

    Returns:
      origins: (height, width, 3) array of ray origins (all zeros in camera space)
      dirs: (height, width, 3) array of ray direction vectors
    """
    xs = np.arange(width)
    ys = np.arange(height)
    px, py = np.meshgrid(xs, ys)  # px: (H, W), py: (H, W)

    # Normalized image coordinates
    x = (px - cx) / fx
    y = (py - cy) / fy

    # Apply distortion
    r2 = x**2 + y**2
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    x_distorted = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    y_distorted = y * radial + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

    z = np.ones_like(x)
    dirs = np.stack((x_distorted, y_distorted, z), axis=-1)  # (H, W, 3)
    if normalize:
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        dirs = dirs / norms
    origins = np.zeros_like(dirs)
    return origins, dirs


def main():
    app = pg.mkQApp()
    # create 3d view
    view = gl.GLViewWidget()
    view.show()
    view.setWindowTitle("3D View")
    view.resize(1280, 720)

    g = gl.GLGridItem()
    view.addItem(g)

    cam_int = CameraIntrinsics(
        fx=500 ,
        fy=500 ,
        cx=320,
        cy=240,
        width=640,
        height=480,
        k1=0.1,
        k2=0.2,
        k3=0.01,
        p1=0.02,
        p2=0.03,
    )

    # cam_int = CameraIntrinsics(
    #     fx=500,
    #     fy=500,
    #     cx=320//10,
    #     cy=240//10,
    #     width=640//10,
    #     height=480//10,
    #     k1=0.1,
    #     k2=0.2,
    #     k3=0.01,
    #     p1=0.02,
    #     p2=0.03,
    # )

    origins, dirs = compute_pixel_rays(
        cam_int.fx,
        cam_int.fy,
        cam_int.cx,
        cam_int.cy,
        cam_int.width,
        cam_int.height,
        cam_int.k1,
        cam_int.k2,
        cam_int.k3,
        cam_int.p1,
        cam_int.p2,
    )


    # add sensor plane
    pw = cam_int.width
    ph = cam_int.height

    vertices = np.array(
        [
            [-pw, -ph, -1],
            [pw, -ph, -1],
            [pw, ph, -1],
            [-pw, ph, -1],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    plane = gl.GLMeshItem(
        vertexes=vertices,
        faces=faces,
        color=(0.01, 0.07, 0.1, 0.3),
        smooth=False,
        drawEdges=True,
        edgeColor=(0, 0.5, 1, 0.1),
        computeNormals=True,
        shader="balloon",
    )
    view.addItem(plane)


    def make_plane_points(size, num, pos) -> np.ndarray:
        ...
        xs = np.linspace(-size, size, num=num)
        ys = np.linspace(-size, size, num=num)
        px, py = np.meshgrid(xs, ys)
        pz = np.full_like(px, pos[2])
        points = np.stack((px+pos[0], py+pos[1], pz), axis=-1)
        return points

    obj_colors = [
        (0, 1, 0, 0.7),
        (0, 0, 1, 0.7),
        (1, 1, 0, 0.7),
        (1, 0, 1, 0.7),
    ]

    planes = [
        make_plane_points(1000, 10, pos=(0, 1000, 2000)),
        make_plane_points(2000, 20, pos=(0, -1000, 5000)),
        make_plane_points(1000, 10, pos=(1000, 0, 2500)),
        make_plane_points(1000, 10, pos=(-1000, 0, 1500)),
    ]
    plane_plots = []

    def project_ph(cam_int: CameraIntrinsics, points: np.ndarray) -> np.ndarray:
        uvw = np.zeros((points.shape[0], 3))
        uvw[:, 0] = -(cam_int.fx * points[:, 0]) / points[:, 2]
        uvw[:, 1] = -(cam_int.fy * points[:, 1]) / points[:, 2]
        uvw[:, 2] = -0 # points[:, 2]
        return uvw[:, :3]

    for i, p in enumerate(planes[:]):
        p = p.reshape(-1, 3)
        s = gl.GLScatterPlotItem(pos=p.reshape(-1, 3), size=50, color=obj_colors[i], pxMode=False)
        view.addItem(s)
        plane_plots.append(s)

        obj_2dpos = project_ph(cam_int, p.reshape(-1, 3))
        s2 = gl.GLScatterPlotItem(pos=obj_2dpos, size=10, color=obj_colors[i], pxMode=False)
        view.addItem(s2)

        ray_pos = np.zeros((p.shape[0] * 2, 3))
        ray_pos[0::2] = p
        ray_pos[1::2] = obj_2dpos
        # # Create line plot
        line_plot = gl.GLLinePlotItem(
             pos=ray_pos[::, :], color=obj_colors[i], width=1, antialias=True, mode='lines'
        )
        view.addItem(line_plot)






    # # Sample rays to avoid cluttering (e.g., every 10th pixel)
    # step = 100
    # sampled_origins = origins[::step, ::step].reshape(-1, 3)
    # sampled_dirs = dirs[::step, ::step].reshape(-1, 3)
    #
    # # Create line segments from origins to endpoints
    # ray_length = 2.0
    # endpoints = sampled_origins + sampled_dirs * ray_length
    #
    # # Create pos array with alternating origin-endpoint pairs
    # pos = np.zeros((sampled_origins.shape[0] * 2, 3))
    # pos[0::2] = sampled_origins
    # pos[1::2] = endpoints
    # print(pos[::10, :].shape)
    # # Create line plot
    # line_plot = gl.GLLinePlotItem(
    #     pos=pos[::, :], color=(1, 1, 1, 0.7), width=1, antialias=True
    # )
    #
    # pw = 1.0
    # ph = cam_int.height / cam_int.width
    #
    # vertices = np.array(
    #     [
    #         [-pw, -ph, 1],
    #         [pw, -ph, 1],
    #         [pw, ph, 1],
    #         [-pw, ph, 1],
    #     ]
    # )
    # faces = np.array([[0, 1, 2], [0, 2, 3]])
    # plane = gl.GLMeshItem(
    #     vertexes=vertices,
    #     faces=faces,
    #     color=(0.5, 0.8, 1.0, 0.3),
    #     smooth=False,
    #     drawEdges=True,
    #     edgeColor=(0, 0.5, 1, 0.1),
    #     computeNormals=True,
    #     shader="balloon",
    # )
    # view.addItem(plane)
    # view.addItem(line_plot)

    app.exec()


if __name__ == "__main__":
    main()
