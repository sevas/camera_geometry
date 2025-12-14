import numpy as np

calib = {
    "intrinsics": [
        {
            "camera_type": "eucm",
            "intrinsics": {
                "fx": 460.76484651566468,
                "fy": 459.4051018049483,
                "cx": 365.8937161309615,
                "cy": 249.33499869752445,
                "alpha": 0.5903365915227143,
                "beta": 1.127468196965374,
            },
        },
        {
            "camera_type": "eucm",
            "intrinsics": {
                "fx": 459.55216904505178,
                "fy": 458.17181312352059,
                "cx": 379.4066773637502,
                "cy": 255.98301446219285,
                "alpha": 0.6049889282227827,
                "beta": 1.0907289821146678,
            },
        },
        {
            "camera_type": "eucm",
            "intrinsics": {
                "fx": 194.29878,
                "fy": 194.187,
                "cx": 251.8,
                "cy": 191.65,
                "alpha": 0.748,
                "beta": 1.027,
            },
        },
    ],
    "resolution": [[752, 480], [752, 480], [502, 378]],
}


def eucm_unit_vectors(calib: dict, index: int) -> np.ndarray:
    fx, fy, cx, cy, alpha, beta = calib["intrinsics"][index]["intrinsics"].values()
    w, h = calib["resolution"][index]
    u = np.arange(w)
    v = np.arange(h)
    uv = np.dstack(np.meshgrid(u, v))

    x = (uv[:, :, 0] - cx) / fx
    y = (uv[:, :, 1] - cy) / fy

    r_sq = x**2 + y**2
    gamma = 1.0 - alpha
    m_z = (1 - (beta * alpha**2 * r_sq)) / (
        alpha * np.sqrt(1.0 - (alpha * gamma * beta * r_sq)) + gamma
    )
    s = 1.0 / np.sqrt(r_sq + m_z**2)

    vv = np.dstack([s * x, s * y, s * m_z])
    n = np.linalg.norm(vv, axis=-1)
    vv_n = vv / np.dstack([n, n, n])
    print(vv_n - vv)
    return vv_n


if __name__ == "__main__":
    v = eucm_unit_vectors(calib, 2)

    # plot the unit vectors
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.quiver(0, 0, 0, v[::10, ::10, 0], v[::10, ::10, 1], v[::10, ::10, 2], length=0.1)
    # plt.show()

    # plot unit vectors with pyqtgraph
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtCore, QtGui

    app = pg.mkQApp()

    w = gl.GLViewWidget()
    w.setWindowTitle("pyqtgraph example: ScatterPlot")
    g = gl.GLGridItem()
    w.addItem(g)

    # v = v[::10, ::10]
    pos = v.reshape(-1, 3) * 10
    sp1 = gl.GLScatterPlotItem(pos=pos, size=0.1, pxMode=True)
    # sp1.translate(5, 5, 0)
    w.addItem(sp1)
    w.show()
    app.exec()
