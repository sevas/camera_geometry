use numpy::ndarray::{
    Array, Array2, ArrayView1, ArrayView2,
};
use numpy::{
    IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2
};
use pyo3::{
    pymodule,
    types::{PyModule},
    PyResult, Python,
};

#[pymodule]
fn cg_rustpy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    fn project_points(
        points: ArrayView2<'_, f64>,
        k: ArrayView2<'_, f64>,
        dist: ArrayView1<'_, f64>,
    ) -> Array2<f64> {
        let n = points.shape()[0];
        let uv_shape = (n, 2);
        let mut uv: Array2<f64> = Array::zeros(uv_shape);


        let fx = k[(0, 0)];
        let fy = k[(1, 1)];
        let cx = k[(0, 2)];
        let cy = k[(1, 2)];
        let k1 = dist[0];
        let k2 = dist[1];
        let p1 = dist[2];
        let p2 = dist[3];
        let k3 = dist[4];

        for i in 0..n {
            let p_x = points[(i, 0)];
            let p_y = points[(i, 1)];
            let p_z = points[(i, 2)];

            let x = p_x / p_z;
            let y = p_y / p_z;

            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let r6 = r2 * r4;
            let a1 = 2.0 * x * y;
            let a2 = r2 + 2.0 * x * x;
            let a3 = r2 + 2.0 * y * y;

            let cdist = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
            let xd = x * cdist + p1 * a1 + p2 * a2;
            let yd = y * cdist + p1 * a3 + p2 * a1;

            uv[(i, 0)] = xd * fx + cx;
            uv[(i, 1)] = yd * fy + cy;
        }

        uv
    }

    // wrapper of `project_points`
    #[pyfn(m)]
    #[pyo3(name = "project_points_rs")]
    fn project_points_py<'py>(
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
        k: PyReadonlyArray2<'_, f64>,
        dist: PyReadonlyArray1<'_, f64>,
    ) -> &'py PyArray2<f64> {
        let points = points.as_array();
        let dist = dist.as_array();
        let k = k.as_array();

        let uv = project_points(points, k, dist);
        uv.into_pyarray(py)
    }

    Ok(())
}
