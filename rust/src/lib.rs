use std::ops::Add;

use numpy::ndarray::{
    Array, Array1, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD, ArrayViewMutD, Zip,
};
use numpy::{
    datetime::{units, Timedelta},
    Complex64, IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn, PyReadwriteArray1, PyReadwriteArrayDyn,
};
use pyo3::{
    exceptions::PyIndexError,
    pymodule,
    types::{PyDict, PyModule},
    FromPyObject, PyAny, PyObject, PyResult, Python,
};

#[pymodule]
fn cg_rustpy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // example using generic PyObject
    fn head(x: ArrayViewD<'_, PyObject>) -> PyResult<PyObject> {
        x.get(0)
            .cloned()
            .ok_or_else(|| PyIndexError::new_err("array index out of range"))
    }

    // example using immutable borrows producing a new array
    fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // example using a mutable borrow to modify an array in-place
    fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
        x *= a;
    }

    // example using complex numbers
    fn conj(x: ArrayViewD<'_, Complex64>) -> ArrayD<Complex64> {
        x.map(|c| c.conj())
    }

    // example using generics
    fn generic_add<T: Copy + Add<Output = T>>(x: ArrayView1<T>, y: ArrayView1<T>) -> Array1<T> {
        &x + &y
    }

    fn project_points(
        points: ArrayView2<'_, f64>,
        k: ArrayView2<'_, f64>,
        dist: ArrayView1<'_, f64>,
    ) -> Array2<f64> {
        let n = points.shape()[0];
        let uv_shape = (2, n);
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

        // let index = IxDyn(&[1, 0]);
        // let element = array.get(index).unwrap();
        // println!("The element at position {:?} is {}", index, element);

        for i in 0..n {
            // let index = IxDyn(&[1, 0]);
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

    // fn get_shape_0(points: ArrayViewD<'_, f64>) -> usize {
    //     let n = points.shape();
    //     n[0]
    // }

    // wrapper of `head`
    #[pyfn(m)]
    #[pyo3(name = "head")]
    fn head_py(_py: Python<'_>, x: PyReadonlyArrayDyn<'_, PyObject>) -> PyResult<PyObject> {
        head(x.as_array())
    }

    // wrapper of `axpy`
    #[pyfn(m)]
    #[pyo3(name = "axpy")]
    fn axpy_py<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<'_, f64>,
        y: PyReadonlyArrayDyn<'_, f64>,
    ) -> &'py PyArrayDyn<f64> {
        let x = x.as_array();
        let y = y.as_array();
        let z = axpy(a, x, y);
        z.into_pyarray(py)
    }

    // wrapper of `project_points`
    #[pyfn(m)]
    #[pyo3(name = "project_points")]
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
        // points.into_pyarray(py)
    }

    // wrapper of `mult`
    #[pyfn(m)]
    #[pyo3(name = "mult")]
    fn mult_py(a: f64, mut x: PyReadwriteArrayDyn<f64>) {
        let x = x.as_array_mut();
        mult(a, x);
    }

    // wrapper of `conj`
    #[pyfn(m)]
    #[pyo3(name = "conj")]
    fn conj_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<'_, Complex64>,
    ) -> &'py PyArrayDyn<Complex64> {
        conj(x.as_array()).into_pyarray(py)
    }

    // example of how to extract an array from a dictionary
    #[pyfn(m)]
    fn extract(d: &PyDict) -> f64 {
        let x = d
            .get_item("x")
            .unwrap()
            .downcast::<PyArray1<f64>>()
            .unwrap();

        x.readonly().as_array().sum()
    }

    // example using timedelta64 array
    #[pyfn(m)]
    fn add_minutes_to_seconds(
        mut x: PyReadwriteArray1<Timedelta<units::Seconds>>,
        y: PyReadonlyArray1<Timedelta<units::Minutes>>,
    ) {
        #[allow(deprecated)]
        Zip::from(x.as_array_mut())
            .and(y.as_array())
            .apply(|x, y| *x = (i64::from(*x) + 60 * i64::from(*y)).into());
    }

    // This crate follows a strongly-typed approach to wrapping NumPy arrays
    // while Python API are often expected to work with multiple element types.
    //
    // That kind of limited polymorphis can be recovered by accepting an enumerated type
    // covering the supported element types and dispatching into a generic implementation.
    #[derive(FromPyObject)]
    enum SupportedArray<'py> {
        F64(&'py PyArray1<f64>),
        I64(&'py PyArray1<i64>),
    }

    #[pyfn(m)]
    fn polymorphic_add<'py>(
        x: SupportedArray<'py>,
        y: SupportedArray<'py>,
    ) -> PyResult<&'py PyAny> {
        match (x, y) {
            (SupportedArray::F64(x), SupportedArray::F64(y)) => Ok(generic_add(
                x.readonly().as_array(),
                y.readonly().as_array(),
            )
            .into_pyarray(x.py())
            .into()),
            (SupportedArray::I64(x), SupportedArray::I64(y)) => Ok(generic_add(
                x.readonly().as_array(),
                y.readonly().as_array(),
            )
            .into_pyarray(x.py())
            .into()),
            (SupportedArray::F64(x), SupportedArray::I64(y))
            | (SupportedArray::I64(y), SupportedArray::F64(x)) => {
                let y = y.cast::<f64>(false)?;

                Ok(
                    generic_add(x.readonly().as_array(), y.readonly().as_array())
                        .into_pyarray(x.py())
                        .into(),
                )
            }
        }
    }

    Ok(())
}
