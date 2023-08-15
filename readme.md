## camera_geometry

![GitHub Workflow Status (rust)](https://img.shields.io/github/actions/workflow/status/sevas/camera_geometry/rust.yml)
![GitHub Workflow Status (c++)](https://img.shields.io/github/actions/workflow/status/sevas/camera_geometry/cmake-multi-platform.yml)


Various implementations and speed comparison of 3D-to-2D camera projection.

Including: 

- pure numpy
- opencv
- numba
- julia
- rust
- c++ scalar
- cuda
- c++ with [halide](https://github.com/halide/Halide)

Coming next:
- opencl
- glsl shader
- fpga

## Purpose

This is purely educational and not made to be a production-ready library.


## Timings

### Macbook Air M2
![benchmark_20230815_130412.svg.png](media%2Fbenchmark_20230815_130412.svg.png)

```text
------------------------------------------------------------------------------------------------------------- benchmark: 5 tests ------------------------------------------------------------------------------------------------------------
Name (time in us)                                                 Min                   Max                  Mean              StdDev                Median                 IQR            Outliers         OPS            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_benchmarkproject_points[project_points_nb_parfor]        75.2080 (1.0)        127.6660 (1.05)       100.3250 (1.0)       21.5014 (11.86)       97.2920 (1.0)       35.8958 (12.14)         2;0  9,967.6051 (1.0)           5           1
test_benchmarkproject_points[project_points_rs]              108.9590 (1.45)       185.1670 (1.52)       112.5200 (1.12)       5.3272 (2.94)       110.7500 (1.14)       2.9580 (1.0)       428;447  8,887.3110 (0.89)       5703           1
test_benchmarkproject_points[project_points_nb]              117.8750 (1.57)       121.5001 (1.0)        119.2164 (1.19)       1.8130 (1.0)        117.9160 (1.21)       3.1255 (1.06)          1;0  8,388.1075 (0.84)          5           1
test_benchmarkproject_points[project_points_np]              354.3750 (4.71)       616.2500 (5.07)       398.7205 (3.97)      44.0745 (24.31)      380.5410 (3.91)      62.1557 (21.01)        59;8  2,508.0227 (0.25)        409           1
test_benchmarkproject_points[project_points_cv]            5,152.7080 (68.51)    5,541.5840 (45.61)    5,283.0488 (52.66)    138.7260 (76.52)    5,274.6880 (54.22)    109.1250 (36.89)         1;1    189.2846 (0.02)          6           1
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



```

## License
The code in this repository is published under the MIT license.
