#pragma warning(push, 3)

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#pragma warning(pop)

#include "cg_scalar.h"
#include "cg_types.h"
#include "geometry3d.h"
#include "happly.h"
#include "imageio.h"
#include "scoped_timer.hpp"

#ifdef USE_CUDA
#include "cuda/gpu.hpp"
#include "cg_cuda.h"
#endif

#ifdef USE_HALIDE

#include "Halide.h"

#endif


using std::begin;
using std::end;

std::vector<uint8_t> render_z_buffer(const int h, const int w, std::vector<float> &us, std::vector<float> &vs,
                                     std::vector<float> &zs)
{
    std::vector<uint8_t> img(h * w);
    std::fill(begin(img), end(img), 255);
    const auto N = us.size();

    for (int i = 0u; i < N; ++i)
    {
        int u = static_cast<int>(us[i]);
        int v = static_cast<int>(vs[i]);
        if (u >= 0 && u < w && v >= 0 && v < h)
        {
            const auto value = static_cast<uint8_t>(zs[i]);
            if (value < img[v * w + u])
            {
                img[v * w + u] = value;
            }
        }
    }

    return img;
}

#ifdef USE_HALIDE
Halide::Func blur_3x3(Halide::Func input)
{
    Halide::Func blur_x, blur_y;
    Halide::Var x, y, xi, yi;

    Halide::Expr val = input(x, y);
    blur_x(x, y) = (val + input(x + 1, y) + input(x - 1, y)) / 3;
    blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y - 1)) / 3;

    return blur_y;
}


Halide::Target find_gpu_target()
{
    using namespace Halide;
    Target target = get_host_target();

    std::vector<Target::Feature> features_to_try;
    if(target.os == Target::Windows)
    {
        if(sizeof(void*) == 8)
        {
            features_to_try.push_back(Target::D3D12Compute);

        }
        features_to_try.push_back(Target::OpenCL);

    }
    else if (target.os == Target::OSX)
    {
        features_to_try.push_back(Target::Metal);
    }
    else
    {
        features_to_try.push_back(Target::OpenCL);
    }
    for(auto feature : features_to_try)
    {
        Target new_target = target.with_feature(feature);
        if(host_supports_target_device(new_target))
        {
            return new_target;
        }
    }
    return target;
}


void project_points_halide(const std::vector<float>& xs,
                           const std::vector<float>& ys,
                           const std::vector<float>& zs,
                           std::vector<float>& us,
                           std::vector<float>& vs,
                           const camera_intrinsics& intrinsics)
{
    using namespace Halide;
    const int n = static_cast<int>(xs.size());
    Buffer<float> xs_buf(const_cast<float*>(xs.data()), n);
    Buffer<float> ys_buf(const_cast<float*>(ys.data()), n);
    Buffer<float> zs_buf(const_cast<float*>(zs.data()), n);

    const float fx = intrinsics.fx;
    const float fy = intrinsics.fy;
    const float cx = intrinsics.cx;
    const float cy = intrinsics.cy;
    const float k1 = intrinsics.k1;
    const float k2 = intrinsics.k2;
    const float k3 = intrinsics.k3;
    const float p1 = intrinsics.p1;
    const float p2 = intrinsics.p2;

    Func project_points("project_points");
    Var i("i");
    Expr x = xs_buf(i) / zs_buf(i);
    Expr y = ys_buf(i) / zs_buf(i);
    Expr r2 = x * x + y * y;
    Expr r4 = r2 * r2;
    Expr r6 = r4 * r2;
    Expr a1 = 2 * x * y;
    Expr a2 = r2 + 2 * x * x;
    Expr a3 = r2 + 2 * y * y;

    Expr frac = (1 + k1 * r2 + k2 * r4 + k3 * r6);
    Expr xd = x * frac + p1 * a1 + p2 * a2;
    Expr yd = y * frac + p1 * a3 + p2 * a1;

    Expr u = fx * xd + cx;
    Expr v = fy * yd + cy;
    project_points(i) = Tuple(u, v);

    Target target = find_gpu_target();

    std::cout << "using target : " << target.to_string() << std::endl;
    {
        scoped_timer<unit::us> t("compiling halide");
        project_points.compile_jit(target);
    }


    Realization r = project_points.realize({n});

    Buffer<float> u_ = r[0];
    Buffer<float> v_ = r[1];

    std::copy(u_.begin(), u_.end(), us.begin());
    std::copy(v_.begin(), v_.end(), vs.begin());
}
#endif

template <typename T> float avg(const std::vector<T> &v)
{
    const auto sum = std::accumulate(v.cbegin(), v.cend(), 0.f);
    return sum / v.size();
}

int main()
{
    using default_scoped_timer = scoped_timer_us;

#ifdef USE_CUDA
    std::cout << "CUDA: On" << std::endl;
    printCudaVersion();
#endif

    camera_intrinsics intrinsics{
        // clang-format off
        240, 180, 70, 70, 240.f / 2, 180.f / 2,
        0.02f, -0.05f, 0.09f, 0.001f, 0.002f
        // clang-format on
    };

    happly::PLYData plyIn("bun_zipper.ply");
    vertex_data bunny;
    bunny.xs = plyIn.getElement("vertex").getProperty<float>("x");
    bunny.ys = plyIn.getElement("vertex").getProperty<float>("y");
    bunny.zs = plyIn.getElement("vertex").getProperty<float>("z");

    rot_z(bunny, M_PI);
    rot_y(bunny, M_PI);
    translate(bunny, {0.03, 0.07, 0.15});
    scale(bunny, 2.f);


    vertex_data far_plane = make_plane(640, 640, 0, 0, 20, 1);
    //    vertex_data points = make_plane(16, 16, 4.5f, 4.5f, 18, 1);
    //
    //
    //    vertex_data far_plane2 = make_plane(1000, 10000, 0, 0, 30, 1);

    //    points.concat(far_plane);
    //    points.concat(far_plane2);

    vertex_data points;
    points.concat(bunny);
    //points.concat(far_plane);

    int N = static_cast<int>(points.xs.size());
    std::cout << "Point count: " << N << std::endl;


    const auto pointcloud = points.pack_vertices_nx3();


#ifdef USE_CUDA
    cuda_array<float> cu_x(N);
    cuda_array<float> cu_y(N);
    cuda_array<float> cu_z(N);
    cuda_array<float> cu_u(N);
    cuda_array<float> cu_v(N);

    {
        default_scoped_timer t("project_points_gpu::host->gpu");
        cu_x.copy_from(points.xs);
        cu_y.copy_from(points.ys);
        cu_z.copy_from(points.zs);
    }
    cudaFuncAttributes attrs{};
    auto cudaStatus = cudaFuncGetAttributes(&attrs, project_points);
    if (cudaStatus != cudaSuccess)
    {
        printf("cudaFuncGetAttributes call failed: %s\n", cudaGetErrorString(cudaStatus));
        return 0;
    }

    auto const maxThreadCount = attrs.maxThreadsPerBlock;
    auto const blocksize = (N + maxThreadCount - 1) / maxThreadCount;

    std::cout << "Running kernel" << std::endl;
    {
        default_scoped_timer t("project_points_gpu::execute");
        void* args[] = {&cu_x.data, &cu_y.data, &cu_z.data, &cu_u.data, &cu_v.data, &N};
        cudaLaunchKernel(project_points, dim3(1, 1, 1), dim3(1, 1, 1), args, 0U, nullptr);
        ////vector_add <<<1, 1 >>> (out, x, y, N);
        cudaDeviceSynchronize();
    }
    

    std::vector<float> u_gpu(N);
    std::vector<float> v_gpu(N);

    {
        default_scoped_timer t("project_points_gpu::gpu->host");
        cu_u.copy_into(u_gpu);
        cu_v.copy_into(v_gpu);
    }
    auto img_gpu = render_z_buffer(intrinsics.h, intrinsics.w, u_gpu, v_gpu, points.zs);
#endif

    // scalar impl
    std::vector<float> u_cpu(N);
    std::vector<float> v_cpu(N);
    std::vector<unsigned int> times;
    for (auto i = 0u; i < 1000; ++i)
    {
        default_scoped_timer t("project_points_cpu");
        project_points_cpu(points.xs, points.ys, points.zs, u_cpu, v_cpu, intrinsics);
        times.emplace_back(t.get());
    }

    std::cout << "avg: " << avg(times) << " us (sample count=" << times.size() << ")" << std::endl;

    auto img_cpu = render_z_buffer(intrinsics.h, intrinsics.w, u_cpu, v_cpu, points.zs);
    auto img_rgb_cpu = grayscale_to_rgb(img_cpu);
    imwrite("img_cpu.png", intrinsics.w, intrinsics.h, 3, img_rgb_cpu);

#ifdef USE_HALIDE
    // halide impl
    std::vector<float> u_hl(N);
    std::vector<float> v_hl(N);

    project_points_halide(points.xs, points.ys, points.zs, u_hl, v_hl, intrinsics);

    {
        default_scoped_timer t("project_points_halide");
        project_points_halide(points.xs, points.ys, points.zs, u_hl, v_hl, intrinsics);
    }

    auto img_hl = render_z_buffer(intrinsics.h, intrinsics.w, u_hl, v_hl, points.zs);
    auto img_rgb_hl = grayscale_to_rgb(img_hl);
    imwrite("img_hl.png", intrinsics.w, intrinsics.h, 3, img_rgb_hl);
    imwrite("img_hl.bin", intrinsics.w, intrinsics.h, 3, img_rgb_hl);


//    for(auto i=0u; i<N; ++i)
//    {
//        if (std::abs(u_cpu[i] - u_hl[i]) > 1e-3f)
//        {
//            std::cout << "u mismatch at " << i << " " << u_cpu[i] << " " << u_hl[i] << std::endl;
//        }
//        if (std::abs(v_cpu[i] - v_hl[i]) > 1e-3f)
//        {
//            std::cout << "v mismatch at " << i << " " << v_cpu[i] << " " << v_hl[i] << std::endl;
//        }
//    }

#endif
    return 0;
}
