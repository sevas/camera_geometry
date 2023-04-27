#pragma warning(push, 3)
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#pragma warning(pop)

#ifdef USE_CUDA
#include "cuda/gpu.hpp"
#endif

#include "scoped_timer.hpp"
#include "geometry3d.h"
#include "cg_types.h"
#include "cg_scalar.h"

using std::begin;
using std::end;



std::vector<uint8_t> render_z_buffer(const int h, const int w, std::vector<float>& us, std::vector<float>& vs, std::vector<float>& zs)
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


int main()
{
    using default_scoped_timer = scoped_timer_us;

#ifdef USE_CUDA
    std::cout << "CUDA: On" << std::endl;
    printCudaVersion();
#endif

    camera_intrinsics intrinsics{240, 180, 50, 50, 240.f / 2, 180.f / 2, 0.01f, 0.2f, 0.0f, 0.0f, 0.0f};

    vertex_data far_plane = make_plane(64, 64, 0, 0, 20, 1);
    vertex_data points = make_plane(16, 16, 4.5f, 4.5f, 18, 1);


    vertex_data far_plane2 = make_plane(640, 640, 0, 0, 30, 1);


    points.concat(far_plane);
    points.concat(far_plane2);


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

    std::vector<float> u_cpu(N);
    std::vector<float> v_cpu(N);

    {
        default_scoped_timer t("project_points_cpu");
        project_points_cpu(points.xs, points.ys, points.zs, u_cpu, v_cpu, intrinsics);
    }

    auto img_cpu = render_z_buffer(intrinsics.h, intrinsics.w, u_cpu, v_cpu, points.zs);

    

    return 0;
}
