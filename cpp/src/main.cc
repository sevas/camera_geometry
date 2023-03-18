#pragma warning(push, 3)
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#pragma warning(pop)

#include "cuda/gpu.hpp"
#include "scoped_timer.hpp"


using std::begin;
using std::end;


struct point
{
    float x, y, z;
};

struct camera_intrinsics
{
    camera_intrinsics(const int w,
                      const int h,
                      const float fx,
                      const float fy,
                      const float cx,
                      const float cy,
                      const float k1 = 0.f,
                      const float k2 = 0.f,
                      const float k3 = 0.f,
                      const float p1 = 0.f,
                      const float p2 = 0.f) :
        h(h),
        w(w),
        fx(fx),
        fy(fy),
        cx(cx),
        cy(cy),
        k1(k1),
        k2(k2),
        k3(k3),
        p1(p1),
        p2(p2)
    {}

    int h, w;
    float fx, fy;
    float cx, cy;
    float k1, k2, k3;
    float p1, p2;
};

void project_points_cpu(const std::vector<float>& xs,
                        const std::vector<float>& ys,
                        const std::vector<float>& zs,
                        std::vector<float>& us,
                        std::vector<float>& vs,
                        const camera_intrinsics& intrinsics)
{
    const int n = static_cast<int>(xs.size());
    for (int i = 0; i < n; ++i)
    {
        const float x = xs[i] / zs[i];
        const float y = ys[i] / zs[i];

        const float fx = intrinsics.fx;
        const float fy = intrinsics.fy;
        const float cx = intrinsics.cx;
        const float cy = intrinsics.cy;
        const float k1 = intrinsics.k1;
        const float k2 = intrinsics.k2;
        const float k3 = intrinsics.k3;
        const float p1 = intrinsics.p1;
        const float p2 = intrinsics.p2;

        const float r2 = x * x + y * y;
        const float r4 = r2 * r2;
        const float r6 = r4 * r2;
        const float a1 = 2 * x * y;
        const float a2 = r2 + 2 * x * x;
        const float a3 = r2 + 2 * y * y;

        const float frac = (1 + k1 * r2 + k2 * r4 + k3 * r6);
        const float xd = x * frac + p1 * a1 + p2 * a2;
        const float yd = y * frac + p1 * a3 + p2 * a1;

        us[i] = xd * fx + cx;
        vs[i] = yd * fy + cy;
    }
}


template<typename T>
struct cuda_array
{
    cuda_array(const int n)
    {
        cudaMalloc(reinterpret_cast<void**>(&data), sizeof(T) * n);
    }
    ~cuda_array()
    {
        cudaFree(data);
    }

    void copy_into(std::vector<T>& out)
    {
        cudaMemcpy(out.data(), data, sizeof(T) * out.size(), cudaMemcpyDeviceToHost);
    }

    void copy_from(const std::vector<T>& in)
    {
        cudaMemcpy(data, in.data(), sizeof(T) * in.size(), cudaMemcpyHostToDevice);
    }

    T* data;
};


struct vertex_data
{
    void add_vertex(float x, float y, float z)
    {
        xs.push_back(x);
        ys.push_back(y);
        zs.push_back(z);
    }

    void concat(const vertex_data& other)
    {
        for (auto i = 0u; i < other.xs.size(); ++i)
        {
            xs.push_back(other.xs[i]);
            ys.push_back(other.ys[i]);
            zs.push_back(other.zs[i]);
        }
    }

    std::vector<float> pack_vertices_nx3() const
    {
        std::vector<float> out;
        const auto n = xs.size();

        out.resize(n * 3);
        for (auto i = 0u; i < n; ++i)
        {
            out[i * 3] = xs[i];
            out[i * 3 + 1] = ys[i];
            out[i * 3 + 2] = zs[i];
        }

        return out;
    }

    std::vector<float> xs;
    std::vector<float> ys;
    std::vector<float> zs;
};


vertex_data make_plane(int w, int h, float cx, float cy, float cz, int step = 1)
{
    vertex_data out;
    for (size_t i = 0; i < w; i += step)
    {
        for (size_t j = 0; j < h; j += step)
        {
            float x = static_cast<float>(cx) - (static_cast<float>(w) / 2) + i * step;
            float y = static_cast<float>(cy) - (static_cast<float>(h) / 2) + j * step;

            out.add_vertex(x, y, static_cast<float>(cz));
        }
    }
    return out;
}

vertex_data make_sphere(const int n)
{
    vertex_data out;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0, 1);

    for (auto i = 0; i < n; ++i)
    {
        float x = d(gen);
        float y = d(gen);
        float z = d(gen);

        const float norm = std::sqrt(x * x + y * y + z * z);

        x /= norm;
        y /= norm;
        z /= norm;

        out.add_vertex(x, y, z + 2);
    }
    return out;
}


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

    std::cout << "Hello, world!" << std::endl;
    std::cout << "CUDA: On" << std::endl;
    printCudaVersion();

    camera_intrinsics intrinsics{240, 180, 50, 50, 240.f / 2, 180.f / 2, 0.01f, 0.2f, 0.0f, 0.0f, 0.0f};

    vertex_data far_plane = make_plane(64, 64, 0, 0, 20, 1);
    vertex_data points = make_plane(16, 16, 4.5f, 4.5f, 18, 1);


    vertex_data far_plane2 = make_plane(64000, 6400, 0, 0, 30, 1);


    points.concat(far_plane);
    points.concat(far_plane2);


    int N = static_cast<int>(points.xs.size());
    std::cout << "Point count: " << N << std::endl;


    const auto pointcloud = points.pack_vertices_nx3();


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


    std::vector<float> u_cpu(N);
    std::vector<float> v_cpu(N);

    {
        default_scoped_timer t("project_points_cpu");
        project_points_cpu(points.xs, points.ys, points.zs, u_cpu, v_cpu, intrinsics);
    }

    auto img_cpu = render_z_buffer(intrinsics.h, intrinsics.w, u_cpu, v_cpu, points.zs);

    

    return 0;
}
