//disable warnings with pragma
#pragma warning(push, 3)
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#pragma warning(pop)

#include "gpu.hpp"


using std::begin;
using std::end;


struct point {
	float x, y, z;
};


void project_points_cpu(const std::vector<float>& xs, const std::vector<float>& ys, const std::vector<float>& zs, std::vector<float>& us, std::vector<float>& vs)
{
	const int n = static_cast<int>(xs.size());
	for (int i = 0; i < n; ++i)
	{
		float x = xs[i] / zs[i];
		float y = ys[i] / zs[i];

		float h = 180.f;
		float w = 240.f;

		float fx = 70.0f;
		float fy = 70.0f;
		float cx = w / 2;
		float cy = h / 2;
		float k1 = 0.0f;
		float k2 = 0.02f;
		float k3 = 0.0f;
		float p1 = 0.0f;
		float p2 = 0.0f;

		float r2 = x * x + y * y;
		float r4 = r2 * r2;
		float r6 = r4 * r2;
		float a1 = 2 * x * y;
		float a2 = r2 + 2 * x * x;
		float a3 = r2 + 2 * y * y;

		float frac = (1 + k1 * r2 + k2 * r4 + k3 * r6);
		float xd = x * frac + p1 * a1 + p2 * a2;
		float yd = y * frac + p1 * a3 + p2 * a1;

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


struct random_sphere_points
{
	random_sphere_points(const int n)
	{
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

			xs.emplace_back(x);
			ys.emplace_back(y);
			zs.emplace_back(z + 2);
		}

	}

	std::vector<float> xs;
	std::vector<float> ys;
	std::vector<float> zs;
};

struct plane
{
	plane(int w, int h, int cx, int cy, int cz, int step = 1) {

		for (size_t i = 0; i < w; i += step)
		{
			for (size_t j = 0; j < h; j += step)
			{
				float x = static_cast<float>(cx) - (static_cast<float>(w) / 2) + i * step;
				float y = static_cast<float>(cy) - (static_cast<float>(h) / 2) + j * step;

				xs.push_back(x);
				ys.push_back(y);
				zs.push_back(static_cast<float>(cz));
			}
		}

	};

	std::vector<float> xs;
	std::vector<float> ys;
	std::vector<float> zs;
};



int main()
{
	std::cout << "Hello, world!" << std::endl;
	std::cout << "CUDA: On" << std::endl;
	printCudaVersion();


	//float* out = nullptr, * x = nullptr, * y = nullptr;
	//int N = 100000000;
	//
	//std::cout << "allocating on gpu" << std::endl;
	//cudaMalloc((void**)&out, sizeof(float) * N);
	//cudaMalloc((void**)&x, sizeof(float) * N);
	//cudaMalloc((void**)&y, sizeof(float) * N);

	//std::cout << "making sample data" << std::endl;
	//std::vector<float> xx(N);
	//std::vector<float> yy(N);

	//std::fill(begin(xx), end(xx), 12.f);
	//std::fill(begin(yy), end(yy), 45.f);

	//std::cout << "Copying to GPU" << std::endl;
	//cudaMemcpy(x, xx.data(), N, cudaMemcpyHostToDevice);
	//cudaMemcpy(y, yy.data(), N, cudaMemcpyHostToDevice);

	//cudaFuncAttributes attrs{ };
	//auto cudaStatus = cudaFuncGetAttributes(&attrs, vector_add);
	//if (cudaStatus != cudaSuccess)
	//{
	//    printf("cudaFuncGetAttributes call failed: %s\n", cudaGetErrorString(cudaStatus));
	//    return 0;
	//}


	//auto const maxThreadCount = attrs.maxThreadsPerBlock;
	//auto const blocksize = (N + maxThreadCount - 1) / maxThreadCount;

	//std::cout << "maxThreadCount: " << maxThreadCount << std::endl;
	//std::cout << "blocksize: " << blocksize << std::endl;

	//std::cout << "RUnning kernel" << std::endl;
	//void* args[] = {&out, &x, &y, &N};
	//cudaLaunchKernel(vector_add, dim3(1, 1, 1), dim3(blocksize, 1, 1), args, 0U, nullptr);

	////vector_add <<<1, 1 >>> (out, x, y, N);

	//cudaDeviceSynchronize();

	//std::cout << "Done" << std::endl;

	//cudaFree(x);
	//cudaFree(y);
	//cudaFree(out);


	//int N = 100000;
	//random_sphere_points points(N);
	plane points(64, 64, 0, 0, 20, 1);

	int N = static_cast<int>(points.xs.size());



	cuda_array<float> x(N);
	cuda_array<float> y(N);
	cuda_array<float> z(N);
	cuda_array<float> u(N);
	cuda_array<float> v(N);


	cudaMemcpy(x.data, points.xs.data(), N, cudaMemcpyHostToDevice);
	cudaMemcpy(y.data, points.ys.data(), N, cudaMemcpyHostToDevice);
	cudaMemcpy(z.data, points.zs.data(), N, cudaMemcpyHostToDevice);

	cudaFuncAttributes attrs{ };
	auto cudaStatus = cudaFuncGetAttributes(&attrs, vector_add);
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaFuncGetAttributes call failed: %s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}


	auto const maxThreadCount = attrs.maxThreadsPerBlock;
	auto const blocksize = (N + maxThreadCount - 1) / maxThreadCount;

	std::cout << "RUnning kernel" << std::endl;
	void* args[] = { &x.data, &y.data, &z.data, &u.data, &v.data };
	cudaLaunchKernel(project_points, dim3(maxThreadCount, 1, 1), dim3(blocksize, 1, 1), args, 0U, nullptr);

	////vector_add <<<1, 1 >>> (out, x, y, N);

	cudaDeviceSynchronize();

	std::vector<float> uu(N);
	std::vector<float> vv(N);

	cudaMemcpy(uu.data(), u.data, N, cudaMemcpyDeviceToHost);
	cudaMemcpy(vv.data(), v.data, N, cudaMemcpyDeviceToHost);

	std::vector<uint8_t> img(static_cast<size_t>(180 * 240));
	std::fill(begin(img), end(img), 0);
	const int w = 240, h = 180;

	for (int i = 0u; i < N; ++i)
	{
		int u_ = int(uu[i]);
		int v_ = int(vv[i]);
		if (u_ >= 0 && u_ < w && v_ >= 0 && v_ < h)
		{
			img[v_ * w + u_] = static_cast<uint8_t>(points.zs[i] * 255);
		}

	}


	std::vector<float> u_cpu(N);
	std::vector<float> v_cpu(N);

	project_points_cpu(points.xs, points.ys, points.zs, u_cpu, v_cpu);

	std::vector<uint8_t> img2(180 * 240);
	std::fill(begin(img2), end(img2), 0);

	for (int i = 0u; i < N; ++i)
	{
		int u_ = int(u_cpu[i]);
		int v_ = int(v_cpu[i]);
		if (u_ >= 0 && u_ < w && v_ >= 0 && v_ < h)
		{
			img2[v_ * w + u_] = static_cast<uint8_t>(points.zs[i] * 255);
		}

	}


	return 0;
}
