#pragma once
#ifdef USE_CUDA
#include <cuda_runtime.h>

namespace cg::cuda {

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
}

#endif // USE_CUDA
