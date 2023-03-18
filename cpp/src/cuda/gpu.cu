#include <iostream>
#include "gpu.hpp"

#include <cuda_runtime_api.h>

void printCudaVersion()
{
    //std::cout << "CUDA Compiled version: " << __CUDA << std::endl;

    int runtime_ver;
    cudaRuntimeGetVersion(&runtime_ver);
    std::cout << "CUDA Runtime version: " << runtime_ver << std::endl;

    int driver_ver;
    cudaDriverGetVersion(&driver_ver);
    std::cout << "CUDA Driver version: " << driver_ver << std::endl;
}
