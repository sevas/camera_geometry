__global__ void vector_add(float *out, float *a, float *b, int n) {

    auto const gtid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gtid >= n ) {
        return;
    }

    out[gtid] = a[gtid] * a[gtid] + (b[gtid]);

}
