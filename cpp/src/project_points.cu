__global__
void project_points(const float* X, const float* Y, const float* Z, float* u, float* v)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float x = X[tid] / Z[tid];
    float y = Y[tid] / Z[tid];

    float h = 180;
    float w = 240;

    float fx = 70.0f;
    float fy = 70.0f;
    float cx = w / 2;
    float cy = h / 2;
    float k1 = 0.0;
    float k2 = 0.2;
    float k3 = 0.0;
    float p1 = 0.0;
    float p2 = 0.0;

    float r2 = x * x + y * y;
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    float a1 = 2 * x * y;
    float a2 = r2 + 2 * x * x;
    float a3 = r2 + 2 * y * y;

    float frac = (1 + k1 * r2 + k2 * r4 + k3 * r6);
    float xd = x * frac + p1 * a1 + p2 * a2;
    float yd = y * frac + p1 * a3 + p2 * a1;

    u[tid] = xd * fx + cx;
    v[tid] = yd * fy + cy;

}
