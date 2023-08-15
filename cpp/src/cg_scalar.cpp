#include "cg_scalar.h"

namespace cg {
void project_points_cpu(const std::vector<float>& xs, const std::vector<float>& ys,
    const std::vector<float>& zs, std::vector<float>& us, std::vector<float>& vs,
    const camera_intrinsics& intrinsics)
{
    const int n = static_cast<int>(xs.size());
    for (int i = 0; i < n; ++i) {
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
}
