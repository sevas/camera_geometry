#include "cg_simd.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cg_simd.cpp" 
#include <hwy/foreach_target.h>  
#include <hwy/highway.h>
HWY_BEFORE_NAMESPACE();

namespace cg {
	//void project_points_simd(const std::vector<float>& xs, const std::vector<float>& ys,
	//	const std::vector<float>& zs, std::vector<float>& us, std::vector<float>& vs,
	//	const camera_intrinsics& intrinsics)
	//{
	//	// SIMD implementation would go here
	//	// For now, we can just call the scalar version as a placeholder
	//	//project_points_cpu(xs, ys, zs, us, vs, intrinsics);
 //       
	//}

    namespace HWY_NAMESPACE {
    namespace hn = hwy::HWY_NAMESPACE;
	using T = float;

    HWY_ATTR void squared(const float* in, float* out, size_t num)
    { 
        hn::ScalableTag<T> d;
		for (size_t i = 0; i < num; i += hn::Lanes(d)) 
		{
            const auto val = hn::Load(d, in + i);
            const auto sq = hn::Mul(val, val);
            hn::Store(sq, d, out + i);
        }
    }

}
}
HWY_AFTER_NAMESPACE();
