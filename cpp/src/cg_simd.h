
#include <vector>
#include "cg_types.h"

namespace cg {
	
	void project_points_simd(const std::vector<float>& xs, const std::vector<float>& ys,
		const std::vector<float>& zs, std::vector<float>& us, std::vector<float>& vs,
		const camera_intrinsics& intrinsics);

}
