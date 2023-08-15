#include "doctest.h"

#include "geometry3d.h"
using namespace cg::geometry3d;

TEST_CASE("vertex_data transformations")
{
    vertex_data vd = make_plane(10, 1, 0, 0, 1, 1);

    REQUIRE(vd.size() == 10);
    REQUIRE(vd.xs.size() == 10);
    REQUIRE(vd.ys.size() == 10);
    REQUIRE(vd.zs.size() == 10);

    const std::vector<float> expected_pcl = {
        // clang-format off
        -5, -0.5, 1,
        -4, -0.5, 1,
        -3, -0.5, 1,
        -2, -0.5, 1,
        -1, -0.5, 1,
        0, -0.5, 1,
        1, -0.5, 1,
        2, -0.5, 1,
        3, -0.5, 1,
        4, -0.5, 1,
        // clang-format on
    };
    const auto pcl = vd.pack_vertices_nx3();
    REQUIRE(pcl == expected_pcl);
}
