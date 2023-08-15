#include <cstddef>
#include <vector>

namespace cg::geometry3d {

struct vertex_data {
    void add_vertex(float x, float y, float z);

    void concat(const vertex_data& other);

    [[nodiscard]] std::vector<float> pack_vertices_nx3() const;

    [[nodiscard]] size_t size() const { return xs.size(); };

    std::vector<float> xs;
    std::vector<float> ys;
    std::vector<float> zs;
};

vertex_data make_plane(int w, int h, float cx, float cy, float cz, int step = 1);

vertex_data make_sphere(int n);

void rotate(vertex_data& vertices, std::array<float, 9> rmat);

void rot_y(vertex_data& vertices, float theta);

void rot_z(vertex_data& vertices, float theta);

void translate(vertex_data& vertices, const std::array<float, 3>& tvec);

void scale(vertex_data& vertices, float scale);

template <typename T> T m3x3_at(const std::array<T, 9>& m3x3, int i, int j)
{
    return m3x3[j * 3 + i];
}
}
