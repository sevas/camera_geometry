#include <vector>




struct vertex_data
{
    void add_vertex(float x, float y, float z);
    void concat(const vertex_data& other);
    std::vector<float> pack_vertices_nx3() const;

    std::vector<float> xs;
    std::vector<float> ys;
    std::vector<float> zs;
};

vertex_data make_plane(int w, int h, float cx, float cy, float cz, int step = 1);
vertex_data make_sphere(const int n);