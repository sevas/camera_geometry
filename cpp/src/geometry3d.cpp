
#include "geometry3d.h"
#include <random>


void vertex_data::add_vertex(float x, float y, float z)
{
    xs.push_back(x);
    ys.push_back(y);
    zs.push_back(z);
}

void vertex_data::concat(const vertex_data& other)
{
    for (auto i = 0u; i < other.xs.size(); ++i)
    {
        xs.push_back(other.xs[i]);
        ys.push_back(other.ys[i]);
        zs.push_back(other.zs[i]);
    }
}

std::vector<float> vertex_data::pack_vertices_nx3() const
{
    std::vector<float> out;
    const auto n = xs.size();

    out.resize(n * 3);
    for (auto i = 0u; i < n; ++i)
    {
        out[i * 3] = xs[i];
        out[i * 3 + 1] = ys[i];
        out[i * 3 + 2] = zs[i];
    }

    return out;
}


vertex_data make_plane(int w, int h, float cx, float cy, float cz, int step)
{
    vertex_data out;
    for (size_t i = 0; i < w; i += step)
    {
        for (size_t j = 0; j < h; j += step)
        {
            float x = static_cast<float>(cx) - (static_cast<float>(w) / 2) + i * step;
            float y = static_cast<float>(cy) - (static_cast<float>(h) / 2) + j * step;

            out.add_vertex(x, y, static_cast<float>(cz));
        }
    }
    return out;
}

vertex_data make_sphere(const int n)
{
    vertex_data out;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0, 1);

    for (auto i = 0; i < n; ++i)
    {
        float x = d(gen);
        float y = d(gen);
        float z = d(gen);

        const float norm = std::sqrt(x * x + y * y + z * z);

        x /= norm;
        y /= norm;
        z /= norm;

        out.add_vertex(x, y, z + 2);
    }
    return out;
}
