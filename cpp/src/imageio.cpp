
#include "imageio.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

void imwrite(const std::string& filename, const int w, const int h, const std::vector<uint8_t>& data)
{
    stbi_write_png(filename.c_str(), w, h, 3, data.data(), w * 3);
}

std::vector<uint8_t> grayscale_to_rgb(const std::vector<uint8_t>& rgb)
{
    std::vector<uint8_t> out;
    out.resize(rgb.size() * 3);

    for (auto i = 0u; i < rgb.size(); ++i)
    {
        out[i * 3] = rgb[i];
        out[i * 3 + 1] = rgb[i];
        out[i * 3 + 2] = rgb[i];
    }

    return out;
}
