
#include "imageio.h"
#include <cstdio>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

std::string splitext(const std::string filename)
{
    auto pos = filename.find_last_of('.');
    if (pos == std::string::npos)
    {
        return filename;
    }
    else
    {
        return filename.substr(pos+1, filename.size());
    }
}

void imwrite(const std::string& filename, const int w, const int h, const int channels, const std::vector<uint8_t>& data)
{
    const auto ext = splitext(filename);
    if (ext == "png")
    {
        stbi_write_png(filename.c_str(), w, h, channels, data.data(), w * channels);
    }
    else if (ext == "bin")
    {
        FILE* fp = fopen(filename.c_str(), "wb");
        fwrite(&w, 4, 1, fp);
        fwrite(&h, 4, 1, fp);
        fwrite(&channels, 4, 1, fp);
        fwrite(data.data(), sizeof(uint8_t), data.size(), fp);
        fclose(fp);
    }
    else
    {
        throw std::runtime_error("Unsupported file extension: " + ext);
    }
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
