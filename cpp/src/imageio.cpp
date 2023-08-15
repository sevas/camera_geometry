
#include "imageio.h"
#include <array>
#include <cstdio>
#include <stdexcept>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma warning(push, 3)
#include <stb_image_write.h>
#pragma warning(pop)

namespace cg::imageio {

std::string splitext(const std::string& filename)
{
    auto pos = filename.find_last_of('.');
    if (pos == std::string::npos) {
        return filename;
    } else {
        return filename.substr(pos + 1, filename.size());
    }
}

void imwrite(const std::string& filename, const int w, const int h, const int channels,
    const std::vector<uint8_t>& data)
{
    const auto ext = splitext(filename);
    if (ext == "png") {
        stbi_write_png(filename.c_str(), w, h, channels, data.data(), w * channels);
    } else if (ext == "bin") {
        FILE* fp = fopen(filename.c_str(), "wb");
        fwrite(&w, 4, 1, fp);
        fwrite(&h, 4, 1, fp);
        fwrite(&channels, 4, 1, fp);
        fwrite(data.data(), sizeof(uint8_t), data.size(), fp);
        fclose(fp);
    } else {
        throw std::runtime_error("Unsupported file extension: " + ext);
    }
}

std::vector<uint8_t> grayscale_to_rgb(const std::vector<uint8_t>& grayscale)
{
    std::vector<uint8_t> out;
    out.resize(grayscale.size() * 3);

    for (auto i = 0u; i < grayscale.size(); ++i) {
        out[i * 3] = grayscale[i];
        out[i * 3 + 1] = grayscale[i];
        out[i * 3 + 2] = grayscale[i];
    }

    return out;
}

std::array<float, 3> lerp(std::array<float, 3>& a, std::array<float, 3>& b, float alpha)
{
    std::array<float, 3> out { 0, 0, 0 };

    out[0] = (1.f - alpha) * a[0] + alpha * b[0];
    out[1] = (1.f - alpha) * a[1] + alpha * b[1];
    out[2] = (1.f - alpha) * a[2] + alpha * b[2];

    return out;
}

std::array<uint8_t, 3> lerp(std::array<uint8_t, 3>& a, std::array<uint8_t, 3>& b, float alpha)
{
    std::array<float, 3> ap
        = { static_cast<float>(a[0]), static_cast<float>(a[1]), static_cast<float>(a[2]) };
    std::array<float, 3> bp
        = { static_cast<float>(b[0]), static_cast<float>(b[1]), static_cast<float>(b[2]) };

    const auto out = lerp(ap, bp, alpha);

    return std::array<uint8_t, 3> {
        static_cast<uint8_t>(floor(out[0])),
        static_cast<uint8_t>(floor(out[1])),
        static_cast<uint8_t>(floor(out[2])),
    };
}

std::array<uint8_t, 3> viridis_palette[] = { { 68, 1, 84 }, { 72, 12, 115 }, { 70, 27, 126 },
    { 64, 41, 129 }, { 56, 54, 130 }, { 46, 66, 129 }, { 35, 78, 123 }, { 24, 91, 114 },
    { 15, 103, 103 }, { 17, 118, 90 }, { 36, 131, 76 }, { 68, 144, 63 }, { 111, 155, 53 },
    { 165, 166, 46 }, { 222, 176, 38 }, { 249, 210, 4 } };

std::vector<uint8_t> apply_palette(const std::vector<uint8_t>& grayscale)
{
    std::vector<uint8_t> out;
    out.resize(grayscale.size() * 3);

    for (auto i = 0u; i < grayscale.size(); ++i) {
        const auto val = grayscale[i];
        const auto alpha = remainder(val, sizeof viridis_palette);
        const auto idx = static_cast<size_t>(floor(val / sizeof viridis_palette));
        const auto color
            = lerp(viridis_palette[idx], viridis_palette[idx + 1], static_cast<float>(alpha));

        out[i * 3] = color[0];
        out[i * 3 + 1] = color[1];
        out[i * 3 + 2] = color[2];
    }

    return out;
}
}
