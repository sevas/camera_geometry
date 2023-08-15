#pragma once
#include <cstdint>
#include <string>
#include <vector>


namespace cg::imageio {
    void imwrite(
        const std::string& filename, int w, int h, int channels, const std::vector<uint8_t>& data);

    std::vector<uint8_t> grayscale_to_rgb(const std::vector<uint8_t>& rgb);
}

