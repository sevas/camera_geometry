#include "scoped_timer.hpp"
#include <sstream>

std::string join(const std::vector<std::string>& stacked_names, const std::string& sep)
{
    std::ostringstream ss;

    for (auto& s : stacked_names) {
        ss << s << sep;
    }
    return ss.str();
}