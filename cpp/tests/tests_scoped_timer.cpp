#include "doctest.h"
#include "scoped_timer.hpp"
#include <set>

using namespace cg::timings;
TEST_CASE("scoped timer") {

    {
        scoped_timer_us level0("level0");
        {
            scoped_timer_us level1("level1");
            {
                scoped_timer_us level2("level2");
            }
        }
    }

    std::set<std::string> expected_names = {"level0", "level0/level1", "level0/level1/level2"};
    CHECK(scoped_timer_us::all_timings.size() == expected_names.size());

    for (const auto& [name, session] : scoped_timer_us::all_timings) {
        CHECK(expected_names.find(name) != expected_names.end());
        CHECK(session.timings.size() == 1);
    }

}
