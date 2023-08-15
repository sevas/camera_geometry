#pragma once

#include <chrono>
#include <iostream>
#include <string>


enum class unit
{
    ms,
    us,
    ns
};

// clang-format off
template<unit U> struct unit_to_chrono_unit;
template<> struct unit_to_chrono_unit<unit::ms> { using type = std::chrono::milliseconds; constexpr static char str[] = "ms";};
template<> struct unit_to_chrono_unit<unit::us> { using type = std::chrono::microseconds; constexpr static char str[] = "us";};
template<> struct unit_to_chrono_unit<unit::ns> { using type = std::chrono::nanoseconds;  constexpr static char str[] = "ns";};
// clang-format on


template<unit U>
class scoped_timer
{
    std::chrono::high_resolution_clock::time_point before;
    std::string name;
    unsigned int elapsed;
    std::atomic<bool> stopped = false;

public:
    explicit scoped_timer(std::string name) :
        before(std::chrono::high_resolution_clock::now()),
        name(std::move(name)),
        elapsed(0),
        stopped(false)
    {}

    unsigned int get(){
        const auto after = std::chrono::high_resolution_clock::now();

        using chrono_unit = typename unit_to_chrono_unit<U>::type;
        elapsed = std::chrono::duration_cast<chrono_unit>(after - before).count();
        stopped = true;
        return elapsed;
    }

    ~scoped_timer()
    {
        if (! stopped)
        {
            get();
        }
        const std::string unit_str = unit_to_chrono_unit<U>::str;
        std::cout << "[" << name << "] " << elapsed << " " << unit_str << std::endl;
    }
};


typedef scoped_timer<unit::ms> scoped_timer_ms;
typedef scoped_timer<unit::us> scoped_timer_us;
typedef scoped_timer<unit::ns> scoped_timer_ns;
