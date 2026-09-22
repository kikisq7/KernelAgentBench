#include "../gray_scott/reference.hpp"

#include <cassert>
#include <cmath>

int main() {
    using namespace gray_scott_reference;
    constexpr int height = 31;
    constexpr int width = 29;
    const Fields initial = initialize_fields(height, width);
    const Fields first = simulate(initial, height, width, Params{}, 5);
    const Fields second = simulate(initial, height, width, Params{}, 5);
    assert(first == second);
    for (int x = 0; x < width; ++x) {
        assert(first.first[x] == initial.first[x]);
        assert(first.first[(height - 1) * width + x] == initial.first[(height - 1) * width + x]);
        assert(first.second[x] == initial.second[x]);
        assert(first.second[(height - 1) * width + x] == initial.second[(height - 1) * width + x]);
    }
    for (float value : first.first) assert(std::isfinite(value));
    for (float value : first.second) assert(std::isfinite(value));
}
