#include "../gray_scott/notebook.hpp"

#include <cmath>
#include <iostream>
#include <numeric>

// Shared with julia/test and python/test: the three harnesses start bit-identical.
int main() {
    const GrayScottFields fields = initialize_fields(31, 29);
    const int draws[] = {70, 10, 16, 94, 47, 3, 53, 17};
    bool ok = fields.nx == 31 && fields.ny == 29 && fields.U.size() == 31u * 29u;
    for (int index = 0; index < 8; ++index) {
        ok = ok && static_cast<int>(std::lround(fields.U[index] * 1000)) == draws[index];
    }
    ok = ok && fields.U.back() == 0.036;
    ok = ok && std::accumulate(fields.V.begin(), fields.V.end(), 0.0) == 30.25;
    // Julia's U[15, 14] (1-based) is the centre of the disturbance.
    ok = ok && fields.U[14 + 31 * 13] == 0.5 && fields.V[14 + 31 * 13] == 0.25;
    if (!ok) {
        std::cerr << "notebook initial condition does not match the Julia harness\n";
        return 1;
    }
    std::cout << "notebook initial condition matches\n";
    return 0;
}
