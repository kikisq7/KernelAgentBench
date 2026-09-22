#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gray_scott_reference {

struct Params {
    float diffusion_u = 0.16f;
    float diffusion_v = 0.08f;
    float feed = 0.035f;
    float kill = 0.065f;
    float dt = 1.0f;
};

using Fields = std::pair<std::vector<float>, std::vector<float>>;

inline Fields initialize_fields(int height, int width, int seed = 20260918) {
    if (height < 5 || width < 5) {
        throw std::invalid_argument("shape must be at least 5x5");
    }
    Fields fields{
        std::vector<float>(static_cast<std::size_t>(height) * width, 1.0f),
        std::vector<float>(static_cast<std::size_t>(height) * width, 0.0f),
    };
    const int radius_y = std::max(1, height / 10);
    const int radius_x = std::max(1, width / 10);
    const int center_y = (height - 1) / 2;
    const int center_x = (width - 1) / 2;
    for (int y = std::max(1, center_y - radius_y);
         y <= std::min(height - 2, center_y + radius_y);
         ++y) {
        for (int x = std::max(1, center_x - radius_x);
             x <= std::min(width - 2, center_x + radius_x);
             ++x) {
            const float noise = 0.01f * std::sin(
                ((y + 1) * 73856093.0 + (x + 1) * 19349663.0 + seed) * 0.001
            );
            const int index = y * width + x;
            fields.first[index] = 0.5f + noise;
            fields.second[index] = 0.25f - noise;
        }
    }
    return fields;
}

inline Fields simulate(Fields fields, int height, int width, Params params, int steps) {
    Fields next = fields;
    for (int step = 0; step < steps; ++step) {
        next = fields;
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                const int index = y * width + x;
                const float u = fields.first[index];
                const float v = fields.second[index];
                const float lap_u =
                    fields.first[index - width] + fields.first[index + width] +
                    fields.first[index - 1] + fields.first[index + 1] - 4.0f * u;
                const float lap_v =
                    fields.second[index - width] + fields.second[index + width] +
                    fields.second[index - 1] + fields.second[index + 1] - 4.0f * v;
                const float reaction = u * v * v;
                next.first[index] = u + params.dt *
                    (params.diffusion_u * lap_u - reaction + params.feed * (1.0f - u));
                next.second[index] = v + params.dt *
                    (params.diffusion_v * lap_v + reaction - (params.feed + params.kill) * v);
            }
        }
        fields.swap(next);
    }
    return fields;
}

}  // namespace gray_scott_reference
