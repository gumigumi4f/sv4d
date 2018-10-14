#include "vector.hpp"

#include <vector>
#include <random>

namespace sv4d {

    Vector::Vector() : Vector(0) {}

    Vector::Vector(int n) : data(n, 0.0), col(n) {}

    void Vector::setZero() {
        std::fill(data.begin(), data.end(), 0.0);
    }

    void Vector::setRandomUniform(float min, float max) {
        std::mt19937 mt(495);
        std::uniform_real_distribution<float> r(min, max);
        for (int i = 0; i < col; ++i) {
            data[i] = r(mt);
        }
    }

    std::vector<float>& Vector::getData() {
        return data;
    }

}