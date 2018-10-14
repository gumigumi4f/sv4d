#pragma once

#include "vector.hpp"
#include <vector>

namespace sv4d {

    class Matrix {
        public:
            Matrix();
            Matrix(int m, int n);

            std::vector<sv4d::Vector> data;
            int row;
            int col;

            void setZero();
            void setRandomUniform(float min, float max);

            inline sv4d::Vector& operator[](int idx) {
                return data[idx];
            }

            inline const sv4d::Vector& operator[](int idx) const {
                return data[idx];
            }
    };

}