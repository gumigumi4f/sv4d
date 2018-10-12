#pragma once

#include "vector.hpp"
#include <vector>

namespace sv4d {

    class Matrix {
        public:
            Matrix();
            Matrix(int m, int n);

            int row;
            int col;

            std::vector<sv4d::Vector> data;

            void setZero();
            void setRandomUniform(float min, float max);

            sv4d::Vector& operator[](int idx);
            const sv4d::Vector& operator[](int idx) const;
    };

}