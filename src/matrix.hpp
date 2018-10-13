#pragma once

#include "vector.hpp"
#include <vector>

namespace sv4d {

    class Matrix {
        protected:
            std::vector<sv4d::Vector> data;
            int row;
            int col;

        public:
            Matrix();
            Matrix(int m, int n);

            void setZero();
            void setRandomUniform(float min, float max);

            sv4d::Vector& operator[](int idx);
            const sv4d::Vector& operator[](int idx) const;
    };

}