#include "matrix.hpp"

#include "vector.hpp"
#include <vector>
#include <random>

namespace sv4d {

    Matrix::Matrix() : Matrix(0, 0) {}

    Matrix::Matrix(int m, int n) {
        row = m;
        col = n;

        data = std::vector<sv4d::Vector>(row, sv4d::Vector(n));
    }

    void Matrix::setZero()
    {
        for (int i = 0; i < row; i++) {
            data[i].setZero();
        }
    }
    
    void Matrix::setGlorotUniform()
    {
        float min = -std::sqrt(6.0 / (row + col));
        float max = std::sqrt(6.0 / (row + col));

        std::random_device rnd;
        std::mt19937 mt(rnd());
        std::uniform_real_distribution<float> glorot(min, max);
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; j++) {
                data[i][j] = glorot(mt);
            }
        }
    }

    sv4d::Vector& Matrix::operator[](int idx) {
        return data[idx];
    }

    const sv4d::Vector& Matrix::operator[](int idx) const {
        return data[idx];
    }

}