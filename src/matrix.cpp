#include "matrix.hpp"

#include "vector.hpp"
#include <vector>
#include <random>

namespace sv4d {

    Matrix::Matrix() : Matrix(0, 0) {}

    Matrix::Matrix(int m, int n) : data(m, sv4d::Vector(n)), row(m), col(n) {}

    void Matrix::setZero() {
        for (int i = 0; i < row; i++) {
            data[i].setZero();
        }
    }
    
    void Matrix::setRandomUniform(float min, float max) {
        std::mt19937 mt(495);
        std::uniform_real_distribution<float> r(min, max);
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; j++) {
                data[i][j] = r(mt);
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