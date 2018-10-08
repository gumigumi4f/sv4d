#include "vector.hpp"

#include <vector>
#include <complex>
#include <random>

namespace sv4d {

    Vector::Vector() : Vector(0) {}

    Vector::Vector(int n) {
        col = n;

        data = std::vector<float>(col, 0.0);
    }

    Vector::Vector(const std::vector<float>& vec) {
        col = vec.size();

        data = vec;
    }

    void Vector::setZero()
    {
        std::fill(data.begin(), data.end(), 0.0f);
    }

    void Vector::setGlorotUniform()
    {
        float min = -std::sqrt(6.0 / col);
        float max = std::sqrt(6.0 / col);

        std::random_device rnd;
        std::mt19937 mt(rnd());
        std::uniform_real_distribution<float> glorot(min, max);
        for (int i = 0; i < col; ++i) {
            data[i] = glorot(mt);
        }
    }

    float& sv4d::Vector::operator[](int idx) {
        return data[idx];
    }

    const float& sv4d::Vector::operator[](int idx) const {
        return data[idx];
    }

    sv4d::Vector sv4d::Vector::operator+(const sv4d::Vector& vector) {
        sv4d::Vector newVector = sv4d::Vector(col);
        newVector += *this;
        newVector += vector;
        return newVector;
    }

    sv4d::Vector sv4d::Vector::operator-(const sv4d::Vector& vector) {
        sv4d::Vector newVector = sv4d::Vector(col);
        newVector += *this;
        newVector -= vector;
        return newVector;
    }

    sv4d::Vector sv4d::Vector::operator*(const sv4d::Vector& vector) {
        sv4d::Vector newVector = sv4d::Vector(col);
        newVector += *this;
        newVector *= vector;
        return newVector;
    }

    sv4d::Vector sv4d::Vector::operator/(const sv4d::Vector& vector) {
        sv4d::Vector newVector = sv4d::Vector(col);
        newVector += *this;
        newVector /= vector;
        return newVector;
    }

    sv4d::Vector sv4d::Vector::operator*(const float value) {
        sv4d::Vector newVector = sv4d::Vector(col);
        newVector += *this;
        newVector *= value;
        return newVector;
    }

    sv4d::Vector sv4d::Vector::operator/(const float value) {
        sv4d::Vector newVector = sv4d::Vector(col);
        newVector += *this;
        newVector /= value;
        return newVector;
    }

    sv4d::Vector sv4d::Vector::operator*(const int value) {
        sv4d::Vector newVector = sv4d::Vector(col);
        newVector += *this;
        newVector *= value;
        return newVector;
    }

    sv4d::Vector sv4d::Vector::operator/(const int value) {
        sv4d::Vector newVector = sv4d::Vector(col);
        newVector += *this;
        newVector /= value;
        return newVector;
    }

    sv4d::Vector sv4d::Vector::operator+=(const sv4d::Vector& vector) {
        for (int i = 0; i < col; ++i) {
            data[i] += vector.data[i];
        }
        return *this;
    }

    sv4d::Vector sv4d::Vector::operator-=(const sv4d::Vector& vector) {
        for (int i = 0; i < col; ++i) {
            data[i] -= vector.data[i];
        }
        return *this;
    }

    sv4d::Vector sv4d::Vector::operator*=(const sv4d::Vector& vector) {
        for (int i = 0; i < col; ++i) {
            data[i] *= vector.data[i];
        }
        return *this;
    }
    
    sv4d::Vector sv4d::Vector::operator/=(const sv4d::Vector& vector) {
        for (int i = 0; i < col; ++i) {
            data[i] /= vector.data[i];
        }
        return *this;
    }

    sv4d::Vector sv4d::Vector::operator*=(const float value) {
        for (int i = 0; i < col; ++i) {
            data[i] *= value;
        }
        return *this;
    }

    sv4d::Vector sv4d::Vector::operator/=(const float value) {
        for (int i = 0; i < col; ++i) {
            data[i] /= value;
        }
        return *this;
    }

    sv4d::Vector sv4d::Vector::operator*=(const int value) {
        for (int i = 0; i < col; ++i) {
            data[i] *= value;
        }
        return *this;
    }

    sv4d::Vector sv4d::Vector::operator/=(const int value) {
        for (int i = 0; i < col; ++i) {
            data[i] /= value;
        }
        return *this;
    }

    sv4d::Vector sv4d::Vector::operator+() {
        sv4d::Vector newVector = sv4d::Vector(col);
        newVector += *this;
        return newVector;
    }

    sv4d::Vector sv4d::Vector::operator-() {
        sv4d::Vector newVector = sv4d::Vector(col);
        newVector -= *this;
        return newVector;
    }

}