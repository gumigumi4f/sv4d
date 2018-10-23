#include "vector.hpp"

#include <vector>
#include <random>
#include <limits>

namespace sv4d {

    Vector::Vector() : Vector(0) {}

    Vector::Vector(int n) : data(n, 0.0), col(n) {}

    void Vector::setZero() {
        for (float& x : data) {
            x = 0.0f;
        }
    }

    void Vector::setRandomUniform(float min, float max) {
        std::mt19937 mt(495);
        std::uniform_real_distribution<float> r(min, max);
        for (float& x : data) {
            x = r(mt);
        }
    }

    std::vector<float>& Vector::getData() {
        return data;
    }

    float Vector::sum() {
        float sum = 0.0f;
        for (float& x : data) {
            sum += x;
        }
        return sum;
    }

    sv4d::Vector Vector::sigmoid() {
        sv4d::Vector outputVector(col);
        for (int i = 0; i < col; ++i) {
            outputVector.data[i] += sv4d::utils::operation::sigmoid(data[i]);
        }
        return outputVector;
    }

    sv4d::Vector Vector::softmax(float temperature) {
        sv4d::Vector outputVector(col);
        if (col == 1) {
            outputVector[0] = 1.0f;
        } else {
            float max = std::numeric_limits<float>::min();
            for (int i = 0; i < col; ++i) {
                float logit = data[i] / temperature;
                outputVector.data[i] = logit;
                max = std::max(max, logit);
            }
            float sum = 0.0f;
            for (float& x : outputVector.data) {
                x = std::exp(x - max);
                sum += x;
            }
            for (float& x : outputVector.data) {
                x /= sum;
            }
        }
        return outputVector;
    }

    void Vector::addVectorWithFactor(const sv4d::Vector& vector, const float factor) {
        for (int i = 0; i < col; ++i) {
            data[i] += vector.data[i] * factor;
        }
    }

    sv4d::Vector Vector::operator+(const sv4d::Vector& vector) {
        sv4d::Vector outputVector(col);
        outputVector += *this;
        outputVector += vector;
        return outputVector;
    }

    sv4d::Vector Vector::operator-(const sv4d::Vector& vector) {
        sv4d::Vector outputVector(col);
        outputVector += *this;
        outputVector -= vector;
        return outputVector;
    }

    sv4d::Vector Vector::operator*(const sv4d::Vector& vector) {
        sv4d::Vector outputVector(col);
        outputVector += *this;
        outputVector *= vector;
        return outputVector;
    }

    sv4d::Vector Vector::operator/(const sv4d::Vector& vector) {
        sv4d::Vector outputVector(col);
        outputVector += *this;
        outputVector /= vector;
        return outputVector;
    }

    sv4d::Vector Vector::operator+(const float value) {
        sv4d::Vector outputVector(col);
        outputVector += *this;
        outputVector += value;
        return outputVector;
    }

    sv4d::Vector Vector::operator-(const float value) {
        sv4d::Vector outputVector(col);
        outputVector += *this;
        outputVector -= value;
        return outputVector;
    }

    sv4d::Vector Vector::operator*(const float value) {
        sv4d::Vector outputVector(col);
        outputVector += *this;
        outputVector *= value;
        return outputVector;
    }

    sv4d::Vector Vector::operator/(const float value) {
        sv4d::Vector outputVector(col);
        outputVector += *this;
        outputVector /= value;
        return outputVector;
    }

    sv4d::Vector Vector::operator+(const int value) {
        sv4d::Vector outputVector(col);
        outputVector += *this;
        outputVector += value;
        return outputVector;
    }

    sv4d::Vector Vector::operator-(const int value) {
        sv4d::Vector outputVector(col);
        outputVector += *this;
        outputVector -= value;
        return outputVector;
    }

    sv4d::Vector Vector::operator*(const int value) {
        sv4d::Vector outputVector(col);
        outputVector += *this;
        outputVector *= value;
        return outputVector;
    }

    sv4d::Vector Vector::operator/(const int value) {
        sv4d::Vector outputVector(col);
        outputVector += *this;
        outputVector /= value;
        return outputVector;
    }

    sv4d::Vector Vector::operator+=(const sv4d::Vector& vector) {
        for (int i = 0; i < col; ++i) {
            data[i] += vector.data[i];
        }
        return *this;
    }

    sv4d::Vector Vector::operator-=(const sv4d::Vector& vector) {
        for (int i = 0; i < col; ++i) {
            data[i] -= vector.data[i];
        }
        return *this;
    }

    sv4d::Vector Vector::operator*=(const sv4d::Vector& vector) {
        for (int i = 0; i < col; ++i) {
            data[i] *= vector.data[i];
        }
        return *this;
    }

    sv4d::Vector Vector::operator/=(const sv4d::Vector& vector) {
        for (int i = 0; i < col; ++i) {
            data[i] /= vector.data[i];
        }
        return *this;
    }

    sv4d::Vector Vector::operator+=(const float value) {
        for (int i = 0; i < col; ++i) {
            data[i] += value;
        }
        return *this;
    }

    sv4d::Vector Vector::operator-=(const float value) {
        for (int i = 0; i < col; ++i) {
            data[i] -= value;
        }
        return *this;
    }

    sv4d::Vector Vector::operator*=(const float value) {
        for (int i = 0; i < col; ++i) {
            data[i] *= value;
        }
        return *this;
    }

    sv4d::Vector Vector::operator/=(const float value) {
        for (int i = 0; i < col; ++i) {
            data[i] /= value;
        }
        return *this;
    }

    sv4d::Vector Vector::operator+=(const int value) {
        for (int i = 0; i < col; ++i) {
            data[i] += value;
        }
        return *this;
    }

    sv4d::Vector Vector::operator-=(const int value) {
        for (int i = 0; i < col; ++i) {
            data[i] -= value;
        }
        return *this;
    }

    sv4d::Vector Vector::operator*=(const int value) {
        for (int i = 0; i < col; ++i) {
            data[i] *= value;
        }
        return *this;
    }

    sv4d::Vector Vector::operator/=(const int value) {
        for (int i = 0; i < col; ++i) {
            data[i] /= value;
        }
        return *this;
    }

    sv4d::Vector Vector::operator+() {
        sv4d::Vector outputVector(col);
        outputVector += *this;
        return outputVector;
    }

    sv4d::Vector Vector::operator-() {
        sv4d::Vector outputVector(col);
        outputVector -= *this;
        return outputVector;
    }

    float Vector::operator%(const sv4d::Vector& vector) {
        float dot = 0.0;
        for (int i = 0; i < col; ++i) {
            dot += data[i] * vector.data[i];
        }
        return dot;
    }

}