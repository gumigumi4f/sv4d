#include "vector.hpp"

#include "utils.hpp"
#include <vector>
#include <complex>
#include <random>
#include <numeric>

namespace sv4d {

    Vector::Vector() : Vector(0) {}

    Vector::Vector(int n) : col(n), data(col, 0.0) {}

    Vector::Vector(const std::vector<float>& vec) : col(vec.size()), data(vec) {}

    void Vector::setZero()
    {
        std::fill(data.begin(), data.end(), 0.0);
    }

    void Vector::setRandomUniform(float min, float max)
    {
        std::mt19937 mt(495);
        std::uniform_real_distribution<float> r(-min, max);
        for (int i = 0; i < col; ++i) {
            data[i] = r(mt);
        }
    }

    float Vector::sum() {
        return std::accumulate(data.begin(), data.end(), 0);
    }

    sv4d::Vector Vector::sigmoid() {
        sv4d::Vector outputVector = sv4d::Vector(col);
        for (int i = 0; i < col; ++i) {
            outputVector.data[i] += sv4d::utils::operation::sigmoid(data[i]);
        }
        return outputVector;
    }

    sv4d::Vector Vector::softmax(float temperature) {
        sv4d::Vector outputVector = sv4d::Vector(col);
        float max = 0.0;
        for (int i = 0; i < col; ++i) {
            auto logit = data[i] / temperature;
            outputVector.data[i] = logit;
            max = std::max(max, logit);
        }
        float sum = 0.0;
        for (int i = 0; i < col; ++i) {
            outputVector.data[i] = std::exp(outputVector.data[i] - max);
            sum += outputVector.data[i];
        }
        for (int i = 0; i < col; ++i) {
            outputVector.data[i] /= sum;
        }
        return outputVector;
    }

    sv4d::Vector Vector::clipByValue(float min, float max) {
        sv4d::Vector outputVector = sv4d::Vector(col);
        for (int i = 0; i < col; ++i) {
            if (data[i] > max) {
                outputVector.data[i] = max;
            } else if (data[i] < min) {
                outputVector.data[i] = min;
            } else {
                outputVector.data[i] = data[i];
            }
        }
        return outputVector;
    }

    float& sv4d::Vector::operator[](int idx) {
        return data[idx];
    }

    const float& sv4d::Vector::operator[](int idx) const {
        return data[idx];
    }

    sv4d::Vector sv4d::Vector::operator+(const sv4d::Vector& vector) {
        sv4d::Vector outputVector = sv4d::Vector(col);
        outputVector += *this;
        outputVector += vector;
        return outputVector;
    }

    sv4d::Vector sv4d::Vector::operator-(const sv4d::Vector& vector) {
        sv4d::Vector outputVector = sv4d::Vector(col);
        outputVector += *this;
        outputVector -= vector;
        return outputVector;
    }

    sv4d::Vector sv4d::Vector::operator*(const sv4d::Vector& vector) {
        sv4d::Vector outputVector = sv4d::Vector(col);
        outputVector += *this;
        outputVector *= vector;
        return outputVector;
    }

    sv4d::Vector sv4d::Vector::operator/(const sv4d::Vector& vector) {
        sv4d::Vector outputVector = sv4d::Vector(col);
        outputVector += *this;
        outputVector /= vector;
        return outputVector;
    }

    sv4d::Vector sv4d::Vector::operator+(const float value) {
        sv4d::Vector outputVector = sv4d::Vector(col);
        outputVector += *this;
        outputVector += value;
        return outputVector;
    }
    
    sv4d::Vector sv4d::Vector::operator-(const float value) {
        sv4d::Vector outputVector = sv4d::Vector(col);
        outputVector += *this;
        outputVector -= value;
        return outputVector;
    }

    sv4d::Vector sv4d::Vector::operator*(const float value) {
        sv4d::Vector outputVector = sv4d::Vector(col);
        outputVector += *this;
        outputVector *= value;
        return outputVector;
    }

    sv4d::Vector sv4d::Vector::operator/(const float value) {
        sv4d::Vector outputVector = sv4d::Vector(col);
        outputVector += *this;
        outputVector /= value;
        return outputVector;
    }

    sv4d::Vector sv4d::Vector::operator+(const int value) {
        sv4d::Vector outputVector = sv4d::Vector(col);
        outputVector += *this;
        outputVector += value;
        return outputVector;
    }

    sv4d::Vector sv4d::Vector::operator-(const int value) {
        sv4d::Vector outputVector = sv4d::Vector(col);
        outputVector += *this;
        outputVector -= value;
        return outputVector;
    }

    sv4d::Vector sv4d::Vector::operator*(const int value) {
        sv4d::Vector outputVector = sv4d::Vector(col);
        outputVector += *this;
        outputVector *= value;
        return outputVector;
    }

    sv4d::Vector sv4d::Vector::operator/(const int value) {
        sv4d::Vector outputVector = sv4d::Vector(col);
        outputVector += *this;
        outputVector /= value;
        return outputVector;
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

    sv4d::Vector sv4d::Vector::operator+=(const float value) {
        for (int i = 0; i < col; ++i) {
            data[i] += value;
        }
        return *this;
    }

    sv4d::Vector sv4d::Vector::operator-=(const float value) {
        for (int i = 0; i < col; ++i) {
            data[i] -= value;
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

    sv4d::Vector sv4d::Vector::operator+=(const int value) {
        for (int i = 0; i < col; ++i) {
            data[i] += value;
        }
        return *this;
    }

    sv4d::Vector sv4d::Vector::operator-=(const int value) {
        for (int i = 0; i < col; ++i) {
            data[i] -= value;
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
        sv4d::Vector outputVector = sv4d::Vector(col);
        outputVector += *this;
        return outputVector;
    }

    sv4d::Vector sv4d::Vector::operator-() {
        sv4d::Vector outputVector = sv4d::Vector(col);
        outputVector -= *this;
        return outputVector;
    }

    float sv4d::Vector::operator%(const sv4d::Vector& vector) {
        float dot = 0.0;
        for (int i = 0; i < col; ++i) {
            dot += data[i] * vector.data[i];
        }
        return dot;
    }

}