#pragma once

#include "utils.hpp"
#include <vector>
#include <complex>
#include <numeric>

namespace sv4d {

    class Vector {
        public:
            Vector();
            Vector(int n);

            std::vector<float> data;
            int col;

            void setZero();
            void setRandomUniform(float min, float max);
            std::vector<float>& getData();

            inline float sum() {
                return std::accumulate(data.begin(), data.end(), 0);
            }

            inline sv4d::Vector sigmoid() {
                sv4d::Vector outputVector(col);
                for (int i = 0; i < col; ++i) {
                    outputVector.data[i] += sv4d::utils::operation::sigmoid(data[i]);
                }
                return outputVector;
            }

            inline sv4d::Vector softmax(float temperature) {
                sv4d::Vector outputVector(col);
                float max = 0.0;
                for (int i = 0; i < col; ++i) {
                    auto logit = data[i] / temperature;
                    outputVector.data[i] = logit;
                    max = std::max(max, logit);
                }
                float sum = 0.0;
                for (auto& x : outputVector.data) {
                    x = std::exp(x - max);
                    sum += x;
                }
                for (auto& x : outputVector.data) {
                    sum /= x;
                }
                return outputVector;
            }

            inline void clipByValue(float min, float max) {
                for (auto& x : data) {
                    if (x > max) {
                        x = max;
                    } else if (x < min) {
                        x = min;
                    }
                }
            }

            inline float& operator[](int idx) {
                return data[idx];
            }

            inline const float& operator[](int idx) const {
                return data[idx];
            }

            inline sv4d::Vector operator+(const sv4d::Vector& vector) {
                sv4d::Vector outputVector(col);
                outputVector += *this;
                outputVector += vector;
                return outputVector;
            }

            inline sv4d::Vector operator-(const sv4d::Vector& vector) {
                sv4d::Vector outputVector(col);
                outputVector += *this;
                outputVector -= vector;
                return outputVector;
            }

            inline sv4d::Vector operator*(const sv4d::Vector& vector) {
                sv4d::Vector outputVector(col);
                outputVector += *this;
                outputVector *= vector;
                return outputVector;
            }

            inline sv4d::Vector operator/(const sv4d::Vector& vector) {
                sv4d::Vector outputVector(col);
                outputVector += *this;
                outputVector /= vector;
                return outputVector;
            }

            inline sv4d::Vector operator+(const float value) {
                sv4d::Vector outputVector(col);
                outputVector += *this;
                outputVector += value;
                return outputVector;
            }
            
            inline sv4d::Vector operator-(const float value) {
                sv4d::Vector outputVector(col);
                outputVector += *this;
                outputVector -= value;
                return outputVector;
            }

            inline sv4d::Vector operator*(const float value) {
                sv4d::Vector outputVector(col);
                outputVector += *this;
                outputVector *= value;
                return outputVector;
            }

            inline sv4d::Vector operator/(const float value) {
                sv4d::Vector outputVector(col);
                outputVector += *this;
                outputVector /= value;
                return outputVector;
            }

            inline sv4d::Vector operator+(const int value) {
                sv4d::Vector outputVector(col);
                outputVector += *this;
                outputVector += value;
                return outputVector;
            }

            inline sv4d::Vector operator-(const int value) {
                sv4d::Vector outputVector(col);
                outputVector += *this;
                outputVector -= value;
                return outputVector;
            }

            inline sv4d::Vector operator*(const int value) {
                sv4d::Vector outputVector(col);
                outputVector += *this;
                outputVector *= value;
                return outputVector;
            }

            inline sv4d::Vector operator/(const int value) {
                sv4d::Vector outputVector(col);
                outputVector += *this;
                outputVector /= value;
                return outputVector;
            }

            inline sv4d::Vector operator+=(const sv4d::Vector& vector) {
                for (int i = 0; i < col; ++i) {
                    data[i] += vector.data[i];
                }
                return *this;
            }

            inline sv4d::Vector operator-=(const sv4d::Vector& vector) {
                for (int i = 0; i < col; ++i) {
                    data[i] -= vector.data[i];
                }
                return *this;
            }

            inline sv4d::Vector operator*=(const sv4d::Vector& vector) {
                for (int i = 0; i < col; ++i) {
                    data[i] *= vector.data[i];
                }
                return *this;
            }
            
            inline sv4d::Vector operator/=(const sv4d::Vector& vector) {
                for (int i = 0; i < col; ++i) {
                    data[i] /= vector.data[i];
                }
                return *this;
            }

            inline sv4d::Vector operator+=(const float value) {
                for (int i = 0; i < col; ++i) {
                    data[i] += value;
                }
                return *this;
            }

            inline sv4d::Vector operator-=(const float value) {
                for (int i = 0; i < col; ++i) {
                    data[i] -= value;
                }
                return *this;
            }

            inline sv4d::Vector operator*=(const float value) {
                for (int i = 0; i < col; ++i) {
                    data[i] *= value;
                }
                return *this;
            }

            inline sv4d::Vector operator/=(const float value) {
                for (int i = 0; i < col; ++i) {
                    data[i] /= value;
                }
                return *this;
            }

            inline sv4d::Vector operator+=(const int value) {
                for (int i = 0; i < col; ++i) {
                    data[i] += value;
                }
                return *this;
            }

            inline sv4d::Vector operator-=(const int value) {
                for (int i = 0; i < col; ++i) {
                    data[i] -= value;
                }
                return *this;
            }

            inline sv4d::Vector operator*=(const int value) {
                for (int i = 0; i < col; ++i) {
                    data[i] *= value;
                }
                return *this;
            }

            inline sv4d::Vector operator/=(const int value) {
                for (int i = 0; i < col; ++i) {
                    data[i] /= value;
                }
                return *this;
            }

            inline sv4d::Vector operator+() {
                sv4d::Vector outputVector(col);
                outputVector += *this;
                return outputVector;
            }

            inline sv4d::Vector operator-() {
                sv4d::Vector outputVector(col);
                outputVector -= *this;
                return outputVector;
            }

            inline float operator%(const sv4d::Vector& vector) {
                float dot = 0.0;
                for (int i = 0; i < col; ++i) {
                    dot += data[i] * vector.data[i];
                }
                return dot;
            }
    };

}