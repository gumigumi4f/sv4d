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
            float sum();
            sv4d::Vector sigmoid();
            sv4d::Vector softmax(float temperature);
            void clipByValue(float min, float max);

            inline float& operator[](int idx) {
                return data[idx];
            }

            inline const float& operator[](int idx) const {
                return data[idx];
            }

            sv4d::Vector operator+(const sv4d::Vector& vector);
            sv4d::Vector operator-(const sv4d::Vector& vector);
            sv4d::Vector operator*(const sv4d::Vector& vector);
            sv4d::Vector operator/(const sv4d::Vector& vector);
            sv4d::Vector operator+(const float value);
            sv4d::Vector operator-(const float value);
            sv4d::Vector operator*(const float value);
            sv4d::Vector operator/(const float value);
            sv4d::Vector operator+(const int value);
            sv4d::Vector operator-(const int value);
            sv4d::Vector operator*(const int value);
            sv4d::Vector operator/(const int value);
            sv4d::Vector operator+=(const sv4d::Vector& vector);
            sv4d::Vector operator-=(const sv4d::Vector& vector);
            sv4d::Vector operator*=(const sv4d::Vector& vector);
            sv4d::Vector operator/=(const sv4d::Vector& vector);
            sv4d::Vector operator+=(const float value);
            sv4d::Vector operator-=(const float value);
            sv4d::Vector operator*=(const float value);
            sv4d::Vector operator/=(const float value);
            sv4d::Vector operator+=(const int value);
            sv4d::Vector operator-=(const int value);
            sv4d::Vector operator*=(const int value);
            sv4d::Vector operator/=(const int value);
            sv4d::Vector operator+();
            sv4d::Vector operator-();
            float operator%(const sv4d::Vector& vector);
    };

}