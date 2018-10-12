#pragma once

#include <vector>
#include <complex>

namespace sv4d {

    class Vector {
        public:
            Vector();
            Vector(int n);
            Vector(const std::vector<float>& vec);

            int col;

            std::vector<float> data;

            void setZero();
            void setRandomUniform(double min, double max);

            float sum();

            float& operator[](int idx);
            const float& operator[](int idx) const;
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
        
        private:
            static const int SigmoidTableSize = 10000;
            static const int SigmoidTableMax = 6.0;
    };

}