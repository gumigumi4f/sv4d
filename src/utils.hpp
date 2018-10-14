#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <cctype>
#include <sstream>

namespace sv4d {

    namespace utils {

        namespace string {

            inline std::string trim(const std::string& s) {
                auto left = std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace)));
                auto right = std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace)));
                return (left < right.base()) ? std::string(left, right.base()) : std::string();
            }

            inline std::vector<std::string> split(const std::string& input, char delimiter) {
                std::istringstream stream(input);
                std::string field;
                std::vector<std::string> result;
                while (std::getline(stream, field, delimiter)) {
                    result.push_back(field);
                }
                return result;
            }

            inline std::vector<int> strvec_to_intvec(const std::vector<std::string>& strvec) {
                auto intvec = std::vector<int>();
                for (auto str : strvec) {
                    intvec.push_back(std::stoi(str));
                }
                return intvec;
            }

            inline std::vector<std::string> intvec_to_strvec(const std::vector<int>& intvec) {
                auto strvec = std::vector<std::string>();
                for (auto num : intvec) {
                    strvec.push_back(std::to_string(num));
                }
                return strvec;
            }

            inline std::vector<float> strvec_to_floatvec(const std::vector<std::string>& strvec) {
                auto intvec = std::vector<float>();
                for (auto str : strvec) {
                    intvec.push_back(std::stof(str));
                }
                return intvec;
            }

            inline std::vector<std::string> floatvec_to_strvec(const std::vector<float>& intvec) {
                auto strvec = std::vector<std::string>();
                for (auto num : intvec) {
                    strvec.push_back(std::to_string(num));
                }
                return strvec;
            }

            inline std::string join(const std::vector<std::string>& v, char delimiter) {
                std::string s;
                if (!v.empty()) {
                    s += v[0];
                    for (decltype(v.size()) i = 1, c = v.size(); i < c; ++i) {
                        if (delimiter) {
                            s += delimiter;
                        }
                        s += v[i];
                    }
                }
                return s;
            }

        }

        namespace operation {

            const int SigmoidTableSize = 1024;
            const int MaxSigmoid = 8.0;

            std::vector<float> computeSigmoidTable();

            const std::vector<float> sigmoidTable = computeSigmoidTable();

            inline float sigmoid(float x) {
                if (x < -MaxSigmoid) {
                    return 0.0;
                } else if (x > MaxSigmoid) {
                    return 1.0;
                } else {
                    int i = int((x + MaxSigmoid) * SigmoidTableSize / MaxSigmoid / 2);
                    return sigmoidTable[i];
                }
            }
        }

    }

}