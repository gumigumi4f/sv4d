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

    }

}