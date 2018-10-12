#include "utils.hpp"

#include <cmath>


namespace sv4d {

    namespace utils {

        namespace string {

        }

        namespace operation {

            std::vector<float> computeSigmoidTable() {
                auto table = std::vector<float>();
                for (int i = 0; i < SigmoidTableSize + 1; ++i) {
                    float x = (float)(i * 2 * MaxSigmoid) / SigmoidTableSize - MaxSigmoid;
                    table.push_back(1.0 / (1.0 + std::exp(-x)));
                }
                return table;
            }

            float sigmoid(float x) {
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

