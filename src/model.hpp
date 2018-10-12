#pragma once

#include "options.hpp"
#include "vocab.hpp"
#include "matrix.hpp"
#include "vector.hpp"
#include <string>
#include <vector>
#include <chrono>
#include <cmath>

namespace sv4d {

    class Model {
        public:
            Model(const sv4d::Options& opt, const sv4d::Vocab& v);

            const int UnigramTableSize = 1e8;

            sv4d::Vocab vocab;

            std::string trainingCorpus;

            int epochs;
            int embeddingLayerSize;
            int windowSize;
            int negativeSample;
            int dictSample;
            int maxDictPair;
            int threadNum;

            long fileSize;

            double subSamplingFactor;
            double initialLearningRate;
            double minLearningRate;
            double initialTemperature;
            double minTemperature;
            double initialBetaDict;
            double minBetaDict;
            double initialBetaReward;
            double minBetaReward;

            sv4d::Matrix senseSelectionOutWeight;
            sv4d::Vector senseSelectionOutBias;
            sv4d::Matrix embeddingInWeight;
            sv4d::Matrix embeddingOutWeight;

            std::vector<int> unigramTable;
            std::vector<double> subsamplingFactorTable;
            std::vector<float> sigmoidTable;

            void initialize();
            void training();
            void trainingThread(const int threadId);
            void saveEmbeddingInWeight(const std::string& filepath);
            void loadEmbeddingInWeight(const std::string& filepath);
            void saveEmbeddingOutWeight(const std::string& filepath);
            void loadEmbeddingOutWeight(const std::string& filepath);
            void saveSenseSelectionOutWeight(const std::string& filepath);
            void loadSenseSelectionOutWeight(const std::string& filepath);
            void saveSenseSelectionBiasWeight(const std::string& filepath);
            void loadSenseSelectionBiasWeight(const std::string& filepath);

        private:
            static const int SigmoidTableSize = 1024;
            static const int MaxSigmoid = 8;

            long trainedWordCount;

            std::chrono::system_clock::time_point startTime;

            void initializeUnigramTable();
            void initializeSubsamplingFactorTable();
            void initializeSigmoidTable();
            void initializeFileSize();

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

            inline std::vector<float> sigmoid(const std::vector<float> x) {
                auto output = std::vector<float>();
                for (int i = 0; i < x.size(); ++i) {
                    output.push_back(sigmoid(x[i]));
                }
                return output;
            }

            inline std::vector<float> softmax(const std::vector<float>& logits, float temperature) {
                float max = 0.0;
                auto output = std::vector<float>();
                for (int i = 0; i < logits.size(); ++i) {
                    auto temped_logit = logits[i] / temperature;
                    output.push_back(temped_logit);
                    max = std::max(max, temped_logit);
                }
                float sum = 0.0;
                for (int i = 0; i < logits.size(); ++i) {
                    output[i] = std::exp(output[i] - max);
                    sum += output[i];
                }
                for (int i = 0; i < logits.size(); ++i) {
                    output[i] /= sum;
                }
                return output;
            }

            inline std::vector<float> clipByValue(const std::vector<float>& x, float min, float max) {
                auto output = std::vector<float>();
                for (int i = 0; i < x.size(); ++i) {
                    if (x[i] > max) {
                        output.push_back(max);
                    } else if (x[i] < min) {
                        output.push_back(min);
                    } else {
                        output.push_back(x[i]);
                    }
                }
                return output;
            }
    };

}