#pragma once

#include "options.hpp"
#include "vocab.hpp"
#include "matrix.hpp"
#include "vector.hpp"
#include <string>
#include <vector>
#include <chrono>

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
            static const int SigmoidTableSize = 10000;
            static constexpr float MaxSigmoid = 10.0f;

            long trainedWordCount;

            std::chrono::system_clock::time_point startTime;

            void initializeUnigramTable();
            void initializeSubsamplingFactorTable();
            void initializeSigmoidTable();
            void initializeFileSize();

            inline float Sigmoid(float f) {
                return sigmoidTable[(int)((f + MaxSigmoid) * (SigmoidTableSize / MaxSigmoid / 2))];
            }
    };

}