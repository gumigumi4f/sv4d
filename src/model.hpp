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

            float subSamplingFactor;
            float initialLearningRate;
            float minLearningRate;
            float initialTemperature;
            float minTemperature;
            float initialBetaDict;
            float minBetaDict;
            float initialBetaReward;
            float minBetaReward;

            sv4d::Matrix senseSelectionOutWeight;
            sv4d::Vector senseSelectionOutBias;
            sv4d::Matrix embeddingInWeight;
            sv4d::Matrix embeddingOutWeight;

            std::vector<int> unigramTable;
            std::vector<float> subsamplingFactorTable;

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
            static const int UnigramTableSize = 1e8;

            long trainedWordCount;

            std::chrono::system_clock::time_point startTime;

            void initializeUnigramTable();
            void initializeSubsamplingFactorTable();
            void initializeFileSize();
    };

}