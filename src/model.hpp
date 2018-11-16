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
            int batchSize;

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
            void wordNearestNeighbour();
            void synsetNearestNeighbour();
            void saveEmbeddingInWeight(const std::string& filepath, bool binary);
            void loadEmbeddingInWeight(const std::string& filepath, bool binary);
            void saveEmbeddingOutWeight(const std::string& filepath, bool binary);
            void loadEmbeddingOutWeight(const std::string& filepath, bool binary);
            void saveSenseSelectionOutWeight(const std::string& filepath, bool binary);
            void loadSenseSelectionOutWeight(const std::string& filepath, bool binary);
            void saveSenseSelectionBiasWeight(const std::string& filepath, bool binary);
            void loadSenseSelectionBiasWeight(const std::string& filepath, bool binary);

        private:
            static const int UnigramTableSize = 1e8;

            long trainedWordCount;

            std::chrono::system_clock::time_point startTime;

            void initializeWeight();
            void initializeUnigramTable();
            void initializeSubsamplingFactorTable();
            void initializeFileSize();

            void saveMatrix(const sv4d::Matrix& matrix, const std::string& filepath, bool binary);
            void loadMatrix(sv4d::Matrix& matrix, const std::string& filepath, bool binary);
            void saveVector(const sv4d::Vector& vector, const std::string& filepath, bool binary);
            void loadVector(sv4d::Vector& vector, const std::string& filepath, bool binary);
    };

}