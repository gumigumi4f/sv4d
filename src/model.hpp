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
            int fileSize;

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

            void Initialize();
            void Training();
            void TrainingThread(const int threadId);

        private:
            long trainedWordCount;

            std::chrono::system_clock::time_point startTime;

            void InitializeUnigramTable();
            void InitializeSubsamplingFactorTable();
            void InitializeFileSize();
            void ProcessBatch(const sv4d::Vector& documentVector, const sv4d::Vector& sentenceVector, const sv4d::Vector& contextVector, const int inputWidx, const int outputWidx);
    };

}