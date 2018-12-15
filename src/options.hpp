#pragma once

#include <string>
#include <vector>

namespace sv4d {

    class Options {
        public:
            Options();

            std::string modelDir;
            std::string synsetDataFile;
            std::string trainingCorpus;
            std::string stopWordsFile;

            int epochs;
            int embeddingLayerSize;
            int minCount;
            int windowSize;
            int negativeSample;
            int dictSample;
            int maxDictPair;
            int threadNum;
            int batchSize;

            float subSamplingFactor;
            float initialLearningRate;
            float minLearningRate;
            float initialTemperature;
            float minTemperature;
            float initialBetaDict;
            float minBetaDict;
            float initialBetaReward;
            float minBetaReward;

            bool binary;

            void parse(const std::vector<std::string>& args);
    };

}