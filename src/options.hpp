#pragma once

#include <string>
#include <vector>

namespace se4d {

    class Options {
        public:
            Options();

            std::string modelDir;
            std::string synsetDataFile;
            std::string trainingCorpus;

            int epochs;
            int embeddingLayerSize;
            int minCount;
            int windowSize;
            int negativeSample;
            int dictSample;
            int maxDictPair;

            double subSamplingFactor;
            double initialLearningRate;
            double minLearningRate;
            double initialTemperature;
            double minTemperature;
            double initialBetaDict;
            double minBetaDict;
            double initialBetaReward;
            double minBetaReward;

            void parse(const std::vector<std::string>& args);
    };

}