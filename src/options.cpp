#include "options.hpp"

#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstdlib>

namespace sv4d {

    Options::Options() {
        modelDir = "./";
        synsetDataFile = "./synset.txt";
        trainingCorpus = "./corpus.txt";

        epochs = 10;
        embeddingLayerSize = 300;
        minCount = 5;
        windowSize = 5;
        negativeSample = 5;
        dictSample = 3;
        maxDictPair = 10;
        threadNum = 1;

        subSamplingFactor = 1e-4;
        initialLearningRate = 0.025;
        minLearningRate = 0.0001;
        initialTemperature = 1.0;
        minTemperature = 0.01;
        initialBetaDict = 0.8;
        minBetaDict = 0.35;
        initialBetaReward = 0.8;
        minBetaReward = 0.35;

        binary = true;
    }

    void Options::parse(const std::vector<std::string>& args) {
        for (int i = 2; i < args.size(); i += 2) {
            if (args[i][0] != '-') {
                throw std::runtime_error("Provided argument without a dash");
            }

            try {
                if (args[i] == "-h" || args[i] == "-help") {
                    throw std::runtime_error("help");
                } else if (args[i] == "-model_dir") {
                    modelDir = std::string(args.at(i + 1));
                    if (modelDir.back() != '/') {
                        modelDir += "/";
                    } 
                } else if (args[i] == "-synset_data_file") {
                    synsetDataFile = std::string(args.at(i + 1));
                } else if (args[i] == "-training_corpus") {
                    trainingCorpus = std::string(args.at(i + 1));
                } else if (args[i] == "-epochs") {
                    epochs = std::stoi(args.at(i + 1));
                } else if (args[i] == "-embedding_layer_size") {
                    embeddingLayerSize = std::stoi(args.at(i + 1));
                } else if (args[i] == "-min_count") {
                    minCount = std::stoi(args.at(i + 1));
                } else if (args[i] == "-window_size") {
                    windowSize = std::stoi(args.at(i + 1));
                } else if (args[i] == "-negative_sample") {
                    negativeSample = std::stoi(args.at(i + 1));
                } else if (args[i] == "-dict_sample") {
                    dictSample = std::stoi(args.at(i + 1));
                } else if (args[i] == "-max_dict_pair") {
                    maxDictPair = std::stoi(args.at(i + 1));
                } else if (args[i] == "-thread_num") {
                    threadNum = std::stoi(args.at(i + 1));
                } else if (args[i] == "-sub_sampling_factor") {
                    subSamplingFactor = std::stof(args.at(i + 1));
                } else if (args[i] == "-initial_learning_rate") {
                    initialLearningRate = std::stof(args.at(i + 1));
                } else if (args[i] == "-min_learning_rate") {
                    minLearningRate = std::stof(args.at(i + 1));
                } else if (args[i] == "-initial_temperature") {
                    initialTemperature = std::stof(args.at(i + 1));
                } else if (args[i] == "-min_temperature") {
                    minTemperature = std::stof(args.at(i + 1));
                } else if (args[i] == "-initial_beta_dict") {
                    initialBetaDict = std::stof(args.at(i + 1));
                } else if (args[i] == "-min_beta_dict") {
                    minBetaDict = std::stof(args.at(i + 1));
                } else if (args[i] == "-initial_beta_reward") {
                    initialBetaReward = std::stof(args.at(i + 1));
                } else if (args[i] == "-min_beta_reward") {
                    minBetaReward = std::stof(args.at(i + 1));
                } else if (args[i] == "-binary") {
                    binary = (std::stoi(args.at(i + 1)) == 1);
                }
            } catch (std::out_of_range) {
                throw std::runtime_error(args[i] + " is missing an argument");
            }
        }
    }

}