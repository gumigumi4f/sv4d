#include "options.hpp"
#include "vocab.hpp"
#include "model.hpp"

#include <iostream>
// #include <fenv.h>

void printUsage() {
    std::cerr
        << "usage: sv4d <command> <options>\n\n"
        << "The commands supported by sv4d are:\n\n"
        << "  training                  train a sense vector and wsd module\n"
        << "  word_nearest_neighbour    query for word nearest neighbour\n"
        << "  synset_nearest_neighbour  query for synset nearest neighbour\n"
        << std::endl;
}

void printOptionsHelp() {
    sv4d::Options options = sv4d::Options();
    std::cerr
        << "\nThe following arguments for training are optional:\n"
        << "  -model_dir                whether model should be saved [" << options.modelDir << "]\n"
        << "  -synset_data_file         model vocabulary file with dictionary pair [" << options.synsetDataFile << "]\n"
        << "  -training_corpus          training corpus file path [" << options.synsetDataFile << "]\n"
        << "  -stop_words_file          stop words file path [" << options.stopWordsFile << "]\n"
        << "  -epoch                    number of epochs [" << options.epochs << "]\n"
        << "  -embedding_layer_size     size of vectors [" << options.embeddingLayerSize << "]\n"
        << "  -min_count                minimal number of word occurences [" << options.minCount << "]\n"
        << "  -window_size              size of the context window [" << options.windowSize << "]\n"
        << "  -negative_sample          number of negatives sampled [" << options.negativeSample << "]\n"
        << "  -dict_sample              number of dictionary pairs sampled per word [" << options.dictSample << "]\n"
        << "  -max_dict_pair            number of dictionary pairs used [" << options.maxDictPair << "]\n"
        << "  -thread_num               number of threads [" << options.threadNum << "]\n"
        << "  -sub_sampling_factor      threshold for occurrence of words [" << options.subSamplingFactor << "]\n"
        << "  -initial_learning_rate    initial learning rate [" << options.initialLearningRate << "]\n"
        << "  -min_learning_rate        min learning rate [" << options.minLearningRate << "]\n"
        << "  -initial_temperature      initial softmax temperature [" << options.initialTemperature << "]\n"
        << "  -min_temperature          min softmax temperature [" << options.minTemperature << "]\n"
        << "  -initial_beta_dict        initial beta dict [" << options.initialBetaDict << "]\n"
        << "  -min_beta_dict            min beta dict [" << options.minBetaDict << "]\n"
        << "  -initial_beta_reward      initial beta reward [" << options.initialBetaReward << "]\n"
        << "  -min_beta_reward          min beta reward [" << options.minBetaReward << "]\n"
        << std::endl;
}

int main(int argc, char** argv) {
    // feenableexcept(FE_INVALID | FE_OVERFLOW);
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() <= 2) {
        printUsage();
        printOptionsHelp();
        exit(EXIT_FAILURE);
    }

    sv4d::Options opt = sv4d::Options();
    try {
        opt.parse(args);
    } catch (const std::exception& e) {
        if (e.what() == std::string("help")) {
            printUsage();
            printOptionsHelp();
            exit(EXIT_FAILURE);
        } else {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }

    std::string command(args[1]);
    if (command == "training") {
        sv4d::Vocab vocab = sv4d::Vocab();
        try {
            vocab.build(opt);
            vocab.save(opt.modelDir + "vocab.txt");
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }

        sv4d::Model model = sv4d::Model(opt, vocab);
        try {
            model.initialize();
            model.training();
            model.saveEmbeddingInWeight(opt.modelDir + "embedding_in_weight", opt.binary);
            model.saveEmbeddingOutWeight(opt.modelDir + "embedding_out_weight", opt.binary);
            model.saveSenseSelectionOutWeight(opt.modelDir + "sense_selection_out_weight", opt.binary);
            model.saveSenseSelectionBiasWeight(opt.modelDir + "sense_selection_out_bias", opt.binary);
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    } else if (command == "word_nearest_neighbour") {
        sv4d::Vocab vocab = sv4d::Vocab();
        try {
            vocab.load(opt.modelDir + "vocab.txt");
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }

        sv4d::Model model = sv4d::Model(opt, vocab);
        try {
            model.loadEmbeddingInWeight(opt.modelDir + "embedding_in_weight", opt.binary);
            model.wordNearestNeighbour();
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    } else if (command == "synset_nearest_neighbour") {
        sv4d::Vocab vocab = sv4d::Vocab();
        try {
            vocab.load(opt.modelDir + "vocab.txt");
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }

        sv4d::Model model = sv4d::Model(opt, vocab);
        try {
            model.loadEmbeddingInWeight(opt.modelDir + "embedding_in_weight", opt.binary);
            model.synsetNearestNeighbour();
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    } else {
        printUsage();
        printOptionsHelp();
    }

    return 0;
}