#include "options.hpp"
#include "vocab.hpp"
#include "model.hpp"

#include <iostream>

int main(int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() <= 2) {
        std::cerr << "usage: sv4d <command> <options>" << '\n';
        exit(EXIT_FAILURE);
    }

    sv4d::Options opt = sv4d::Options();
    try {
        opt.parse(args);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        exit(EXIT_FAILURE);
    }

    std::string command(args[1]);
    if (command == "training") {
        sv4d::Vocab vocab = sv4d::Vocab();
        vocab.build(opt);
        vocab.save(opt.modelDir + "vocab.txt");
        sv4d::Model model = sv4d::Model(opt, vocab);
        model.initialize();
        model.training();
        model.saveEmbeddingInWeight(opt.modelDir + "embedding_in_weight", opt.binary);
        model.saveEmbeddingOutWeight(opt.modelDir + "embedding_out_weight", opt.binary);
        model.saveSenseSelectionOutWeight(opt.modelDir + "sense_selection_out_weight", opt.binary);
        model.saveSenseSelectionBiasWeight(opt.modelDir + "sense_selection_out_bias", opt.binary);
    } else if (command == "word_nearest_neighbour") {
        sv4d::Vocab vocab = sv4d::Vocab();
        vocab.load(opt.modelDir + "vocab.txt");
        sv4d::Model model = sv4d::Model(opt, vocab);
        model.loadEmbeddingInWeight(opt.modelDir + "embedding_in_weight", opt.binary);
        model.wordNearestNeighbour();
    } else if (command == "synset_nearest_neighbour") {
        sv4d::Vocab vocab = sv4d::Vocab();
        vocab.load(opt.modelDir + "vocab.txt");
        sv4d::Model model = sv4d::Model(opt, vocab);
        model.loadEmbeddingInWeight(opt.modelDir + "embedding_in_weight", opt.binary);
        model.synsetNearestNeighbour();
    }

    return 0;
}