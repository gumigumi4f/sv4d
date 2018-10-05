#include <iostream>

#include "options.hpp"
#include "vocab.hpp"

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
        // training();
        vocab.save(opt.modelDir + "vocab.txt");
        sv4d::Vocab vocab2 = sv4d::Vocab();
        vocab2.load(opt.modelDir + "vocab.txt");
        // save_vector();
    } else if (command == "synset_nearest_neighbour") {
        // load_vocab();
        // load_vector();
        // synset_nearest_neighbour();
    } else if (command == "word_sense_disambiguation") {
        // load_vocab();
        // load_vector();
        // word_sense_disambiguation();
    }

    return 0;
}