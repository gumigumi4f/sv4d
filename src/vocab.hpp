#pragma once

#include "options.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace se4d {

    struct SynsetData {
        SynsetData();

        std::vector<int> Noun;
        std::vector<int> Verb;
        std::vector<int> Adjective;
        std::vector<int> Adverb;
    };

    class Vocab {
        public:
            Vocab();

            int sentenceNum;
            int documentNum;
            int lemmaVocabSize;
            int wordVocabSize;
            int synsetVocabSize;

            std::unordered_map<std::string, int> lemmaVocab;
            std::unordered_map<std::string, int> wordVocab;
            std::unordered_map<std::string, int> synsetVocab;

            std::vector<int> wordFreq;

            std::vector<float> lemmaProb;

            std::vector<std::string> widx2Word;
            std::vector<std::string> sidx2Synset;
            
            std::unordered_map<int, std::vector<int>> synsetWordPair;
            std::unordered_map<int, SynsetData> widx2sidxs;

            void build(const se4d::Options& opt);
            void save(const std::string& filepath);
            void load(const std::string& filepath);
    };

}