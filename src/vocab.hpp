#pragma once

#include "options.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace sv4d {

    enum Pos {
        Noun = 0,
        Verb = 1,
        Adjective = 2,
        Adverb = 3
    };

    struct SynsetData {
        SynsetData();

        std::vector<int> SynsetLemmaIndices[4];
        int WordLemmaIndex;
    };

    class Vocab {
        public:
            Vocab();

            int sentenceNum;
            int documentNum;
            int lemmaVocabSize;
            int synsetVocabSize;
            int wordVocabSize;

            std::unordered_map<std::string, int> lemmaVocab;
            std::unordered_map<std::string, int> synsetVocab;

            std::vector<std::string> lidx2Lemma;
            std::vector<std::string> sidx2Synset;

            std::vector<float> lemmaProb;

            std::vector<int> wordFreq;

            std::unordered_map<int, std::vector<int>> synsetDictPair;

            std::unordered_map<int, sv4d::SynsetData> widx2lidxs;
            std::unordered_map<int, int> lidx2sidx;

            void build(const sv4d::Options& opt);
            void save(const std::string& filepath);
            void load(const std::string& filepath);
    };

}