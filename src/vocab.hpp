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
        Adverb = 3,
        Other = 4,
    };

    struct SynsetData {
        SynsetData();

        std::vector<int> synsetLemmaIndices[4];
        std::vector<int> validPos;
        int wordLemmaIndex;
    };

    struct SynsetDictPair {
        SynsetDictPair() : dictPair() {};

        std::vector<int> dictPair;
    };

    class Vocab {
        public:
            Vocab();

            int lemmaVocabSize;
            int synsetVocabSize;
            int wordVocabSize;

            long totalWordsNum;
            long totalSentenceNum;
            long totalDocumentNum;

            std::unordered_map<std::string, int> lemmaVocab;
            std::unordered_map<std::string, int> synsetVocab;

            std::vector<std::string> lidx2Lemma;
            std::vector<std::string> sidx2Synset;

            std::vector<sv4d::SynsetData> widx2lidxs;
            std::vector<int> lidx2sidx;

            std::vector<float> lemmaProb;
            std::vector<int> wordFreq;
            std::unordered_map<int, sv4d::SynsetDictPair> synsetDictPair;

            void build(const sv4d::Options& opt);
            void save(const std::string& filepath);
            void load(const std::string& filepath);
    };

}