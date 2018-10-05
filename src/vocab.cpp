#include "vocab.hpp"

#include "options.hpp"
#include "utils.hpp"
#include <unordered_map>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <vector>
#include <cstdlib>

namespace se4d {

    SynsetData::SynsetData() {
        Noun = std::vector<int>();
        Verb = std::vector<int>();
        Adjective = std::vector<int>();
        Adverb = std::vector<int>();
    }

    Vocab::Vocab() {
        sentenceNum = 0;
        documentNum = 0;
        lemmaVocabSize = 0;
        wordVocabSize = 0;
        synsetVocabSize = 0;

        lemmaVocab = std::unordered_map<std::string, int>();
        wordVocab = std::unordered_map<std::string, int>();
        synsetVocab = std::unordered_map<std::string, int>();

        wordFreq = std::vector<int>();

        lemmaProb = std::vector<float>();

        widx2Word = std::vector<std::string>();
        sidx2Synset = std::vector<std::string>();

        synsetWordPair = std::unordered_map<int, std::vector<int>>();
        widx2sidxs = std::unordered_map<int, se4d::SynsetData>();
    }

    void Vocab::build(const se4d::Options& opt) {
        std::string linebuf;
        std::ifstream fin;

        auto wordStats = std::unordered_map<std::string, int>();
        fin = std::ifstream(opt.trainingCorpus);
        while (getline(fin, linebuf)) {
            linebuf = se4d::utils::string::trim(linebuf);
            if (linebuf == "<doc>") {
                documentNum += 1;
                continue;
            } else if (linebuf == "</doc>") {
                continue;
            } else if (linebuf == "") {
                continue;
            }

            for (auto word : se4d::utils::string::split(linebuf, ' ')) {
                ++wordStats[word];
            }

            sentenceNum += 1;
            if (sentenceNum % 1000000 == 0) {
                std::cout << sentenceNum << std::endl;
            }
        }

        auto sortedWordStats = std::vector<std::pair<std::string, int>>(wordStats.begin(), wordStats.end());
        std::sort(sortedWordStats.begin(), sortedWordStats.end(), [](const std::pair<std::string, int> & a, const std::pair<std::string, int> & b) -> bool { return a.second > b.second; });
        for (auto pair : sortedWordStats) {
            auto word = pair.first;
            auto freq = pair.second;

            if (freq < opt.minCount) {
                break;
            }

            auto lidx = lemmaVocab.size();
            auto widx = wordVocab.size();

            lemmaVocab[word + "|*|*"] = lidx;
            lemmaProb.push_back(1.0f);
            wordVocab[word] = widx;
            wordFreq.push_back(freq);
            widx2Word.push_back(word);
        }

        fin = std::ifstream(opt.synsetDataFile);
        while (getline(fin, linebuf)) {
            linebuf = se4d::utils::string::trim(linebuf);
            auto data = se4d::utils::string::split(linebuf, ' ');
            auto lemmaData = se4d::utils::string::split(data[0], '|');

            auto word = lemmaData[0];

            if (wordVocab.find(word) != wordVocab.end()) {
                continue;
            }

            auto lidx = lemmaVocab.size();
            auto widx = wordVocab.size();

            lemmaVocab[word + "|*|*"] = lidx;
            lemmaProb.push_back(1.0f);
            wordVocab[word] = widx;
            wordFreq.push_back(0);
            widx2Word.push_back(word);
        }

        fin = std::ifstream(opt.synsetDataFile);
        while (getline(fin, linebuf)) {
            linebuf = se4d::utils::string::trim(linebuf);
            auto data = se4d::utils::string::split(linebuf, ' ');
            if (lemmaVocab.find(data[0]) != lemmaVocab.end()) {
                continue;
            }

            auto lemmaData = se4d::utils::string::split(data[0], '|');
            auto word = lemmaData[0];
            auto pos = lemmaData[1];
            auto synset = lemmaData[2];
            auto widx = wordVocab[word];

            int sidx = 0;
            if (synsetVocab.find(lemmaData[2]) == synsetVocab.end()) {
                sidx = synsetVocab.size();
                synsetVocab[synset] = sidx;
                sidx2Synset.push_back(synset);

                auto wordpair = std::vector<int>();
                for (auto word : se4d::utils::string::split(data[2], ',')) {
                    if (wordVocab.find(word) == wordVocab.end()) {
                        continue;
                    }
                    wordpair.push_back(wordVocab[word]);
                    if (wordpair.size() >= opt.maxDictPair) {
                        break;
                    } 
                }
                synsetWordPair[sidx] = wordpair;
            } else {
                sidx = synsetVocab[synset];
            }

            if (widx2sidxs.find(widx) == widx2sidxs.end()) {
                widx2sidxs[widx] = se4d::SynsetData();
            }

            if (pos == "n") {
                widx2sidxs[widx].Noun.push_back(sidx);
            } else if (pos == "v") {
                widx2sidxs[widx].Verb.push_back(sidx);
            } else if (pos == "a") {
                widx2sidxs[widx].Adjective.push_back(sidx);
            } else if (pos == "r") {
                widx2sidxs[widx].Adverb.push_back(sidx);
            }

            auto lidx = lemmaVocab.size();
            lemmaVocab[word + "|" + pos + "|" + synset] = lidx;
            lemmaProb.push_back(std::stof(data[1]));
        }

        // For Debug
        std::cout << lemmaVocab.size() << " " << lemmaProb.size() << "\n";
        std::cout << wordVocab.size() << " " << wordFreq.size() << " " << widx2Word.size() << " " << widx2sidxs.size() << "\n";
        std::cout << synsetVocab.size() << " " << sidx2Synset.size() << " " << synsetWordPair.size() << "\n";
        //std::string s;
        //std::cin >> s; 
    }

    void Vocab::save(const std::string& filepath) {
        std::ofstream fout = std::ofstream(filepath);

        fout << lemmaVocab.size() << " " << wordVocab.size() << " " << synsetVocab.size() << "\n";

        for (auto pair : lemmaVocab) {
            auto lemma = pair.first;
            auto lidx = pair.second;

            auto lemmaData = se4d::utils::string::split(lemma, '|');
            auto word = lemmaData[0];
            auto synset = lemmaData[2];

            auto widx = wordVocab[word];
            
            auto sidx = synsetVocab[synset];
            std::string wordpair;
            if (synsetWordPair.find(sidx) == synsetWordPair.end()) {
                wordpair = "-1";
            } else {
                wordpair = se4d::utils::string::join(se4d::utils::string::intvec_to_strvec(synsetWordPair[sidx]), ',');
            }

            fout << lemma << " " << lidx << " " << lemmaProb[lidx] << " " << widx << " " << wordFreq[widx] << " " << sidx << " " << wordpair << "\n";
        }

        fout.close();
    }

    void Vocab::load(const std::string& filepath) {
        std::string linebuf;
        auto fin = std::ifstream(filepath);
        getline(fin, linebuf);

        auto sizes = se4d::utils::string::split(se4d::utils::string::trim(linebuf), ' ');
        auto lemmaVocabSize = std::stoi(sizes[0]);
        auto wordVocabSize = std::stoi(sizes[1]);
        auto synsetVocabSize = std::stoi(sizes[2]);
        lemmaProb.resize(lemmaVocabSize);
        widx2Word.resize(wordVocabSize);
        wordFreq.resize(wordVocabSize);
        sidx2Synset.resize(synsetVocabSize);

        while (getline(fin, linebuf)) {
            linebuf = se4d::utils::string::trim(linebuf);
            auto data = se4d::utils::string::split(linebuf, ' ');
            auto lemmaData = se4d::utils::string::split(data[0], '|');
            auto word = lemmaData[0];
            auto pos = lemmaData[1];
            auto synset = lemmaData[2];

            auto lidx = lemmaVocab.size();
            lemmaVocab[word + "|" + pos + "|" + synset] = lidx;
            lemmaProb[lidx] = std::stof(data[2]);

            if (wordVocab.find(word) == wordVocab.end()) {
                auto widx = std::stoi(data[3]);
                wordVocab[word] = widx;
                widx2Word[widx] = word;
                wordFreq[widx] = std::stoi(data[4]);
            }

            if (synset != "*" && synsetVocab.find(synset) == synsetVocab.end()) {
                auto sidx = std::stoi(data[5]);
                synsetVocab[synset] = sidx;
                sidx2Synset[sidx] = synset;
                if (data[6] != "-1") {
                    synsetWordPair[sidx] = se4d::utils::string::strvec_to_intvec(se4d::utils::string::split(data[6], ','));
                }
            }

            if (synset != "*" && pos != "*") {
                auto widx = wordVocab[word];
                auto sidx = synsetVocab[synset];

                if (widx2sidxs.find(widx) == widx2sidxs.end()) {
                    widx2sidxs[widx] = se4d::SynsetData();
                }

                if (pos == "n") {
                    widx2sidxs[widx].Noun.push_back(sidx);
                } else if (pos == "v") {
                    widx2sidxs[widx].Verb.push_back(sidx);
                } else if (pos == "a") {
                    widx2sidxs[widx].Adjective.push_back(sidx);
                } else if (pos == "r") {
                    widx2sidxs[widx].Adverb.push_back(sidx);
                }
            }
            
        }

        // For Debug
        std::cout << lemmaVocab.size() << " " << lemmaProb.size() << "\n";
        std::cout << wordVocab.size() << " " << wordFreq.size() << " " << widx2Word.size() << " " << widx2sidxs.size() << "\n";
        std::cout << synsetVocab.size() << " " << sidx2Synset.size() << " " << synsetWordPair.size() << "\n";
    }

}