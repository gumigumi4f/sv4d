#include "vocab.hpp"

#include "options.hpp"
#include "utils.hpp"
#include <unordered_map>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <vector>
#include <cstdlib>

namespace sv4d {

    SynsetData::SynsetData() {
        for (int i = 0; i < 4; i++) {
            SynsetLemmaIndices[i] = std::vector<int>();
        }
        WordLemmaIndex = 0;
    }

    Vocab::Vocab() {
        sentenceNum = 0;
        documentNum = 0;
        lemmaVocabSize = 0;
        synsetVocabSize = 0;
        wordVocabSize = 0;

        lemmaVocab = std::unordered_map<std::string, int>();
        synsetVocab = std::unordered_map<std::string, int>();

        lidx2Lemma = std::vector<std::string>();
        sidx2Synset = std::vector<std::string>();

        lemmaProb = std::vector<float>();

        wordFreq = std::vector<int>();

        synsetDictPair = std::unordered_map<int, std::vector<int>>();

        widx2lidxs = std::unordered_map<int, sv4d::SynsetData>();
        lidx2sidx = std::unordered_map<int, int>();
    }

    void Vocab::build(const sv4d::Options& opt) {
        std::string linebuf;
        std::ifstream fin;

        auto wordStats = std::unordered_map<std::string, int>();
        fin = std::ifstream(opt.trainingCorpus);
        while (getline(fin, linebuf)) {
            linebuf = sv4d::utils::string::trim(linebuf);
            if (linebuf == "<doc>") {
                documentNum += 1;
                continue;
            } else if (linebuf == "</doc>") {
                continue;
            } else if (linebuf == "") {
                continue;
            }

            for (auto word : sv4d::utils::string::split(linebuf, ' ')) {
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
            auto sidx = synsetVocab.size();

            lemmaVocab[word + "|*|*"] = lidx;
            lidx2Lemma.push_back(word + "|*|*");
            lemmaProb.push_back(1.0f);
            synsetVocab[word] = sidx;
            sidx2Synset.push_back(word);
            wordFreq.push_back(freq);
            widx2lidxs[sidx] = sv4d::SynsetData();
            widx2lidxs[sidx].WordLemmaIndex = sidx;
            lidx2sidx[lidx] = sidx;
        }

        fin = std::ifstream(opt.synsetDataFile);
        while (getline(fin, linebuf)) {
            linebuf = sv4d::utils::string::trim(linebuf);
            auto data = sv4d::utils::string::split(linebuf, ' ');
            auto lemmaData = sv4d::utils::string::split(data[0], '|');

            auto word = lemmaData[0];

            if (synsetVocab.find(word) != synsetVocab.end()) {
                continue;
            }

            auto lidx = lemmaVocab.size();
            auto sidx = synsetVocab.size();

            lemmaVocab[word + "|*|*"] = lidx;
            lidx2Lemma.push_back(word + "|*|*");
            lemmaProb.push_back(1.0f);
            synsetVocab[word] = sidx;
            sidx2Synset.push_back(word);
            wordFreq.push_back(0);
            widx2lidxs[sidx] = sv4d::SynsetData();
            widx2lidxs[sidx].WordLemmaIndex = sidx;
            lidx2sidx[lidx] = sidx;
        }

        wordVocabSize = widx2lidxs.size();

        fin = std::ifstream(opt.synsetDataFile);
        while (getline(fin, linebuf)) {
            linebuf = sv4d::utils::string::trim(linebuf);
            auto data = sv4d::utils::string::split(linebuf, ' ');
            if (lemmaVocab.find(data[0]) != lemmaVocab.end()) {
                continue;
            }

            auto lemmaData = sv4d::utils::string::split(data[0], '|');
            auto word = lemmaData[0];
            auto pos = lemmaData[1];
            auto synset = lemmaData[2];

            if (synsetVocab.find(word) == synsetVocab.end()) {
                continue;
            }

            auto lidx = lemmaVocab.size();
            lemmaVocab[word + "|" + pos + "|" + synset] = lidx;
            lidx2Lemma.push_back(word + "|" + pos + "|" + synset);
            lemmaProb.push_back(std::stof(data[1]));

            if (synsetVocab.find(synset) == synsetVocab.end()) {
                auto sidx = synsetVocab.size();

                synsetVocab[synset] = sidx;
                sidx2Synset.push_back(synset);

                auto dictPair = std::vector<int>();
                for (auto word : sv4d::utils::string::split(data[2], ',')) {
                    if (synsetVocab.find(word) == synsetVocab.end()) {
                        continue;
                    }
                    dictPair.push_back(synsetVocab[word]);
                    if (dictPair.size() >= opt.maxDictPair) {
                        break;
                    } 
                }
                synsetDictPair[sidx] = dictPair;
            }

            auto widx = synsetVocab[word];
            auto sidx = synsetVocab[synset];
            lidx2sidx[lidx] = sidx;
            if (pos == "n") {
                widx2lidxs[widx].SynsetLemmaIndices[sv4d::Pos::Noun].push_back(sidx);
            } else if (pos == "v") {
                widx2lidxs[widx].SynsetLemmaIndices[sv4d::Pos::Verb].push_back(sidx);
            } else if (pos == "a") {
                widx2lidxs[widx].SynsetLemmaIndices[sv4d::Pos::Adjective].push_back(sidx);
            } else if (pos == "r") {
                widx2lidxs[widx].SynsetLemmaIndices[sv4d::Pos::Adverb].push_back(sidx);
            }
        }

        lemmaVocabSize = lemmaVocab.size();
        synsetVocabSize = synsetVocab.size();
        // For Debug
        std::cout << lemmaVocab.size() << " " << lidx2Lemma.size() << " " << lemmaProb.size() << " " << lidx2sidx.size() << "\n";
        std::cout << wordVocabSize << " " << wordFreq.size() << " " << widx2lidxs.size() << "\n";
        std::cout << synsetVocab.size() << " " << sidx2Synset.size() << "\n";
        //std::string s;
        //std::cin >> s; 
    }

    void Vocab::save(const std::string& filepath) {
        std::ofstream fout = std::ofstream(filepath);

        fout << lemmaVocabSize << " " << synsetVocabSize << " " << wordVocabSize << "\n";

        for (auto pair : lemmaVocab) {
            auto lemma = pair.first;
            auto lidx = pair.second;

            auto lemmaData = sv4d::utils::string::split(lemma, '|');
            auto word = lemmaData[0];
            auto synset = lemmaData[2];

            auto widx = synsetVocab[word];
            auto sidx = synsetVocab[synset];

            std::string dictPair;
            if (synsetDictPair.find(sidx) == synsetDictPair.end()) {
                dictPair = "-1";
            } else {
                dictPair = sv4d::utils::string::join(sv4d::utils::string::intvec_to_strvec(synsetDictPair[sidx]), ',');
            }

            fout << lemma << " " << lidx << " " << lemmaProb[lidx] << " " << widx << " " << wordFreq[widx] << " " << sidx << " " << dictPair << "\n";
        }

        fout.close();
    }

    void Vocab::load(const std::string& filepath) {
        std::string linebuf;
        auto fin = std::ifstream(filepath);

        getline(fin, linebuf);
        auto sizes = sv4d::utils::string::split(sv4d::utils::string::trim(linebuf), ' ');
        auto lemmaVocabSize = std::stoi(sizes[0]);
        auto synsetVocabSize = std::stoi(sizes[1]);
        auto wordVocabSize = std::stoi(sizes[2]);
        lidx2Lemma.resize(lemmaVocabSize);
        lemmaProb.resize(lemmaVocabSize);
        sidx2Synset.resize(synsetVocabSize);
        wordFreq.resize(wordVocabSize);

        while (getline(fin, linebuf)) {
            linebuf = sv4d::utils::string::trim(linebuf);
            auto data = sv4d::utils::string::split(linebuf, ' ');
            auto lemmaData = sv4d::utils::string::split(data[0], '|');
            auto word = lemmaData[0];
            auto pos = lemmaData[1];
            auto synset = lemmaData[2];

            auto lidx = std::stoi(data[1]);
            auto sidx = std::stoi(data[5]);
            auto widx = std::stoi(data[3]);
            lemmaVocab[word + "|" + pos + "|" + synset] = lidx;
            lidx2Lemma[lidx] = word + "|" + pos + "|" + synset;
            lemmaProb[lidx] = std::stof(data[2]);
            lidx2sidx[lidx] = sidx;

            //fout << lemma << " " << lidx << " " << lemmaProb[lidx] << " " << widx << " " << wordFreq[widx] << " " << sidx << " " << dictPair << "\n";
            if (synsetVocab.find(word) == synsetVocab.end()) {
                synsetVocab[word] = widx;
                sidx2Synset[widx] = word;
                wordFreq[widx] = std::stoi(data[4]);
                widx2lidxs[widx] = sv4d::SynsetData();
                widx2lidxs[widx].WordLemmaIndex = widx;
            }
            
            if (synset != "*") {
                if (synsetVocab.find(synset) == synsetVocab.end()) {

                    synsetVocab[synset] = sidx;
                    sidx2Synset[sidx] = synset;
                    if (data[6] != "-1") {
                        synsetDictPair[sidx] = sv4d::utils::string::strvec_to_intvec(sv4d::utils::string::split(data[6], ','));
                    }
                }
                
                if (pos == "n") {
                    widx2lidxs[widx].SynsetLemmaIndices[sv4d::Pos::Noun].push_back(sidx);
                } else if (pos == "v") {
                    widx2lidxs[widx].SynsetLemmaIndices[sv4d::Pos::Verb].push_back(sidx);
                } else if (pos == "a") {
                    widx2lidxs[widx].SynsetLemmaIndices[sv4d::Pos::Adjective].push_back(sidx);
                } else if (pos == "r") {
                    widx2lidxs[widx].SynsetLemmaIndices[sv4d::Pos::Adverb].push_back(sidx);
                }
            }
        }

        // For Debug
        std::cout << lemmaVocab.size() << " " << lidx2Lemma.size() << " " << lemmaProb.size() << " " << lidx2sidx.size() << "\n";
        std::cout << wordVocabSize << " " << wordFreq.size() << " " << widx2lidxs.size() << "\n";
        std::cout << synsetVocab.size() << " " << sidx2Synset.size() << "\n";
    }

}