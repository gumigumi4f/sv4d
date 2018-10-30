#include "vocab.hpp"

#include "options.hpp"
#include "utils.hpp"
#include <unordered_map>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <numeric>
#include <stdio.h>

namespace sv4d {

    SynsetData::SynsetData() {
        for (int i = 0; i < 4; i++) {
            synsetLemmaIndices[i] = std::vector<int>();
        }
        validPos = std::vector<int>();
        wordLemmaIndex = 0;
    }

    Vocab::Vocab() {
        lemmaVocabSize = 0;
        synsetVocabSize = 0;
        wordVocabSize = 0;

        totalWordsNum = 0;
        totalSentenceNum = 0;
        totalDocumentNum = 0;

        lemmaVocab = std::unordered_map<std::string, int>();
        synsetVocab = std::unordered_map<std::string, int>();

        lidx2Lemma = std::vector<std::string>();
        sidx2Synset = std::vector<std::string>();

        widx2lidxs = std::vector<sv4d::SynsetData>();
        lidx2sidx = std::vector<int>();

        lemmaProb = std::vector<float>();
        wordFreq = std::vector<int>();
        synsetDictPair = std::unordered_map<int, sv4d::SynsetDictPair>();
    }

    void Vocab::build(const sv4d::Options& opt) {
        std::string linebuf;

        auto wordStats = std::unordered_map<std::string, int>();
        
        std::ifstream corpusfin(opt.trainingCorpus);
        if (corpusfin.fail()) {
            throw std::runtime_error("Cannot open training corpus file");
        }
        while (std::getline(corpusfin, linebuf)) {
            linebuf = sv4d::utils::string::trim(linebuf);
            if (linebuf == "<doc>") {
                totalDocumentNum += 1;
                continue;
            } else if (linebuf == "</doc>") {
                continue;
            } else if (linebuf == "") {
                continue;
            }

            for (auto word : sv4d::utils::string::split(linebuf, ' ')) {
                if (wordStats.find(word) == wordStats.end()) {
                    wordStats[word] = 0;
                }
                ++wordStats[word];
            }

            totalSentenceNum += 1;
            if (totalSentenceNum % 10000 == 0) {
                printf("%cReading Line: %ldk  ", 13, totalSentenceNum / 1000);
                fflush(stdout);
            }
        }

        auto sortedWordStats = std::vector<std::pair<std::string, int>>(wordStats.begin(), wordStats.end());
        std::sort(sortedWordStats.begin(), sortedWordStats.end(), [](const std::pair<std::string, int> & a, const std::pair<std::string, int> & b) -> bool { return a.second > b.second; });
        for (auto& pair : sortedWordStats) {
            auto word = pair.first;
            int freq = pair.second;

            if (freq < opt.minCount) {
                break;
            }

            int lidx = lemmaVocab.size();
            int sidx = synsetVocab.size();

            lemmaVocab[word + "|*|*"] = lidx;
            lidx2Lemma.push_back(word + "|*|*");
            lemmaProb.push_back(1.0f);
            synsetVocab[word] = sidx;
            sidx2Synset.push_back(word);
            wordFreq.push_back(freq);
            widx2lidxs.push_back(sv4d::SynsetData());
            widx2lidxs[sidx].wordLemmaIndex = sidx;
            lidx2sidx.push_back(sidx);
        }

        printf("\n");
        printf("SentenceNum: %ld  DocumentNum: %ld  \n", totalSentenceNum, totalDocumentNum);
        printf("Reading sense file:  \n");

        std::ifstream synsetfin(opt.synsetDataFile);
        if (synsetfin.fail()) {
            throw std::runtime_error("Cannot open synset data file");
        }
        while (std::getline(synsetfin, linebuf)) {
            linebuf = sv4d::utils::string::trim(linebuf);
            auto data = sv4d::utils::string::split(linebuf, ' ');
            auto lemmaData = sv4d::utils::string::split(data[0], '|');

            auto word = lemmaData[0];

            if (synsetVocab.find(word) != synsetVocab.end()) {
                continue;
            }

            int lidx = lemmaVocab.size();
            int sidx = synsetVocab.size();

            lemmaVocab[word + "|*|*"] = lidx;
            lidx2Lemma.push_back(word + "|*|*");
            lemmaProb.push_back(1.0f);
            synsetVocab[word] = sidx;
            sidx2Synset.push_back(word);
            wordFreq.push_back(0);
            widx2lidxs.push_back(sv4d::SynsetData());
            widx2lidxs[sidx].wordLemmaIndex = sidx;
            lidx2sidx.push_back(sidx);
        }

        wordVocabSize = widx2lidxs.size();

        synsetfin.clear();
        synsetfin.seekg(0, synsetfin.beg);
        while (std::getline(synsetfin, linebuf)) {
            linebuf = sv4d::utils::string::trim(linebuf);
            auto data = sv4d::utils::string::split(linebuf, ' ');
            if (lemmaVocab.find(data[0]) != lemmaVocab.end()) {
                continue;
            }

            auto lemmaData = sv4d::utils::string::split(data[0], '|');
            auto word = lemmaData[0];
            auto pos = lemmaData[1];
            auto synset = lemmaData[2];

            if (pos == "*" || synset == "*") {
                continue;
            }

            if (synsetVocab.find(word) == synsetVocab.end()) {
                continue;
            }

            int lidx = lemmaVocab.size();
            lemmaVocab[word + "|" + pos + "|" + synset] = lidx;
            lidx2Lemma.push_back(word + "|" + pos + "|" + synset);
            lemmaProb.push_back(std::stof(data[1]));

            if (synsetVocab.find(synset) == synsetVocab.end()) {
                int sidx = synsetVocab.size();

                synsetVocab[synset] = sidx;
                sidx2Synset.push_back(synset);
                
                if (data.size() >= 3) {
                    auto dictPair = sv4d::SynsetDictPair();
                    for (auto word : sv4d::utils::string::split(data[2], ',')) {
                        if (synsetVocab.find(word) == synsetVocab.end()) {
                            continue;
                        }
                        dictPair.dictPair.push_back(synsetVocab[word]);
                        if (dictPair.dictPair.size() >= opt.maxDictPair) {
                            break;
                        } 
                    }
                    synsetDictPair[sidx] = dictPair;
                } else {
                    synsetDictPair[sidx] = sv4d::SynsetDictPair();
                }
            }

            int widx = synsetVocab[word];
            int sidx = synsetVocab[synset];
            lidx2sidx.push_back(sidx);
            if (pos == "n") {
                widx2lidxs[widx].synsetLemmaIndices[sv4d::Pos::Noun].push_back(lidx);
                if (std::find(widx2lidxs[widx].validPos.begin(), widx2lidxs[widx].validPos.end(), (int)sv4d::Pos::Noun) == widx2lidxs[widx].validPos.end()) {
                    widx2lidxs[widx].validPos.push_back(sv4d::Pos::Noun);
                }
            } else if (pos == "v") {
                widx2lidxs[widx].synsetLemmaIndices[sv4d::Pos::Verb].push_back(lidx);
                if (std::find(widx2lidxs[widx].validPos.begin(), widx2lidxs[widx].validPos.end(), (int)sv4d::Pos::Verb) == widx2lidxs[widx].validPos.end()) {
                    widx2lidxs[widx].validPos.push_back(sv4d::Pos::Verb);
                }
            } else if (pos == "a") {
                widx2lidxs[widx].synsetLemmaIndices[sv4d::Pos::Adjective].push_back(lidx);
                if (std::find(widx2lidxs[widx].validPos.begin(), widx2lidxs[widx].validPos.end(), (int)sv4d::Pos::Adjective) == widx2lidxs[widx].validPos.end()) {
                    widx2lidxs[widx].validPos.push_back(sv4d::Pos::Adjective);
                }
            } else if (pos == "r") {
                widx2lidxs[widx].synsetLemmaIndices[sv4d::Pos::Adverb].push_back(lidx);
                if (std::find(widx2lidxs[widx].validPos.begin(), widx2lidxs[widx].validPos.end(), (int)sv4d::Pos::Adverb) == widx2lidxs[widx].validPos.end()) {
                    widx2lidxs[widx].validPos.push_back(sv4d::Pos::Adverb);
                }
            }
        }

        for (auto& synsetData : widx2lidxs) {
            for (int pos : synsetData.validPos) {
                std::sort(synsetData.synsetLemmaIndices[pos].begin(), synsetData.synsetLemmaIndices[pos].end());
            }
        }

        lemmaVocabSize = lemmaVocab.size();
        synsetVocabSize = synsetVocab.size();
        totalWordsNum = std::accumulate(wordFreq.begin(), wordFreq.end(), 0);

        printf("LemmaVocabSize: %d  SynsetVocabSize: %d  WordVocabSize: %d  TotalWordsNum: %ld  \n", lemmaVocabSize, synsetVocabSize, wordVocabSize, totalWordsNum);
    }

    void Vocab::save(const std::string& filepath) {
        std::ofstream fout(filepath);
        if (fout.fail()) {
            throw std::runtime_error("Cannot open vocab data file");
        }

        fout << lemmaVocabSize << " " << synsetVocabSize << " " << wordVocabSize << "\n";
        fout << totalWordsNum << " " << totalSentenceNum << " " << totalDocumentNum << "\n";

        for (auto& pair : lemmaVocab) {
            auto lemma = pair.first;
            int lidx = pair.second;

            auto lemmaData = sv4d::utils::string::split(lemma, '|');
            auto word = lemmaData[0];
            auto synset = lemmaData[2];

            int widx = synsetVocab[word];
            int sidx = 0;
            if (synset != "*") {
                sidx = synsetVocab[synset];
            } else {
                sidx = widx;
            }

            std::string dictPair;
            if (synsetDictPair.find(sidx) != synsetDictPair.end()) {
                if (synsetDictPair[sidx].dictPair.size() != 0) {
                    dictPair = sv4d::utils::string::join(sv4d::utils::string::intvec_to_strvec(synsetDictPair[sidx].dictPair), ',');
                } else {
                    dictPair = "-1";
                }
            } else {
                dictPair = "-1";
            }

            fout << lemma << " " << lidx << " " << lemmaProb[lidx] << " " << widx << " " << wordFreq[widx] << " " << sidx << " " << dictPair << "\n";
        }

        fout.close();
    }

    void Vocab::load(const std::string& filepath) {
        std::string linebuf;
        std::ifstream fin(filepath);
        if (fin.fail()) {
            throw std::runtime_error("Cannot open vocab data file");
        }

        std::getline(fin, linebuf);
        auto sizes = sv4d::utils::string::split(sv4d::utils::string::trim(linebuf), ' ');
        lemmaVocabSize = std::stol(sizes[0]);
        synsetVocabSize = std::stol(sizes[1]);
        wordVocabSize = std::stol(sizes[2]);
        lidx2Lemma.resize(lemmaVocabSize);
        lemmaProb.resize(lemmaVocabSize);
        lidx2sidx.resize(lemmaVocabSize);
        sidx2Synset.resize(synsetVocabSize);
        widx2lidxs.resize(wordVocabSize);
        wordFreq.resize(wordVocabSize);

        std::getline(fin, linebuf);
        auto nums = sv4d::utils::string::split(sv4d::utils::string::trim(linebuf), ' ');
        totalWordsNum = std::stoi(nums[0]);
        totalSentenceNum = std::stoi(nums[1]);
        totalDocumentNum = std::stoi(nums[2]);

        while (std::getline(fin, linebuf)) {
            linebuf = sv4d::utils::string::trim(linebuf);
            auto data = sv4d::utils::string::split(linebuf, ' ');
            auto lemmaData = sv4d::utils::string::split(data[0], '|');
            auto word = lemmaData[0];
            auto pos = lemmaData[1];
            auto synset = lemmaData[2];

            int lidx = std::stoi(data[1]);
            int sidx = std::stoi(data[5]);
            int widx = std::stoi(data[3]);
            lemmaVocab[word + "|" + pos + "|" + synset] = lidx;
            lidx2Lemma[lidx] = word + "|" + pos + "|" + synset;
            lemmaProb[lidx] = std::stof(data[2]);
            lidx2sidx[lidx] = sidx;

            if (synsetVocab.find(word) == synsetVocab.end()) {
                synsetVocab[word] = widx;
                sidx2Synset[widx] = word;
                wordFreq[widx] = std::stoi(data[4]);
                widx2lidxs[widx] = sv4d::SynsetData();
                widx2lidxs[widx].wordLemmaIndex = widx;
            }
            
            if (synset != "*") {
                if (synsetVocab.find(synset) == synsetVocab.end()) {
                    synsetVocab[synset] = sidx;
                    sidx2Synset[sidx] = synset;
                    if (data[6] != "-1") {
                        synsetDictPair[sidx] = sv4d::SynsetDictPair();
                        synsetDictPair[sidx].dictPair = sv4d::utils::string::strvec_to_intvec(sv4d::utils::string::split(data[6], ','));
                    } else {
                        synsetDictPair[sidx] = sv4d::SynsetDictPair();
                    }
                }
                
                if (pos == "n") {
                    widx2lidxs[widx].synsetLemmaIndices[sv4d::Pos::Noun].push_back(lidx);
                    if (std::find(widx2lidxs[widx].validPos.begin(), widx2lidxs[widx].validPos.end(), (int)sv4d::Pos::Noun) == widx2lidxs[widx].validPos.end()) {
                        widx2lidxs[widx].validPos.push_back(sv4d::Pos::Noun);
                    }
                } else if (pos == "v") {
                    widx2lidxs[widx].synsetLemmaIndices[sv4d::Pos::Verb].push_back(lidx);
                    if (std::find(widx2lidxs[widx].validPos.begin(), widx2lidxs[widx].validPos.end(), (int)sv4d::Pos::Verb) == widx2lidxs[widx].validPos.end()) {
                        widx2lidxs[widx].validPos.push_back(sv4d::Pos::Verb);
                    }
                } else if (pos == "a") {
                    widx2lidxs[widx].synsetLemmaIndices[sv4d::Pos::Adjective].push_back(lidx);
                    if (std::find(widx2lidxs[widx].validPos.begin(), widx2lidxs[widx].validPos.end(), (int)sv4d::Pos::Adjective) == widx2lidxs[widx].validPos.end()) {
                        widx2lidxs[widx].validPos.push_back(sv4d::Pos::Adjective);
                    }
                } else if (pos == "r") {
                    widx2lidxs[widx].synsetLemmaIndices[sv4d::Pos::Adverb].push_back(lidx);
                    if (std::find(widx2lidxs[widx].validPos.begin(), widx2lidxs[widx].validPos.end(), (int)sv4d::Pos::Adverb) == widx2lidxs[widx].validPos.end()) {
                        widx2lidxs[widx].validPos.push_back(sv4d::Pos::Adverb);
                    }
                }
            }
        }
    }

}