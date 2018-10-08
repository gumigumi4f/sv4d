#include "model.hpp"

#include "options.hpp"
#include "vocab.hpp"
#include "matrix.hpp"
#include "vector.hpp"
#include "utils.hpp"
#include <vector>
#include <thread>
#include <fstream>
#include <chrono>
#include <stdio.h>

namespace sv4d {

    Model::Model(const sv4d::Options& opt, const sv4d::Vocab& v) {
        vocab = v;

        trainingCorpus = opt.trainingCorpus;

        epochs = opt.epochs;
        embeddingLayerSize = opt.embeddingLayerSize;
        windowSize = opt.windowSize;
        negativeSample = opt.negativeSample;
        dictSample = opt.dictSample;
        maxDictPair = opt.maxDictPair;
        threadNum = opt.threadNum;
        fileSize = 0;

        subSamplingFactor = opt.subSamplingFactor;
        initialLearningRate = opt.initialLearningRate;
        minLearningRate = opt.minLearningRate;
        initialTemperature = opt.initialTemperature;
        minTemperature = opt.minTemperature;
        initialBetaDict = opt.initialBetaDict;
        minBetaDict = opt.minBetaDict;
        initialBetaReward = opt.initialBetaReward;
        minBetaReward = opt.minBetaReward;
        
        senseSelectionOutWeight = sv4d::Matrix(vocab.lemmaVocabSize, embeddingLayerSize * 3);
        senseSelectionOutBias = sv4d::Vector(vocab.lemmaVocabSize);
        embeddingInWeight = sv4d::Matrix(vocab.synsetVocabSize, embeddingLayerSize);
        embeddingOutWeight = sv4d::Matrix(vocab.wordVocabSize, embeddingLayerSize);

        senseSelectionOutWeight.setGlorotUniform();
        senseSelectionOutBias.setGlorotUniform();
        embeddingInWeight.setGlorotUniform();

        unigramTable = std::vector<int>();
        subsamplingFactorTable = std::vector<double>();

        trainedWordCount = 0;
    }

    void Model::Initialize() {
        InitializeUnigramTable();
        InitializeSubsamplingFactorTable();
        InitializeSigmoidTable();
        InitializeFileSize();
    }

    void Model::Training() {
        startTime = std::chrono::system_clock::now();
        trainedWordCount = 0;
        auto threads = std::vector<std::thread>();
        if (threadNum > 1) {
            for (int i = 0; i < threadNum; i++) {
                threads.push_back(std::thread(&Model::TrainingThread, this, i));
            }
            for (int i = 0; i < threadNum; i++) {
                threads[i].join();
            }
        } else {
            Model::TrainingThread(0);
        }
    }

    void Model::InitializeUnigramTable() {
        const double power = 0.75;
        double trainWordsPow = 0;

        unigramTable.resize(UnigramTableSize);
        
        for (int a = 0; a < vocab.wordVocabSize; ++a) {
            trainWordsPow += std::pow(vocab.wordFreq[a], power);
        }

        int i = 0;
        double d1 = std::pow(vocab.wordFreq[i], power) / trainWordsPow;
        for (int a = 0; a < UnigramTableSize; ++a) {
            unigramTable[a] = i;
            if (a / (double)UnigramTableSize > d1) {
                i++;
                d1 += std::pow(vocab.wordFreq[i], power) / trainWordsPow;
            }

            if (i >= vocab.wordVocabSize) {
                i = vocab.wordVocabSize - 1;
            }
        }
    }

    void Model::InitializeSubsamplingFactorTable() {
        subsamplingFactorTable.resize(vocab.wordVocabSize);
        for (int i = 0; i < vocab.wordVocabSize; ++i) {
            auto factor = (std::sqrt(vocab.wordFreq[i] / (subSamplingFactor * vocab.totalWordsNum)) + 1) * (subSamplingFactor * vocab.totalWordsNum) / vocab.wordFreq[i];
            subsamplingFactorTable[i] = factor;
        }
    }

    void Model::InitializeFileSize() {
        auto fin = std::ifstream(trainingCorpus);
        fin.seekg(0, fin.end);
        fileSize = fin.tellg();
    }

    void Model::TrainingThread(const int threadId) {
        std::string linebuf;
        auto documentCache = std::vector<std::vector<int>>();

        auto fin = std::ifstream(trainingCorpus);

        for (int iter = 0; iter < epochs; iter++) {
            fin.seekg(fileSize / threadNum * threadId, fin.beg);
            // seek to head of document
            while (true) {
                std::getline(fin, linebuf);
                if (sv4d::utils::string::trim(linebuf) == "</doc>") {
                    break;
                }
            }

            // generate batch and process
            while (std::getline(fin, linebuf)) {
                linebuf = sv4d::utils::string::trim(linebuf);
                if (linebuf == "<doc>") {
                    documentCache.clear();
                } else if (linebuf == "</doc>") {
                    // document vector
                    sv4d::Vector documentVector = sv4d::Vector(embeddingLayerSize);
                    int documentNum = 0;
                    for (auto sentence : documentCache) {
                        for (auto widx : sentence) {
                            documentNum += 1;
                            documentVector += embeddingInWeight[widx];
                        }
                    }
                    documentVector /= documentNum;

                    for (auto sentence : documentCache) {
                        // sentence vector
                        sv4d::Vector sentenceVector = sv4d::Vector(embeddingLayerSize);
                        int sentenceNum = 0;
                        for (auto widx : sentence) {
                            sentenceNum += 1;
                            sentenceVector += embeddingInWeight[widx];
                        }
                        sentenceVector /= sentenceNum;
                        Model::ProcessBatch(documentVector, sentenceVector);

                        trainedWordCount += sentence.size();
                    }

                    if (fin.tellg() > fileSize / threadNum * (threadId + 1)) {
                        break;
                    }

                    // print log
                    auto now = std::chrono::system_clock::now();
                    auto progress = trainedWordCount / (double)((iter + 1) * vocab.totalWordsNum + 1) * 100.0;
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
                    auto speed = trainedWordCount / ((elapsed + 1) / 1000.0) / (1000.0 * threadNum);
                    printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, 0.025, progress, speed);
                    fflush(stdout);

                } else {
                    linebuf = sv4d::utils::string::trim(linebuf);
                    auto widxes = std::vector<int>();
                    auto words = sv4d::utils::string::split(linebuf, ' ');
                    if (words.size() < 5) {
                        continue;
                    }
                    for (auto word : words) {
                        if (vocab.synsetVocab.find(word) == vocab.synsetVocab.end()) {
                            continue;
                        }
                        widxes.push_back(vocab.synsetVocab[word]);
                    }
                    documentCache.push_back(widxes);
                }
            }
        }
    }

    void Model::ProcessBatch(const sv4d::Vector& documentVector, const sv4d::Vector& sentenceVector) {
        
    }

}