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
#include <random>
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
        std::ifstream fin(trainingCorpus);
        fin.seekg(0, fin.end);
        fileSize = fin.tellg();
    }

    void Model::TrainingThread(const int threadId) {
        std::string linebuf;

        // random
        std::mt19937 mt(495);
        std::uniform_real_distribution<double> rand(0, 1);
        std::uniform_int_distribution<int> rndwindow(0, windowSize - 1);

        // file
        std::ifstream fin(trainingCorpus);

        // cache
        auto documentCache = std::vector<std::vector<int>>();
        sv4d::Vector documentVector = sv4d::Vector(embeddingLayerSize);
        sv4d::Vector sentenceVector = sv4d::Vector(embeddingLayerSize);
        sv4d::Vector contextVector = sv4d::Vector(embeddingLayerSize);
        auto candidateOutputWidx = std::vector<int>(windowSize * 2);
        auto subSampled = std::vector<bool>();

        for (int iter = 0; iter < epochs; ++iter) {
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
                    documentVector.setZero();
                    int documentNum = 0;
                    for (auto sentence : documentCache) {
                        for (auto widx : sentence) {
                            documentNum += 1;
                            documentVector += embeddingInWeight[widx];
                        }
                    }
                    documentVector /= documentNum;

                    for (auto sentence : documentCache) {
                        auto sentenceSize = sentence.size();
                        subSampled.clear();

                        // sentence vector
                        sentenceVector.setZero();
                        int sentenceNum = 0;
                        for (auto widx : sentence) {
                            sentenceNum += 1;
                            sentenceVector += embeddingInWeight[widx];
                            subSampled.push_back(subsamplingFactorTable[widx] < rand(mt));
                        }
                        sentenceVector /= sentenceNum;

                        for (int pos = 0; pos < sentenceSize; ++pos) {
                            if (subSampled[pos]) {
                                continue;
                            }
                            // input widx
                            int inputWidx = sentence[pos];

                            // context vector
                            contextVector.setZero();
                            int contextNum = 0;
                            int minPos = pos - windowSize < 0 ? 0 : pos - windowSize;
                            int maxPos = pos + windowSize >= sentenceSize ? sentenceSize - 1 : pos + windowSize;
                            for (int pos2 = minPos; pos2 <= maxPos; ++pos2) {
                                contextVector += embeddingInWeight[sentence[pos2]];
                                contextNum += 1;
                            }
                            contextVector /= contextNum;

                            // output widx
                            candidateOutputWidx.clear();
                            int reducedWindowSize = windowSize - rndwindow(mt);
                            int count;
                            count = reducedWindowSize;
                            for (int pos2 = pos; pos2 >= 0; --pos2) {
                                if (subSampled[pos2]) {
                                    continue;
                                }
                                candidateOutputWidx.push_back(sentence[pos2]);
                                count -= 1;
                                if (count == 0) {
                                    break;
                                }
                            }
                            count = reducedWindowSize;
                            for (int pos2 = pos; pos2 < sentenceSize; ++pos2) {
                                if (subSampled[pos2]) {
                                    continue;
                                }
                                candidateOutputWidx.push_back(sentence[pos2]);
                                count -= 1;
                                if (count == 0) {
                                    break;
                                }
                            }

                            // output widx
                            int outputWidx = candidateOutputWidx[mt() % candidateOutputWidx.size()];

                            Model::ProcessBatch(documentVector, sentenceVector, contextVector, inputWidx, outputWidx);
                        }

                        trainedWordCount += sentence.size();
                    }

                    if (fin.tellg() > fileSize / threadNum * (threadId + 1)) {
                        break;
                    }

                    // print log
                    auto now = std::chrono::system_clock::now();
                    auto progress = trainedWordCount / (double)((epochs + 1) * vocab.totalWordsNum + 1) * 100.0;
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
                    auto speed = trainedWordCount / ((elapsed + 1) / 1000.0) / (1000.0 * threadNum);
                    printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, 0.025, progress, speed);
                    fflush(stdout);

                } else {
                    auto widxes = std::vector<int>();
                    auto sentence = sv4d::utils::string::split(linebuf, ' ');
                    if (sentence.size() < 5) {
                        trainedWordCount += sentence.size();
                        continue;
                    }
                    for (auto word : sentence) {
                        if (vocab.synsetVocab.find(word) == vocab.synsetVocab.end()) {
                            continue;
                        }
                        auto widx = vocab.synsetVocab[word];
                        if (widx >= vocab.wordVocabSize) {
                            continue;
                        }

                        widxes.push_back(widx);
                    }
                    documentCache.push_back(widxes);
                }
            }
        }
    }

    inline void Model::ProcessBatch(const sv4d::Vector& documentVector, const sv4d::Vector& sentenceVector, const sv4d::Vector& contextVector, const int inputWidx, const int outputWidx) {
        
    }

}