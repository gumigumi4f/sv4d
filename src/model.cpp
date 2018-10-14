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
#include <cmath>
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

        //senseSelectionOutWeight.setRandomUniform(-0.5 / embeddingLayerSize, 0.5 / embeddingLayerSize);
        //senseSelectionOutBias.setRandomUniform(-0.5 / embeddingLayerSize, 0.5 / embeddingLayerSize);
        embeddingInWeight.setRandomUniform(-0.5 / embeddingLayerSize, 0.5 / embeddingLayerSize);

        unigramTable = std::vector<int>();
        subsamplingFactorTable = std::vector<float>();

        trainedWordCount = 0;
    }

    void Model::initialize() {
        initializeUnigramTable();
        initializeSubsamplingFactorTable();
        initializeFileSize();
    }

    void Model::training() {
        startTime = std::chrono::system_clock::now();
        trainedWordCount = 0;
        auto threads = std::vector<std::thread>();
        if (threadNum > 1) {
            for (int i = 0; i < threadNum; i++) {
                threads.push_back(std::thread(&Model::trainingThread, this, i));
            }
            for (int i = 0; i < threadNum; i++) {
                threads[i].join();
            }
        } else {
            Model::trainingThread(0);
        }
    }

    void Model::initializeUnigramTable() {
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

        std::mt19937 engine(495);
        std::shuffle(unigramTable.begin(), unigramTable.end(), engine);
    }

    void Model::initializeSubsamplingFactorTable() {
        subsamplingFactorTable.resize(vocab.wordVocabSize);
        for (int i = 0; i < vocab.wordVocabSize; ++i) {
            auto factor = (std::sqrt(vocab.wordFreq[i] / (subSamplingFactor * vocab.totalWordsNum)) + 1) * (subSamplingFactor * vocab.totalWordsNum) / vocab.wordFreq[i];
            subsamplingFactorTable[i] = factor;
        }
    }

    void Model::initializeFileSize() {
        std::ifstream fin(trainingCorpus);
        fin.seekg(0, fin.end);
        fileSize = fin.tellg();
    }

    void Model::trainingThread(const int threadId) {
        std::string linebuf;

        // file
        std::ifstream fin(trainingCorpus);

        // random
        std::mt19937 mt(495 + threadId);
        std::uniform_real_distribution<double> rand(0, 1);
        std::uniform_int_distribution<int> rndwindow(0, windowSize - 1);

        // initialize negative sampling position
        int negativePos = mt() % UnigramTableSize;

        // hyper parameter
        float lr = initialLearningRate;
        float temperature = initialTemperature;
        float betaDict = initialBetaDict;
        float betaReward = initialBetaReward;

        // cache
        auto documentCache = std::vector<std::vector<int>>();
        auto outputWidxCandidateCache = std::vector<int>(windowSize * 2);
        auto subSampledCache = std::vector<bool>();
        sv4d::Vector documentVectorCache = sv4d::Vector(embeddingLayerSize);
        sv4d::Vector sentenceVectorCache = sv4d::Vector(embeddingLayerSize);
        sv4d::Vector contextVectorCache = sv4d::Vector(embeddingLayerSize);
        sv4d::Vector featureVectorCache = sv4d::Vector(embeddingLayerSize * 3);

        for (int iter = 0; iter < epochs; ++iter) {
            fin.clear();
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
                    documentVectorCache.setZero();
                    int documentNum = 0;
                    for (auto& sentence : documentCache) {
                        for (auto widx : sentence) {
                            documentNum += 1;
                            sv4d::Vector& embeddingInVector = embeddingInWeight[widx];
                            documentVectorCache += embeddingInVector;
                        }
                    }
                    documentVectorCache /= documentNum;

                    for (auto& sentence : documentCache) {
                        auto sentenceSize = sentence.size();
                        subSampledCache.clear();

                        // sentence vector
                        sentenceVectorCache.setZero();
                        int sentenceNum = 0;
                        for (auto widx : sentence) {
                            sentenceNum += 1;
                            sv4d::Vector& embeddingInVector = embeddingInWeight[widx];
                            sentenceVectorCache += embeddingInVector;
                            subSampledCache.push_back(subsamplingFactorTable[widx] < rand(mt));
                        }
                        sentenceVectorCache /= sentenceNum;

                        for (int pos = 0; pos < sentenceSize; ++pos) {
                            if (subSampledCache[pos]) {
                                continue;
                            }

                            // context vector
                            contextVectorCache.setZero();
                            int contextNum = 0;
                            int minPos = pos - windowSize < 0 ? 0 : pos - windowSize;
                            int maxPos = pos + windowSize >= sentenceSize ? sentenceSize - 1 : pos + windowSize;
                            for (int pos2 = minPos; pos2 <= maxPos; ++pos2) {
                                if (pos == pos2) {
                                    continue;
                                }
                                sv4d::Vector& embeddingInVector = embeddingInWeight[sentence[pos2]];
                                contextVectorCache += embeddingInVector;
                                contextNum += 1;
                            }
                            contextVectorCache /= contextNum;

                            // output widx
                            outputWidxCandidateCache.clear();
                            int reducedWindowSize = windowSize - rndwindow(mt);
                            int count = reducedWindowSize;
                            for (int pos2 = pos - 1; pos2 >= 0; --pos2) {
                                if (subSampledCache[pos2]) {
                                    continue;
                                }
                                outputWidxCandidateCache.push_back(sentence[pos2]);
                                count -= 1;
                                if (count == 0) {
                                    break;
                                }
                            }
                            count = reducedWindowSize;
                            for (int pos2 = pos + 1; pos2 < sentenceSize; ++pos2) {
                                if (outputWidxCandidateCache[pos2]) {
                                    continue;
                                }
                                outputWidxCandidateCache.push_back(sentence[pos2]);
                                count -= 1;
                                if (count == 0) {
                                    break;
                                }
                            }

                            if (outputWidxCandidateCache.size() == 0) {
                                continue;
                            }

                            int outputWidx = outputWidxCandidateCache[mt() % outputWidxCandidateCache.size()];

                            // input widx
                            int inputWidx = sentence[pos];

                            // feature vector
                            for (int i = 0; i < embeddingLayerSize; ++i) {
                                featureVectorCache[i] = contextVectorCache[i];
                                featureVectorCache[i + embeddingLayerSize] = sentenceVectorCache[i];
                                featureVectorCache[i + embeddingLayerSize * 2] = documentVectorCache[i];
                            }

                            // training
                            // % means dot operation
                            {
                                sv4d::Vector embeddingOutBufVector = sv4d::Vector(embeddingLayerSize);

                                sv4d::SynsetData& synsetData = vocab.widx2lidxs[inputWidx];

                                sv4d::Vector& vWordOut = embeddingOutWeight[outputWidx];
                                sv4d::Vector& vWordOutIn = embeddingInWeight[outputWidx];

                                // sense training
                                if (synsetData.validPos.size() != 0) {
                                    int targetPos = synsetData.validPos[mt() % synsetData.validPos.size()];
                                    auto& synsetLemmaIndices = synsetData.synsetLemmaIndices[targetPos];

                                    // sense selection
                                    int senseNum = synsetLemmaIndices.size();
                                    sv4d::Vector senseSelectionLogits = sv4d::Vector(senseNum);
                                    for (int i = 0; i < senseNum; ++i) {
                                        int lidx = synsetLemmaIndices[i];
                                        senseSelectionLogits[i] = (senseSelectionOutWeight[lidx] % featureVectorCache) + senseSelectionOutBias[lidx];
                                    }
                                    auto senseSelectionProb = senseSelectionLogits.softmax(temperature);

                                    sv4d::Vector rewardLogits = sv4d::Vector(senseNum);

                                    // embedding
                                    for (int i = 0; i < senseNum; ++i) {
                                        sv4d::Vector embeddingInBufVector = sv4d::Vector(embeddingLayerSize);

                                        auto senseWeight = senseSelectionProb[i];

                                        auto sidx = vocab.lidx2sidx[synsetLemmaIndices[i]];
                                        sv4d::SynsetDictPair synsetDictPair;
                                        if (vocab.synsetDictPair.find(sidx) == vocab.synsetDictPair.end()) {
                                            synsetDictPair = sv4d::SynsetDictPair();
                                        } else {
                                            synsetDictPair = vocab.synsetDictPair[sidx];
                                        }
                                        auto& dictPair = vocab.synsetDictPair[sidx].dictPair;

                                        sv4d::Vector& vSynsetIn = embeddingInWeight[sidx];

                                        // Positive: example predicts label.
                                        //   forward: x = v_in' * v_out
                                        //            l = log(sigmoid(x))
                                        //   backward: dl/dx = g = sigmoid(-x)
                                        //             dl/d(v_in) = g * v_out'
                                        //             dl/d(v_out) = v_in' * g
                                        {
                                            float dot = vSynsetIn % vWordOut;
                                            rewardLogits[i] += dot * betaReward;
                                            float g = sv4d::utils::operation::sigmoid(-dot);
                                            embeddingInBufVector += vWordOut * (g * lr * senseWeight);
                                            embeddingOutBufVector += vSynsetIn * (g * lr * senseWeight);
                                        }

                                        // Negative samples:
                                        //   forward: x = v_in' * v_sample
                                        //            l = log(sigmoid(-x))
                                        //   backward: dl/dx = g = -sigmoid(x)
                                        //             dl/d(v_in) = g * v_out'
                                        //             dl/d(v_out) = v_in' * g
                                        for (int j = 0; j < negativeSample; ++j) {
                                            int sample = unigramTable[negativePos];
                                            if (negativePos == UnigramTableSize - 1) {
                                                negativePos = 0;
                                            } else {
                                                negativePos += 1;
                                            }
                                            if (sample == outputWidx) {
                                                continue;
                                            }
                                            if (std::find(dictPair.begin(), dictPair.end(), sample) != dictPair.end()) {
                                                break;
                                            } 
                                            sv4d::Vector& vSample = embeddingOutWeight[sample];
                                            float dot = vSynsetIn % vSample;
                                            float g = -sv4d::utils::operation::sigmoid(dot);
                                            embeddingInBufVector += vSample * (g * lr * senseWeight);
                                            vSample += vSynsetIn * (g * lr * senseWeight);
                                        }

                                        // Positive: dictionary pairs for accurate prediction
                                        //   forward: x = v_in' * v_out
                                        //            l = log(sigmoid(x))
                                        //   backward: dl/dx = g = sigmoid(-x)
                                        //             dl/d(v_in) = g * v_out'
                                        //             dl/d(v_out) = v_in' * g
                                        for (int j = 0; j < dictSample; ++j) {
                                            auto sample = dictPair[synsetDictPair.dpos];
                                            if (synsetDictPair.dpos == dictPair.size() - 1) {
                                                synsetDictPair.dpos = 0;
                                            } else {
                                                synsetDictPair.dpos += 1;
                                            }
                                            sv4d::Vector& vSample = embeddingOutWeight[sample];
                                            rewardLogits[i] += (vWordOutIn % vSample) * betaReward;
                                            float dot = vSynsetIn % vSample;
                                            float g = sv4d::utils::operation::sigmoid(-dot);
                                            float d = (g * lr * betaDict / senseNum);
                                            embeddingInBufVector += vSample * d;
                                            vSample += vSynsetIn * d;
                                        }

                                        vSynsetIn += embeddingInBufVector;
                                    }

                                    // sense selection (update)
                                    auto rewardProb = rewardLogits.softmax(1.0);
                                    rewardProb.clipByValue(0.1, 0.9);
                                    for (int i = 0; i < senseNum; ++i) {
                                        // Update sense selection weight.
                                        //   forward: x = v_feature' * v_sense_selection + v_sense_bias
                                        //            l = z * log(sigmoid(x)) + (1 - z) * log(sigmoid(-x))
                                        //   backward: dl/dx = g = z * sigmoid(-x) + (1 - z) * -sigmoid(x)
                                        //             dl/d(v_sense_selection) = v_feature' * g
                                        //             dl/d(v_sense_bias) = g
                                        int lidx = synsetLemmaIndices[i];
                                        float g = rewardProb[i] * sv4d::utils::operation::sigmoid(-senseSelectionLogits[i])
                                                  + (1.0 - rewardProb[i]) * -sv4d::utils::operation::sigmoid(senseSelectionLogits[i]);
                                        sv4d::Vector& vSenseSelection = senseSelectionOutWeight[lidx];
                                        auto& bSenseSelection = senseSelectionOutBias[lidx];
                                        vSenseSelection += featureVectorCache * (g * lr);
                                        bSenseSelection += (g * lr);
                                    }
                                }

                                // word training
                                {
                                    sv4d::Vector embeddingInBufVector = sv4d::Vector(embeddingLayerSize);

                                    auto wsidx = vocab.lidx2sidx[synsetData.wordLemmaIndex];

                                    sv4d::Vector& vWordIn = embeddingInWeight[wsidx];
                                    
                                    // Positive: example predicts label.
                                    //   forward: x = v_in' * v_out
                                    //            l = log(sigmoid(x))
                                    //   backward: dl/dx = g = sigmoid(-x)
                                    //             dl/d(v_in) = g * v_out'
                                    //             dl/d(v_out) = v_in' * g
                                    {
                                        float dot = vWordIn % vWordOut;
                                        float g = sv4d::utils::operation::sigmoid(-dot);
                                        embeddingInBufVector += vWordOut * (g * lr);
                                        embeddingOutBufVector += vWordIn * (g * lr);
                                    }

                                    // Negative samples:
                                    //   forward: x = v_in' * v_sample
                                    //            l = log(sigmoid(-x))
                                    //   backward: dl/dx = g = -sigmoid(x)
                                    //             dl/d(v_in) = g * v_out'
                                    //             dl/d(v_out) = v_in' * g
                                    for (int j = 0; j < negativeSample; ++j) {
                                        int sample = unigramTable[negativePos];
                                        if (negativePos == UnigramTableSize - 1) {
                                            negativePos = 0;
                                        } else {
                                            negativePos += 1;
                                        }
                                        if (sample == outputWidx) {
                                            continue;
                                        }
                                        sv4d::Vector& vSample = embeddingOutWeight[sample];
                                        float dot = vWordIn % vSample;
                                        float g = -sv4d::utils::operation::sigmoid(dot);
                                        embeddingInBufVector += vSample * (g * lr);
                                        vSample += vWordIn * (g * lr);
                                    }

                                    vWordIn += embeddingInBufVector;
                                }

                                vWordOut += embeddingOutBufVector;
                            }
                        }

                        trainedWordCount += sentenceSize;
                    }

                    if (fin.tellg() > fileSize / threadNum * (threadId + 1)) {
                        break;
                    }

                    lr = initialLearningRate * (1 - trainedWordCount / (double)(epochs * vocab.totalWordsNum + 1));
                    if (lr < minLearningRate) {
                        lr = minLearningRate;
                    }
                    betaDict = initialBetaDict * (1 - trainedWordCount / (double)(epochs * vocab.totalWordsNum + 1));
                    if (temperature < minBetaDict) {
                        temperature = minBetaDict;
                    }
                    betaReward = initialBetaReward * (1 - trainedWordCount / (double)(epochs * vocab.totalWordsNum + 1));
                    if (temperature < minBetaReward) {
                        temperature = minBetaReward;
                    }

                    // print log
                    auto now = std::chrono::system_clock::now();
                    auto progress = trainedWordCount / (double)(epochs * vocab.totalWordsNum + 1) * 100.0;
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
                    auto speed = trainedWordCount / ((elapsed + 1) / 1000.0) / (1000.0 * threadNum);
                    auto eta = ((double)(epochs * vocab.totalWordsNum) / (trainedWordCount + 1) * elapsed - elapsed) / 60000.0;
                    printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  Remaining: %.2fm  ", 13, lr, progress, speed, eta);
                    fflush(stdout);
                } else {
                    auto widxes = std::vector<int>();
                    auto sentence = sv4d::utils::string::split(linebuf, ' ');
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

                    if (widxes.size() < 5) {
                        trainedWordCount += widxes.size();
                        continue;
                    }
                    documentCache.push_back(widxes);
                }
            }
        }
    }

    void Model::saveEmbeddingInWeight(const std::string& filepath) {
        std::ofstream fout(filepath);

        fout << vocab.synsetVocabSize << " " << embeddingLayerSize << "\n";
        for (int sidx = 0; sidx < vocab.synsetVocabSize; ++sidx) {
            auto synset = vocab.sidx2Synset[sidx];
            auto strvec = sv4d::utils::string::join(sv4d::utils::string::floatvec_to_strvec(embeddingInWeight[sidx].getData()), ' ');
            fout << synset << " " << strvec << "\n";
        }

        fout.close();
    }

    void Model::loadEmbeddingInWeight(const std::string& filepath) {
        std::string linebuf;
        std::ifstream fin(filepath);

        std::getline(fin, linebuf);
        auto sizes = sv4d::utils::string::split(sv4d::utils::string::trim(linebuf), ' ');
        auto synsetVocabSize = std::stoi(sizes[0]);
        embeddingLayerSize = std::stoi(sizes[1]);

        for (int i = 0; i < synsetVocabSize; ++i) {
            std::getline(fin, linebuf);
            auto data = sv4d::utils::string::split(sv4d::utils::string::trim(linebuf), ' ');
            if (vocab.synsetVocab.find(data[0]) == vocab.synsetVocab.end()) {
                continue;
            }
            auto strvec = std::vector<std::string>(data.begin() + 1, data.end());
            embeddingInWeight[vocab.synsetVocab[data[0]]].getData() = sv4d::utils::string::strvec_to_floatvec(strvec);
        }
    }

    void Model::saveEmbeddingOutWeight(const std::string& filepath) {
        std::ofstream fout(filepath);

        fout << vocab.wordVocabSize << " " << embeddingLayerSize << "\n";
        for (int widx = 0; widx < vocab.wordVocabSize; ++widx) {
            auto word = vocab.sidx2Synset[widx];
            auto strvec = sv4d::utils::string::join(sv4d::utils::string::floatvec_to_strvec(embeddingOutWeight[widx].getData()), ' ');
            fout << word << " " << strvec << "\n";
        }

        fout.close();
    }

    void Model::loadEmbeddingOutWeight(const std::string& filepath) {
        std::string linebuf;
        std::ifstream fin(filepath);

        std::getline(fin, linebuf);
        auto sizes = sv4d::utils::string::split(sv4d::utils::string::trim(linebuf), ' ');
        auto wordVocabSize = std::stoi(sizes[0]);
        embeddingLayerSize = std::stoi(sizes[1]);

        for (int i = 0; i < wordVocabSize; ++i) {
            std::getline(fin, linebuf);
            auto data = sv4d::utils::string::split(sv4d::utils::string::trim(linebuf), ' ');
            if (vocab.synsetVocab.find(data[0]) == vocab.synsetVocab.end()) {
                continue;
            }
            auto strvec = std::vector<std::string>(data.begin() + 1, data.end());
            embeddingOutWeight[vocab.synsetVocab[data[0]]].getData() = sv4d::utils::string::strvec_to_floatvec(strvec);
        }
    }

    void Model::saveSenseSelectionOutWeight(const std::string& filepath) {
        std::ofstream fout(filepath);

        fout << vocab.lemmaVocabSize << " " << embeddingLayerSize * 3 << "\n";
        for (int lidx = 0; lidx < vocab.lemmaVocabSize; ++lidx) {
            auto lemma = vocab.lidx2Lemma[lidx];
            auto strvec = sv4d::utils::string::join(sv4d::utils::string::floatvec_to_strvec(senseSelectionOutWeight[lidx].getData()), ' ');
            fout << lemma << " " << strvec << "\n";
        }

        fout.close();
    }

    void Model::loadSenseSelectionOutWeight(const std::string& filepath) {
        std::string linebuf;
        std::ifstream fin(filepath);

        std::getline(fin, linebuf);
        auto sizes = sv4d::utils::string::split(sv4d::utils::string::trim(linebuf), ' ');
        auto lemmaVocabSize = std::stoi(sizes[0]);
        embeddingLayerSize = std::stoi(sizes[1]);

        for (int i = 0; i < lemmaVocabSize; ++i) {
            std::getline(fin, linebuf);
            auto data = sv4d::utils::string::split(sv4d::utils::string::trim(linebuf), ' ');
            if (vocab.lemmaVocab.find(data[0]) == vocab.lemmaVocab.end()) {
                continue;
            }
            auto strvec = std::vector<std::string>(data.begin() + 1, data.end());
            embeddingOutWeight[vocab.lemmaVocab[data[0]]].getData() = sv4d::utils::string::strvec_to_floatvec(strvec);
        }
    }

    void Model::saveSenseSelectionBiasWeight(const std::string& filepath) {
        std::ofstream fout(filepath);

        fout << vocab.lemmaVocabSize << " " << 1 << "\n";
        for (int lidx = 0; lidx < vocab.lemmaVocabSize; ++lidx) {
            auto lemma = vocab.lidx2Lemma[lidx];
            fout << lemma << " " << senseSelectionOutBias[lidx] << "\n";
        }

        fout.close();
    }

    void Model::loadSenseSelectionBiasWeight(const std::string& filepath) {
        std::string linebuf;
        std::ifstream fin(filepath);

        std::getline(fin, linebuf);
        auto sizes = sv4d::utils::string::split(sv4d::utils::string::trim(linebuf), ' ');
        auto lemmaVocabSize = std::stoi(sizes[0]);

        for (int i = 0; i < lemmaVocabSize; ++i) {
            std::getline(fin, linebuf);
            auto data = sv4d::utils::string::split(sv4d::utils::string::trim(linebuf), ' ');
            senseSelectionOutBias.getData()[i] = std::stof(data[1]);
        }
    }

}