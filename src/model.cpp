#include "model.hpp"

#include "options.hpp"
#include "vocab.hpp"
#include "matrix.hpp"
#include "vector.hpp"
#include "utils.hpp"
#include <vector>
#include <algorithm>
#include <thread>
#include <fstream>
#include <chrono>
#include <random>
#include <cmath>
#include <utility>
#include <deque>
#include <unordered_set>
#include <stdio.h>

namespace sv4d {

    Model::Model(const sv4d::Options& opt, const sv4d::Vocab& v) {
        vocab = v;

        trainingCorpus = opt.trainingCorpus;
        stopWordsFile = opt.stopWordsFile;

        epochs = opt.epochs;
        embeddingLayerSize = opt.embeddingLayerSize;
        windowSize = opt.windowSize;
        negativeSample = opt.negativeSample;
        dictSample = opt.dictSample;
        maxDictPair = opt.maxDictPair;
        threadNum = opt.threadNum;
        batchSize = opt.batchSize;
        fileSize = 0;

        subSamplingFactor = opt.subSamplingFactor;
        initialLearningRate = opt.initialLearningRate;
        minLearningRate = opt.minLearningRate;
        temperature = opt.temperature;
        betaDict = opt.betaDict;
        betaReward = opt.betaReward;
        
        senseSelectionOutWeight = sv4d::Matrix(vocab.lemmaVocabSize, embeddingLayerSize * 3);
        senseSelectionOutBias = sv4d::Vector(vocab.lemmaVocabSize);
        embeddingInWeight = sv4d::Matrix(vocab.synsetVocabSize, embeddingLayerSize);
        embeddingOutWeight = sv4d::Matrix(vocab.wordVocabSize, embeddingLayerSize);

        unigramTable = std::vector<int>();
        subsamplingFactorTable = std::vector<float>();
        stopWords = std::unordered_set<int>();

        trainedWordCount = 0;
    }

    void Model::initialize() {
        initializeWeight();
        initializeUnigramTable();
        initializeSubsamplingFactorTable();
        initializeFileSize();
        initializeStopWords();
    }

    void Model::training() {
        startTime = std::chrono::system_clock::now();
        trainedWordCount = 0;
        printf("Training model:  \n");
        if (threadNum > 1) {
            auto threads = std::vector<std::thread>();
            for (int i = 0; i < threadNum; i++) {
                threads.push_back(std::thread(&Model::trainingThread, this, i));
            }
            for (int i = 0; i < threadNum; i++) {
                threads[i].join();
            }
        } else {
            Model::trainingThread(0);
        }
        printf("\n");
    }

    void Model::initializeWeight() {
        embeddingInWeight.setRandomUniform(-0.5 / embeddingLayerSize, 0.5 / embeddingLayerSize);
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
                ++i;
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
            if (vocab.wordFreq[i] == 0) {
                subsamplingFactorTable[i] = 0.0f;
            } else {
                float factor = (std::sqrt(vocab.wordFreq[i] / (subSamplingFactor * vocab.totalWordsNum)) + 1) * (subSamplingFactor * vocab.totalWordsNum) / vocab.wordFreq[i];
                subsamplingFactorTable[i] = factor;
            }
        }
    }

    void Model::initializeFileSize() {
        std::ifstream fin(trainingCorpus);
        if (fin.fail()) {
            throw std::runtime_error("Cannot open training data file");
        }
        fin.seekg(0, fin.end);
        fileSize = fin.tellg();
    }

    void Model::initializeStopWords() {
        std::string linebuf;
        std::ifstream fin(trainingCorpus);
        if (fin.fail()) {
            return;
        }

        while (std::getline(fin, linebuf)) {
            auto word = sv4d::utils::string::trim(linebuf);
            if (vocab.synsetVocab.find(word) == vocab.synsetVocab.end()) {
                continue;
            }

            stopWords.insert(vocab.synsetVocab[word]);
        }
    }

    void Model::trainingThread(const int threadId) {
        std::string linebuf;

        // file
        std::ifstream fin(trainingCorpus);
        if (fin.fail()) {
            throw std::runtime_error("Cannot open training data file");
        }

        // random
        std::mt19937 mt(495 + threadId);
        std::uniform_real_distribution<double> rand(0, 1);
        std::uniform_int_distribution<int> rndwindow(0, windowSize - 1);

        // initialize negative sampling position
        int negativePos = mt() % UnigramTableSize + 1;

        // hyper parameter
        float lr = initialLearningRate;

        // cache
        int processedWordCount = 0;
        auto sentencesCache = std::deque<std::vector<int>>();
        auto sentenceVectorsCache = std::deque<sv4d::Vector>();

        auto outputWidxCandidateCache = std::vector<int>();
        outputWidxCandidateCache.reserve(windowSize * 2);
        auto subSampledCache = std::vector<bool>();
        subSampledCache.reserve(4096);
        auto dictPairPos = std::unordered_map<int, int>();

        sv4d::Vector documentVectorCache = sv4d::Vector(embeddingLayerSize);
        sv4d::Vector contextVectorCache = sv4d::Vector(embeddingLayerSize);
        sv4d::Vector featureVectorCache = sv4d::Vector(embeddingLayerSize * 3);

        for (int iter = 0; iter < epochs; ++iter) {
            fin.clear();
            fin.seekg(fileSize / threadNum * threadId, fin.beg);
            // seek to head of sentence
            std::getline(fin, linebuf);

            sentencesCache.clear();
            sentenceVectorsCache.clear();

            // generate batch and process
            while (true) {
                if (fin.tellg() > fileSize / threadNum * (threadId + 1)) {
                    break;
                }

                // clear cache
                processedWordCount = 0;

                // cache sentence for faster calculation
                bool bod = false; // begin of document
                bool eod = false; // end of document
                while (std::getline(fin, linebuf)) {
                    linebuf = sv4d::utils::string::trim(linebuf);
                    if (linebuf == "<doc>") {
                        bod = true;
                        continue;
                    } else if (linebuf == "") {
                        continue;
                    } else if (linebuf == "</doc>") {
                        eod = true;
                        break;
                    } else {
                        auto sentence = std::vector<int>();
                        for (auto word : sv4d::utils::string::split(linebuf, ' ')) {
                            
                            if (vocab.synsetVocab.find(word) == vocab.synsetVocab.end()) {
                                continue;
                            }
                            int widx = vocab.synsetVocab[word];
                            if (vocab.wordFreq[widx] == 0) {
                                continue;
                            }
                            sentence.push_back(widx);
                        }
                        processedWordCount += sentence.size();
                        if (sentence.size() >= 5) {
                            sentencesCache.push_back(sentence);

                            sv4d::Vector sentenceVector = sv4d::Vector(embeddingLayerSize);
                            for (auto d : sentence) {
                                sv4d::Vector& embeddingInVector = embeddingInWeight[d];
                                sentenceVector += embeddingInVector;
                            }
                            sentenceVector /= (int)sentence.size();
                            sentenceVectorsCache.push_back(sentenceVector);
                        }
                        if (sentencesCache.size() > batchSize) {
                            break;
                        }
                        if (fin.tellg() > fileSize / threadNum * (threadId + 1)) {
                            break;
                        }
                    }
                }

                int sentenceCount = sentencesCache.size();
                if (sentenceCount == 0) {
                    continue;
                }

                int targetSentenceCount = sentenceCount - (!eod);
                // process batch
                for (int r = 0 + (!bod); r < targetSentenceCount; ++r) {
                    auto& sentence = sentencesCache[r];
                    int sentenceSize = sentence.size();

                    subSampledCache.clear();
                    for (auto d : sentence) {
                        subSampledCache.push_back(subsamplingFactorTable[d] < rand(mt));
                    }

                    // document vector
                    documentVectorCache.setZero();
                    int minSentPos = r - 1 < 0 ? 0 : r - 1;
                    int maxSentPos = r + 1 > sentenceCount ? sentenceCount : r + 1;
                    for (int i = minSentPos; i < maxSentPos; ++i) {
                        documentVectorCache += sentenceVectorsCache[i];
                    }
                    documentVectorCache /= (maxSentPos - minSentPos);

                    // sentence vector
                    sv4d::Vector& sentenceVectorCache = sentenceVectorsCache[r];

                    for (int pos = 0; pos < sentenceSize; ++pos) {
                        if (subSampledCache[pos]) {
                            continue;
                        }

                        // output widx
                        outputWidxCandidateCache.clear();
                        int reducedWindowSize = windowSize - rndwindow(mt);
                        for (int pos2 = pos - 1, count = reducedWindowSize; pos2 >= 0 && count != 0; --pos2) {
                            if (subSampledCache[pos2]) {
                                continue;
                            }
                            outputWidxCandidateCache.push_back(sentence[pos2]);
                            count -= 1;
                        }
                        for (int pos2 = pos + 1, count = reducedWindowSize; pos2 < sentenceSize && count != 0; ++pos2) {
                            if (subSampledCache[pos2]) {
                                continue;
                            }
                            outputWidxCandidateCache.push_back(sentence[pos2]);
                            count -= 1;
                        }
                        if (outputWidxCandidateCache.size() == 0) {
                            continue;
                        }
                        int outputWidx = outputWidxCandidateCache[mt() % outputWidxCandidateCache.size()];

                        // context vector
                        contextVectorCache.setZero();
                        int minPos = pos - windowSize < 0 ? 0 : pos - windowSize;
                        int maxPos = pos + windowSize >= sentenceSize ? sentenceSize - 1 : pos + windowSize;
                        for (int pos2 = minPos; pos2 <= maxPos; ++pos2) {
                            if (pos == pos2) {
                                continue;
                            }
                            sv4d::Vector& embeddingInVector = embeddingInWeight[sentence[pos2]];
                            contextVectorCache += embeddingInVector;
                        }
                        contextVectorCache /= (maxPos - minPos - 1);

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
                            sv4d::Vector embeddingInBufVector = sv4d::Vector(embeddingLayerSize);
                            sv4d::Vector embeddingOutBufVector = sv4d::Vector(embeddingLayerSize);

                            sv4d::SynsetData& synsetData = vocab.widx2lidxs[inputWidx];

                            sv4d::Vector& vWordOut = embeddingOutWeight[outputWidx];
                            sv4d::Vector& vWordOutIn = embeddingInWeight[outputWidx];

                            // sense training
                            if (synsetData.validPos.size() != 0) {
                                // pos selection (random)
                                int targetPos = synsetData.validPos[mt() % synsetData.validPos.size()];
                                auto& synsetLemmaIndices = synsetData.synsetLemmaIndices[targetPos];

                                // sense selection
                                int senseNum = synsetLemmaIndices.size();
                                sv4d::Vector senseSelectionLogits = sv4d::Vector(senseNum);
                                for (int i = 0; i < senseNum; ++i) {
                                    int lidx = synsetLemmaIndices[i];
                                    senseSelectionLogits[i] = (featureVectorCache % senseSelectionOutWeight[lidx]) + senseSelectionOutBias[lidx];
                                }
                                sv4d::Vector senseSelectionProbTemperature = senseSelectionLogits.softmax(temperature);

                                sv4d::Vector rewardLogits = sv4d::Vector(senseNum);

                                // embedding
                                for (int i = 0; i < senseNum; ++i) {
                                    embeddingInBufVector.setZero();

                                    float senseWeight = senseSelectionProbTemperature[i];

                                    int sidx = vocab.lidx2sidx[synsetLemmaIndices[i]];
                                    sv4d::SynsetDictPair& synsetDictPair = vocab.synsetDictPair[sidx];
                                    auto& dictPair = synsetDictPair.dictPair;

                                    sv4d::Vector& vSynsetIn = embeddingInWeight[sidx];

                                    // Positive: example predicts label.
                                    //   forward: x = v_in' * v_out
                                    //            l = log(sigmoid(x))
                                    //   backward: dl/dx = g = sigmoid(-x)
                                    //             dl/d(v_in) = g * v_out'
                                    //             dl/d(v_out) = v_in' * g
                                    {
                                        float dot = vSynsetIn % vWordOut;
                                        float g = sv4d::utils::operation::sigmoid(-dot);
                                        float w = g * lr * senseWeight;
                                        // embeddingInBufVector += vWordOut * w;
                                        // embeddingOutBufVector += vSynsetIn * w;
                                        embeddingInBufVector.fusedMultiplyAdd(vWordOut, w);
                                        embeddingOutBufVector.fusedMultiplyAdd(vSynsetIn, w);
                                    }

                                    // Negative samples:
                                    //   forward: x = v_in' * v_sample
                                    //            l = log(sigmoid(-x))
                                    //   backward: dl/dx = g = -sigmoid(x)
                                    //             dl/d(v_in) = g * v_out'
                                    //             dl/d(v_out) = v_in' * g
                                    for (int j = 0; j < negativeSample; ++j) {
                                        int sample = unigramTable[--negativePos];
                                        if (negativePos == 0) {
                                            negativePos = UnigramTableSize;
                                        }
                                        if (sample == outputWidx) {
                                            continue;
                                        }
                                        if (std::find(dictPair.begin(), dictPair.end(), sample) != dictPair.end()) {
                                            continue;
                                        } 
                                        sv4d::Vector& vSample = embeddingOutWeight[sample];
                                        float dot = vSynsetIn % vSample;
                                        float g = -sv4d::utils::operation::sigmoid(dot);
                                        float w = g * lr * senseWeight;
                                        // embeddingInBufVector += vSample * w;
                                        // vSample += vSynsetIn * w;
                                        embeddingInBufVector.fusedMultiplyAdd(vSample, w);
                                        vSample.fusedMultiplyAdd(vSynsetIn, w);
                                    }

                                    vSynsetIn += embeddingInBufVector;
                                }

                                for (int i = 0; i < senseNum; ++i) {
                                    int sidx = vocab.lidx2sidx[synsetLemmaIndices[i]];
                                    sv4d::SynsetDictPair& synsetDictPair = vocab.synsetDictPair[sidx];
                                    auto& dictPair = synsetDictPair.dictPair;

                                    // Positive: dictionary pairs for accurate prediction
                                    //   forward: x = v_in' * v_out
                                    //            l = log(sigmoid(x))
                                    //   backward: dl/dx = g = sigmoid(-x)
                                    //             dl/d(v_in) = g * v_out'
                                    //             dl/d(v_out) = v_in' * g
                                    for (int j = 0; j < dictSample; ++j) {
                                        if (dictPair.size() == 0) {
                                            break;
                                        }
                                        int& dpos = dictPairPos[sidx];
                                        int sample = dictPair[dpos];
                                        if (dpos == dictPair.size() - 1) {
                                            dpos = 0;
                                        } else {
                                            dpos += 1;
                                        }
                                        sv4d::Vector& vSample = embeddingOutWeight[sample];

                                        float maxDot = -10000.0f;
                                        for (int pos2 = pos - 1, count = reducedWindowSize; pos2 >= 0 && count != 0; --pos2) {
                                            if (subSampledCache[pos2]) {
                                                continue;
                                            }
                                            maxDot = std::max(embeddingInWeight[sentence[pos2]] % vSample, maxDot);
                                            count -= 1;
                                        }
                                        for (int pos2 = pos + 1, count = reducedWindowSize; pos2 < sentenceSize && count != 0; ++pos2) {
                                            if (subSampledCache[pos2]) {
                                                continue;
                                            }
                                            maxDot = std::max((embeddingInWeight[sentence[pos2]] % vSample), maxDot);
                                            count -= 1;
                                        }

                                        rewardLogits[i] += maxDot * betaReward;
                                    }
                                }

                                if (stopWords.find(outputWidx) == stopWords.end()) {
                                    // sense selection (update)
                                    sv4d::Vector rewardProb = rewardLogits.softmax(1.0);
                                    sv4d::Vector senseSelectionProb = senseSelectionLogits.softmax(1.0);

                                    // Update sense selection weight.
                                    //   forward: x = v_feature' * v_sense_selection + v_sense_bias
                                    //            l = Σz * log(softmax(x))
                                    //   backward: dl/dx = g = z * (1 - x) - Σz * x
                                    //             dl/d(v_sense_selection) = v_feature' * g
                                    //             dl/d(v_sense_bias) = g
                                    for (int i = 0; i < senseNum; ++i) {
                                        int lidx = synsetLemmaIndices[i];
                                        float g = rewardProb[i] - senseSelectionProb[i];
                                        sv4d::Vector& vSenseSelection = senseSelectionOutWeight[lidx];
                                        float& bSenseSelection = senseSelectionOutBias[lidx];
                                        float w = g * lr;
                                        // vSenseSelection += featureVectorCache * w;
                                        // bSenseSelection += w;
                                        vSenseSelection.fusedMultiplyAdd(featureVectorCache, w);
                                        bSenseSelection += w;
                                    }
                                }
                            }

                            // word training
                            {
                                embeddingInBufVector.setZero();

                                int wsidx = vocab.lidx2sidx[synsetData.wordLemmaIndex];

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
                                    float w = g * lr;
                                    // embeddingInBufVector += vWordOut * w;
                                    // embeddingOutBufVector += vWordIn * w;
                                    embeddingInBufVector.fusedMultiplyAdd(vWordOut, w);
                                    embeddingOutBufVector.fusedMultiplyAdd(vWordIn, w);
                                }

                                // Negative samples:
                                //   forward: x = v_in' * v_sample
                                //            l = log(sigmoid(-x))
                                //   backward: dl/dx = g = -sigmoid(x)
                                //             dl/d(v_in) = g * v_out'
                                //             dl/d(v_out) = v_in' * g
                                for (int j = 0; j < negativeSample; ++j) {
                                    int sample = unigramTable[--negativePos];
                                    if (negativePos == 0) {
                                        negativePos = UnigramTableSize;
                                    }
                                    if (sample == outputWidx) {
                                        continue;
                                    }
                                    sv4d::Vector& vSample = embeddingOutWeight[sample];
                                    float dot = vWordIn % vSample;
                                    float g = -sv4d::utils::operation::sigmoid(dot);
                                    float w = g * lr;
                                    // embeddingInBufVector += vSample * w;
                                    // vSample += vWordIn * w;
                                    embeddingInBufVector.fusedMultiplyAdd(vSample, w);
                                    vSample.fusedMultiplyAdd(vWordIn, w);
                                }

                                vWordIn += embeddingInBufVector;
                            }

                            vWordOut += embeddingOutBufVector;
                        }
                    }
                }

                if (eod) {
                    sentencesCache.clear();
                    sentenceVectorsCache.clear();
                } else {
                    for (int i = 0; i < sentenceCount - 2; ++i) {
                        sentencesCache.pop_front();
                        sentenceVectorsCache.pop_front();
                    }
                }

                trainedWordCount += processedWordCount;

                // change hyper parameter
                float progress = trainedWordCount / (float)(epochs * vocab.totalWordsNum + 1);
                lr = (initialLearningRate - minLearningRate) * (1.0f - progress) + minLearningRate;

                // print log
                auto now = std::chrono::system_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
                float speed = trainedWordCount / (float)((elapsed + 1) * threadNum);
                float eta = ((float)(epochs * vocab.totalWordsNum) / (trainedWordCount + 1) * elapsed - elapsed) / 60000.0f;
                printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  Remaining: %.2fm  ", 13, lr, progress * 100.0f, speed, eta);
                fflush(stdout);
            }
        }
    }

    void Model::wordNearestNeighbour() {
        sv4d::Matrix normedEmbeddingInWeight = sv4d::Matrix(embeddingInWeight);
        for (int i = 0; i < normedEmbeddingInWeight.row; ++i) {
            float norm = (normedEmbeddingInWeight[i] * normedEmbeddingInWeight[i]).sum();
            normedEmbeddingInWeight[i] /= std::sqrt(norm);
        }

        while (true) {
            printf("Enter word (EXIT to break): ");
            std::string word = "";
            while (1) {
                char c = fgetc(stdin);
                if (c == '\n') {
                    break;
                }
                word += c;
            }
            if (word == "EXIT") {
                break;
            }
            if (vocab.synsetVocab.find(word) == vocab.synsetVocab.end()) {
                printf("Out of dictionary word!\n");
                continue;
            }
            int widx = vocab.synsetVocab[word];
            auto& wordVector = normedEmbeddingInWeight[widx];
            auto similarities = std::vector<std::pair<int, float>>();
            for (int i = 0; i < normedEmbeddingInWeight.row; ++i) {
                if (i == widx) {
                    continue;
                }
                std::pair<int, float> pair;
                pair.first = i;
                pair.second = wordVector % normedEmbeddingInWeight[i];
                similarities.push_back(pair);
            }
            std::sort(similarities.begin(), similarities.end(), [](const std::pair<int, float> & a, const std::pair<int, float> & b) -> bool { return a.second > b.second; });
            printf("Word %d: %s", widx, vocab.sidx2Synset[widx].c_str());
            printf("\n                                              Word       Cosine distance\n------------------------------------------------------------------------\n");
            for (int i = 0; i < 40; ++i) {
                printf("%50s\t\t%f\n", vocab.sidx2Synset[similarities[i].first].c_str(), similarities[i].second);
            }
            printf("\n");
        }
    }

    void Model::synsetNearestNeighbour() {
        sv4d::Matrix normedEmbeddingInWeight = sv4d::Matrix(embeddingInWeight);
        for (int i = 0; i < normedEmbeddingInWeight.row; ++i) {
            float norm = (normedEmbeddingInWeight[i] * normedEmbeddingInWeight[i]).sum();
            normedEmbeddingInWeight[i] /= std::sqrt(norm);
        }

        while (true) {
            printf("Enter word (EXIT to break): ");
            std::string word = "";
            while (1) {
                char c = fgetc(stdin);
                if (c == '\n') {
                    break;
                }
                word += c;
            }
            if (word == "EXIT") {
                break;
            }
            if (vocab.synsetVocab.find(word) == vocab.synsetVocab.end()) {
                printf("Out of dictionary word!\n");
                continue;
            }
            int widx = vocab.synsetVocab[word];
            for (int pos : {sv4d::Pos::Noun, sv4d::Pos::Verb, sv4d::Pos::Adjective, sv4d::Pos::Adverb}) {
                if (std::find(vocab.widx2lidxs[widx].validPos.begin(), vocab.widx2lidxs[widx].validPos.end(), pos) == vocab.widx2lidxs[widx].validPos.end()) {
                    continue;
                }
                auto lemmas = vocab.widx2lidxs[widx].synsetLemmaIndices[pos];
                std::sort(lemmas.begin(), lemmas.end());
                for (int lidx : lemmas) {
                    int sidx = vocab.lidx2sidx[lidx];
                    auto synsetVector = normedEmbeddingInWeight[sidx];

                    auto similarities = std::vector<std::pair<int, float>>();
                    for (int i = 0; i < normedEmbeddingInWeight.row; ++i) {
                        if (i == sidx) {
                            continue;
                        }
                        std::pair<int, float> pair;
                        pair.first = i;
                        pair.second = synsetVector % normedEmbeddingInWeight[i];
                        similarities.push_back(pair);
                    }

                    std::sort(similarities.begin(), similarities.end(), [](const std::pair<int, float> & a, const std::pair<int, float> & b) -> bool { return a.second > b.second; });
                    printf("Synset %d: %s", sidx, vocab.sidx2Synset[sidx].c_str());
                    printf("\n                                              Word       Cosine distance\n------------------------------------------------------------------------\n");
                    for (int i = 0; i < 20; ++i) {
                        printf("%50s\t\t%f\n", vocab.sidx2Synset[similarities[i].first].c_str(), similarities[i].second);
                    }
                    printf("\n");
                }
            }
        }
    }

    void Model::saveEmbeddingInWeight(const std::string& filepath, bool binary) {
        std::ofstream fout(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
        if (fout.fail()) {
            throw std::runtime_error("Cannot open weight file");
        }
        fout << vocab.synsetVocabSize << " " << embeddingLayerSize << "\n";
        for (int sidx = 0; sidx < vocab.synsetVocabSize; ++sidx) {
            auto synset = vocab.sidx2Synset[sidx];
            fout << synset << " ";
            auto& vector = embeddingInWeight[sidx].getData();
            if (binary) {
                for (int i = 0; i < embeddingLayerSize; ++i) {
                    fout.write((char *)&vector[i], sizeof(float));
                }
            } else {
                fout << sv4d::utils::string::join(sv4d::utils::string::floatvec_to_strvec(vector), ' ');
            }
            fout << "\n";
        }
        fout.close();
    }

    void Model::loadEmbeddingInWeight(const std::string& filepath, bool binary) {
        std::string linebuf;
        std::ifstream fin(filepath, std::ios::in | std::ios::binary);
        if (fin.fail()) {
            throw std::runtime_error("Cannot open weight file");
        }
        std::getline(fin, linebuf);
        auto sizes = sv4d::utils::string::split(sv4d::utils::string::trim(linebuf), ' ');
        int synsetVocabSize = std::stoi(sizes[0]);
        int embeddingLayerSize = std::stoi(sizes[1]);
        for (int i = 0; i < synsetVocabSize; ++i) {
            std::getline(fin, linebuf, ' ');
            auto synset = sv4d::utils::string::trim(linebuf);
            if (vocab.synsetVocab.find(synset) == vocab.synsetVocab.end()) {
                continue;
            }
            auto& vector = embeddingInWeight[vocab.synsetVocab[synset]];
            if (binary) {
                float value;
                for (int i = 0; i < embeddingLayerSize; ++i) {
                    fin.read((char *)&value, sizeof(float));
                    vector[i] = value;
                }
                fin.seekg(sizeof(char), fin.cur);
            } else {
                std::getline(fin, linebuf, '\n');
                auto strvec = sv4d::utils::string::split(sv4d::utils::string::trim(linebuf), ' ');
                for (int i = 0; i < embeddingLayerSize; ++i) {
                    vector[i] = std::stof(strvec[i]);
                }
            }
        }
    }

    void Model::saveEmbeddingOutWeight(const std::string& filepath, bool binary) {
        std::ofstream fout(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
        if (fout.fail()) {
            throw std::runtime_error("Cannot open weight file");
        }
        fout << vocab.wordVocabSize << " " << embeddingLayerSize << "\n";
        for (int widx = 0; widx < vocab.wordVocabSize; ++widx) {
            auto word = vocab.sidx2Synset[widx];
            fout << word << " ";
            auto& vector = embeddingOutWeight[widx].getData();
            if (binary) {
                for (int i = 0; i < embeddingLayerSize; ++i) {
                    fout.write((char *)&vector[i], sizeof(float));
                }
            } else {
                fout << sv4d::utils::string::join(sv4d::utils::string::floatvec_to_strvec(vector), ' ');
            }
            fout << "\n";
        }
        fout.close();
    }

    void Model::loadEmbeddingOutWeight(const std::string& filepath, bool binary) {
        std::string linebuf;
        std::ifstream fin(filepath, std::ios::in | std::ios::binary);
        if (fin.fail()) {
            throw std::runtime_error("Cannot open weight file");
        }
        std::getline(fin, linebuf);
        auto sizes = sv4d::utils::string::split(sv4d::utils::string::trim(linebuf), ' ');
        int wordVocabSize = std::stoi(sizes[0]);
        int embeddingLayerSize = std::stoi(sizes[1]);
        for (int i = 0; i < wordVocabSize; ++i) {
            std::getline(fin, linebuf, ' ');
            auto word = sv4d::utils::string::trim(linebuf);
            if (vocab.synsetVocab.find(word) == vocab.synsetVocab.end()) {
                continue;
            }
            auto& vector = embeddingOutWeight[vocab.synsetVocab[word]];
            if (binary) {
                float value;
                for (int i = 0; i < embeddingLayerSize; ++i) {
                    fin.read((char *)&value, sizeof(float));
                    vector[i] = value;
                }
                fin.seekg(sizeof(char), fin.cur);
            } else {
                std::getline(fin, linebuf, '\n');
                auto strvec = sv4d::utils::string::split(sv4d::utils::string::trim(linebuf), ' ');
                for (int i = 0; i < embeddingLayerSize; ++i) {
                    vector[i] = std::stof(strvec[i]);
                }
            }
        }
    }

    void Model::saveSenseSelectionOutWeight(const std::string& filepath, bool binary) {
        std::ofstream fout(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
        if (fout.fail()) {
            throw std::runtime_error("Cannot open weight file");
        }
        fout << vocab.lemmaVocabSize << " " << (embeddingLayerSize * 3) << "\n";
        for (int lidx = 0; lidx < vocab.lemmaVocabSize; ++lidx) {
            auto lemma = vocab.lidx2Lemma[lidx];
            fout << lemma << " ";
            auto& vector = senseSelectionOutWeight[lidx].getData();
            if (binary) {
                for (int i = 0; i < embeddingLayerSize * 3; ++i) {
                    fout.write((char *)&vector[i], sizeof(float));
                }
            } else {
                fout << sv4d::utils::string::join(sv4d::utils::string::floatvec_to_strvec(vector), ' ');
            }
            fout << "\n";
        }
        fout.close();
    }

    void Model::loadSenseSelectionOutWeight(const std::string& filepath, bool binary) {
        std::string linebuf;
        std::ifstream fin(filepath, std::ios::in | std::ios::binary);
        if (fin.fail()) {
            throw std::runtime_error("Cannot open weight file");
        }
        std::getline(fin, linebuf);
        auto sizes = sv4d::utils::string::split(sv4d::utils::string::trim(linebuf), ' ');
        int lemmaVocabSize = std::stoi(sizes[0]);
        int embeddingLayerSize = std::stoi(sizes[1]);
        for (int i = 0; i < lemmaVocabSize; ++i) {
            std::getline(fin, linebuf, ' ');
            auto lemma = sv4d::utils::string::trim(linebuf);
            if (vocab.lemmaVocab.find(lemma) == vocab.lemmaVocab.end()) {
                continue;
            }
            auto& vector = senseSelectionOutWeight[vocab.lemmaVocab[lemma]];
            if (binary) {
                float value;
                for (int i = 0; i < embeddingLayerSize; ++i) {
                    fin.read((char *)&value, sizeof(float));
                    vector[i] = value;
                }
                fin.seekg(sizeof(char), fin.cur);
            } else {
                std::getline(fin, linebuf, '\n');
                auto strvec = sv4d::utils::string::split(sv4d::utils::string::trim(linebuf), ' ');
                for (int i = 0; i < embeddingLayerSize; ++i) {
                    vector[i] = std::stof(strvec[i]);
                }
            }
        }
    }

    void Model::saveSenseSelectionBiasWeight(const std::string& filepath, bool binary) {
        std::ofstream fout(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
        if (fout.fail()) {
            throw std::runtime_error("Cannot open weight file");
        }
        fout << vocab.lemmaVocabSize << " " << 1 << "\n";
        for (int lidx = 0; lidx < vocab.lemmaVocabSize; ++lidx) {
            auto lemma = vocab.lidx2Lemma[lidx];
            fout << lemma << " ";
            auto& value = senseSelectionOutBias[lidx];
            if (binary) {
                fout.write((char *)&value, sizeof(float));
            } else {
                fout << value;
            }
            fout << "\n";
        }
        fout.close();
    }

    void Model::loadSenseSelectionBiasWeight(const std::string& filepath, bool binary) {
        std::string linebuf;
        std::ifstream fin(filepath, std::ios::in | std::ios::binary);
        if (fin.fail()) {
            throw std::runtime_error("Cannot open weight file");
        }
        std::getline(fin, linebuf);
        auto sizes = sv4d::utils::string::split(sv4d::utils::string::trim(linebuf), ' ');
        int lemmaVocabSize = std::stoi(sizes[0]);
        for (int i = 0; i < lemmaVocabSize; ++i) {
            std::getline(fin, linebuf, ' ');
            auto lemma = sv4d::utils::string::trim(linebuf);
            if (vocab.lemmaVocab.find(lemma) == vocab.lemmaVocab.end()) {
                continue;
            }
            auto& value = senseSelectionOutBias[vocab.lemmaVocab[lemma]];
            if (binary) {
                fin.read((char *)&value, sizeof(float));
                fin.seekg(sizeof(char), fin.cur);
            } else {
                std::getline(fin, linebuf, '\n');
                value = std::stof(sv4d::utils::string::trim(linebuf));
            }
        }
    }

}