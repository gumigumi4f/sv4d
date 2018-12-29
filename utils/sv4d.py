import os
import sys
import struct
import numpy as np
import gensim


class Model:
    def __init__(self, model_dir):
        self.model_dir = model_dir

        self.lemma_vocab_size = 0
        self.synset_vocab_size = 0
        self.word_vocab_size = 0

        self.total_words_num = 0
        self.total_sentence_num = 0
        self.total_document_num = 0

        self.lemma_vocab = {}
        self.synset_vocab = {}

        self.lidx2lemma = {}
        self.sidx2synset = {}

        self.widx2lidxs = {}
        self.lidx2sidx = {}

        self.lemma_prob = {}

    def __getitem__(self, word):
        if word not in self.synset_vocab:
            raise ValueError("Word %s is not in vocab" % word)
        return self.embedding_in_weight[self.synset_vocab[word]]

    def load_vocab(self):
        if not self.model_dir:
            raise ValueError("Model directory is empty")

        vocab_file = os.path.join(self.model_dir, "vocab.txt")
        lines = open(vocab_file).readlines()
        self.lemma_vocab_size, self.synset_vocab_size, self.word_vocab_size = lines[0].rstrip().split(" ")
        self.total_words_num, self.total_sentence_num, self.total_document_num = lines[1].rstrip().split(" ")
        for line in lines[2:]:
            lemma, lidx_str, lemma_prob_str, widx_str, _, sidx_str, _ = line.rstrip().split(" ")
            lidx = int(lidx_str)
            lemma_prob = float(lemma_prob_str)
            widx = int(widx_str)
            sidx = int(sidx_str)

            word, pos, synset = lemma.split("|")

            self.lemma_vocab[lemma] = lidx
            self.lidx2lemma[lidx] = lemma
            self.lemma_prob[lidx] = lemma_prob

            if word not in self.synset_vocab:
                self.synset_vocab[word] = widx
                self.sidx2synset[widx] = word
                self.widx2lidxs[widx] = {}
                self.widx2lidxs[widx]["*"] = widx
            
            if synset != "*" and synset not in self.synset_vocab:
                self.synset_vocab[synset] = sidx
                self.sidx2synset[sidx] = synset

            if pos != "*" and synset != "*":
                if pos not in self.widx2lidxs[widx]:
                    self.widx2lidxs[widx][pos] = []
                
                self.widx2lidxs[widx][pos].append(lidx)
            
            self.lidx2sidx[lidx] = sidx

    def load_weight(self):
        if not self.model_dir:
            raise ValueError("Model directory is empty")
        if not self.lemma_vocab:
            raise ValueError("Vocab is not loaded")

        self.embedding_in_weight = self._read_weight_from_file_gensim(os.path.join(self.model_dir, "embedding_in_weight"), self.synset_vocab)
        self.embedding_out_weight = self._read_weight_from_file_gensim(os.path.join(self.model_dir, "embedding_out_weight"), self.synset_vocab)
        self.sense_selection_out_weight = self._read_weight_from_file_gensim(os.path.join(self.model_dir, "sense_selection_out_weight"), self.lemma_vocab)
        self.sense_selection_out_bias = self._read_weight_from_file_gensim(os.path.join(self.model_dir, "sense_selection_out_bias"), self.lemma_vocab)
        self.sense_selection_out_bias = self.sense_selection_out_bias.reshape(-1)
    
    def calculate_sense_probability(self, word, pos, contexts, sentence, document, use_sense_prob=False):
        if word not in self.synset_vocab:
            raise ValueError("Word %s is not found in vocab" % (word))

        synset_data = self.widx2lidxs[self.synset_vocab[word]]
        if pos not in synset_data:
            return (np.ones((1,)), [word])
            # raise ValueError("Pos %s is not found in synset data" % (pos))
        
        contexts = [self.synset_vocab[word] for word in contexts if word in self.synset_vocab]
        sentence = [self.synset_vocab[word] for word in sentence if word in self.synset_vocab]
        document = [[self.synset_vocab[word] for word in s if word in self.synset_vocab] for s in document]

        if len(contexts) != 0:
            contexts_vector = self.embedding_in_weight[contexts].mean(axis=0)
        else:
            contexts_vector = np.zeros((self.embedding_in_weight.shape[1],))
        if len(sentence) != 0:
            sentence_vector = self.embedding_in_weight[sentence].mean(axis=0)
        else:
            sentence_vector = np.zeros((self.embedding_in_weight.shape[1],))
        if len(document) != 0:
            sentences_vector = np.array([self.embedding_in_weight[s].mean(axis=0) for s in document if len(s) != 0])
            document_vector = sentences_vector.mean(axis=0)
        else:
            document_vector = np.zeros((self.embedding_in_weight.shape[1],))

        feature_vector = np.concatenate([contexts_vector, sentence_vector, document_vector])

        synset_logits = []
        for lidx in synset_data[pos]:
            dot = np.dot(feature_vector, self.sense_selection_out_weight[lidx]) + self.sense_selection_out_bias[lidx]
            synset_logits.append(dot)
        
        synset_prob = self._softmax(synset_logits)
        if use_sense_prob:
            synset_prob = synset_prob * np.array([self.lemma_prob[lidx] for lidx in synset_data[pos]])
            synset_prob = synset_prob / np.sum(synset_prob)
        return (synset_prob, [self.sidx2synset[self.lidx2sidx[lidx]] for lidx in synset_data[pos]])

    def _softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y 

    def _read_weight_from_file_gensim(self, filename, data):
        kv = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
        weight = np.zeros(shape=kv.vectors.shape)
        for word, idx in data.items():
            if idx >= weight.shape[0]:
                continue
            weight[idx] = kv[word]
        return weight

    def _read_weight_from_file(self, filename, data):
        weight_data = open(filename, "rb").read()
        offset = 0
        
        # read row size
        row_str = ""
        while True:
            c = struct.unpack_from("c", weight_data, offset)[0].decode('utf-8')
            offset += 1
            if c == " ":
                break
            row_str += c
        row = int(row_str)

        # read col size
        col_str = ""
        while True:
            c = struct.unpack_from("c", weight_data, offset)[0].decode('utf-8')
            offset += 1
            if c == "\n":
                break
            col_str += c
        col = int(col_str)

        weight = np.zeros(shape=(row, col))

        for _ in range(row):
            # read row name
            row_name = ""
            while True:
                c = struct.unpack_from("c", weight_data, offset)[0].decode('utf-8')
                offset += 1
                if c == " ":
                    break
                row_name += c
            
            idx = data[row_name]

            for i in range(col):
                value = struct.unpack_from("f", weight_data, offset)[0]
                offset += 4
                weight[idx, i] = value
            
            offset += 1
        
        return weight
