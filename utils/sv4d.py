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


if __name__ == "__main__":
    model = Model("./models/default/")
    model.load_vocab()
    model.load_weight()