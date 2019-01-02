# -*- coding: utf-8 -*-

import sys
import json
import tqdm
import numpy as np
import scipy as sp
import pickle
from more_itertools import flatten
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.parse.corenlp import CoreNLPParser
from gensim.models.keyedvectors import KeyedVectors


use_extended_gloss = True
use_glossdisambiguated = True
exclude_synset_lemma = False


class SynsetGloss(object):
    def __init__(self):
        self.synset_gloss = {}
        self.tokenizer = CoreNLPParser(url='http://localhost:42636')

        self.use_babelnet = use_extended_gloss
        if self.use_babelnet:
            from py4j.java_gateway import JavaGateway
            gateway = JavaGateway() 
            self.sense = gateway.entry_point

    def __getitem__(self, name):
        if name not in self.synset_gloss:
            self.synset_gloss[name] = self.get_synset_gloss(name)
        
        return self.synset_gloss[name]
    
    def get_synset_gloss(self, name):
        synset = wn.synset(name)
        if self.use_babelnet:
            synset_pos = synset.pos()
            if synset_pos == "s":
                synset_pos = "a"
            synset_id = 'wn:{}{}'.format(str(synset.offset()).zfill(8), synset_pos)
            definition = self.sense.getGlossByWnSynsetId(synset_id)
            if not definition:
                definition = synset.definition()
        else:
            definition = synset.definition()
        return [x.lower() for x in self.tokenizer.tokenize(definition)]


class SynsetExample(object):
    def __init__(self):
        self.synset_example = {}
        self.tokenizer = CoreNLPParser(url='http://localhost:42636')

        self.use_babelnet = use_extended_gloss
        if self.use_babelnet:
            from py4j.java_gateway import JavaGateway
            gateway = JavaGateway() 
            self.sense = gateway.entry_point

    def __getitem__(self, name):
        if name not in self.synset_example:
            self.synset_example[name] = self.get_synset_example(name)
        
        return self.synset_example[name]
    
    def get_synset_example(self, name):
        synset = wn.synset(name)
        if self.use_babelnet:
            synset_pos = synset.pos()
            if synset_pos == "s":
                synset_pos = "a"
            synset_id = 'wn:{}{}'.format(str(synset.offset()).zfill(8), synset_pos)
            example = self.sense.getExampleByWnSynsetId(synset_id)
            if not example:
                example = " ".join(synset.examples()).strip()
        else:
            example = " ".join(synset.examples()).strip()
        return [x.lower() for x in self.tokenizer.tokenize(example)]


class SynsetGlossRelation(object):
    def __init__(self):
        self.synset_gloss_relation = {}
        from py4j.java_gateway import JavaGateway
        gateway = JavaGateway() 
        self.sense = gateway.entry_point

    def __getitem__(self, name):
        if name not in self.synset_gloss_relation:
            self.synset_gloss_relation[name] = self.get_gloss_relation(name)
        
        return self.synset_gloss_relation[name]
    
    def get_gloss_relation(self, name):
        synset = wn.synset(name)
        synset_pos = synset.pos()
        if synset_pos == "s":
            synset_pos = "a"
        synset_id = 'wn:{}{}'.format(str(synset.offset()).zfill(8), synset_pos)
        related_synset_keys = self.sense.getGlossRelatedWordNetSynsetIds(synset_id)
        related_gloss_synsets = []
        for synset_key in related_synset_keys:
            try:
                related_gloss_synset = wn.synset_from_sense_key(synset_key)
                related_gloss_synsets.append(related_gloss_synset)
            except:
                continue
        return related_gloss_synsets


synset_gloss = SynsetGloss()
synset_example = SynsetExample()
synset_gloss_relation = SynsetGlossRelation()


def init_gloss_data(depth=2):
    gloss_data = {}
    
    for synset in tqdm.tqdm(list(wn.all_synsets())):
        gloss_data[synset.name()] = {}

        related_synsets = [synset]
        for d in range(depth):
            gloss_words = []
            example_words = []
            for s in related_synsets:
                gloss_words += synset_gloss[s.name()]
                example_words += synset_example[s.name()]

            for w in gloss_words:
                if w not in gloss_data[synset.name()]:
                    gloss_data[synset.name()][w] = {
                        "freq": 1,
                        "graph_distance": d,
                    }
                else:
                    gloss_data[synset.name()][w]["freq"] += 1

            for w in example_words:
                if w not in gloss_data[synset.name()]:
                    gloss_data[synset.name()][w] = {
                        "freq": 1,
                        "graph_distance": d + 1,
                    }
                else:
                    gloss_data[synset.name()][w]["freq"] += 1

            new_related_synset = []
            for s in related_synsets:
                ns = s.also_sees() \
                     + s.attributes() \
                     + s.causes() \
                     + s.entailments() \
                     + s.hyponyms() \
                     + s.hypernyms() \
                     + s.instance_hypernyms() \
                     + s.instance_hyponyms() \
                     + s.member_meronyms() \
                     + s.member_holonyms() \
                     + s.part_holonyms() \
                     + s.part_meronyms() \
                     + s.region_domains() \
                     + s.substance_meronyms() \
                     + s.substance_holonyms() \
                     + s.topic_domains() \
                     + s.usage_domains() \
                     + s.verb_groups() \
                     + s.similar_tos()

                for l in s.lemmas():
                    ns += [x.synset() for x in l.derivationally_related_forms()]
                    ns += [x.synset() for x in l.pertainyms()]
                
                if use_glossdisambiguated:
                    ns += synset_gloss_relation[s.name()]

                new_related_synset += ns
            
            related_synsets = list(set(new_related_synset))
    
    return gloss_data


def get_synset_pair(synset, stwords, vocab, gloss_data):
    target_synset_names = []
    for lemma in synset.lemmas():
        target_synset_names.extend([x.name() for x in wn.synsets(lemma.name(), pos=synset.pos())])
    target_synset_names = list(set(target_synset_names))

    for w in gloss_data[synset.name()].keys():
        freq = gloss_data[synset.name()][w]["freq"]
        graph_distance = (1 / (1 + gloss_data[synset.name()][w]["graph_distance"]))
        igf = 1 + np.log2(len(target_synset_names) / sum([w in gloss_data[x] for x in target_synset_names]))
        weight = freq * igf * graph_distance
        gloss_data[synset.name()][w]["weight"] = weight
    
    if exclude_synset_lemma:
        synset_words = set([x.name().lower() for x in synset.lemmas()])
    else:
        synset_words = set()

    sorted_pair = [k for k, v in sorted(gloss_data[synset.name()].items(), key=lambda x: x[1]["weight"], reverse=True)
                   if k not in stwords and k in vocab and v["weight"] >= 0.0 and k not in synset_words]
    dict_pair = sorted_pair[:25]
    
    return dict_pair


def main():
    stwords = set(stopwords.words('english'))
    stwords.add("n't")
    stwords.add("un")
    
    vocab = set()
    for line in open(sys.argv[1]):
        vocab.add(line.rstrip())
    
    #gloss_data = init_gloss_data()
    #pickle.dump(gloss_data, open("gloss_data_depth2.pickle", "wb"))
    gloss_data = pickle.load(open("gloss_data_depth2.pickle", "rb"))

    dict_pair_cache = {}

    with open(sys.argv[2], "w") as fout:
        for line in tqdm.tqdm(list(open(sys.argv[1]))):
            for pos in ["n", "v", "a", "r"]:
                lemma_name = wn.morphy(line.rstrip(), pos=pos)
                if lemma_name is None:
                    continue

                freqs = [l.count() for l in wn.lemmas(lemma_name, pos=pos)]
                total_freq = sum(freqs)
                probs = [(freq + 1) / (total_freq + len(freqs)) for freq in freqs]

                for lemma, prob in zip(wn.lemmas(lemma_name, pos=pos), probs):
                    synset = lemma.synset()
                    pos = synset.pos()
                    if pos == "s":
                        pos = "a"

                    if synset.name() in dict_pair_cache:
                        dict_pair = dict_pair_cache[synset.name()]
                    else:
                        dict_pair = get_synset_pair(synset, stwords, vocab, gloss_data)
                        dict_pair_cache[synset.name()] = dict_pair
                    
                    print(line.rstrip() + "|" + pos + "|" + synset.name(), "{:.8f}".format(prob), ",".join(dict_pair), ",".join([x.synset().name() for x in lemma.derivationally_related_forms()]), sep=" ", file=fout)
                    if lemma.name().lower() != lemma.name():
                        print(lemma.name() + "|" + pos + "|" + synset.name(), "{:.8f}".format(prob), ",".join(dict_pair), ",".join([x.synset().name() for x in lemma.derivationally_related_forms()]), sep=" ", file=fout)

if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print("usage: python sense.py <vocab_file> <output_file>", file=sys.stderr)
        exit()

    main()