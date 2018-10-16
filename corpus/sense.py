# -*- coding: utf-8 -*-

import sys
import json
import numpy as np
import scipy as sp
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.parse.corenlp import CoreNLPParser


use_stopwords = True
use_glossdisambiguated = True


class SynsetGloss(object):
    def __init__(self):
        self.synset_gloss = {}
        self.tokenizer = CoreNLPParser(url='http://localhost:42636')

        self.use_babelnet = True
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
            synset_id = 'wn:{}{}'.format(str(synset.offset()).zfill(8), synset.pos())
            definition = self.sense.getGlossByWnSynsetId(synset_id)
            if not definition:
                definition = synset.definition()
        else:
            definition = synset.definition()
        return [x.lower() for x in self.tokenizer.tokenize(definition)]


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
        synset_id = 'wn:{}{}'.format(str(synset.offset()).zfill(8), synset.pos())
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
synset_gloss_relation = SynsetGlossRelation()


def get_synsets_pair(word, synsets, stwords, vocab):
    gloss_data = {}
    for synset in synsets:
        gloss_data[synset.name()] = {}

        gloss_words = synset_gloss[synset.name()]
        for gloss_word in gloss_words:
            if gloss_word not in gloss_data[synset.name()]:
                gloss_data[synset.name()][gloss_word] = {
                    "freq": 1,
                    "graph_distance": 0,
                }
            else:
                gloss_data[synset.name()][gloss_word]["freq"] += 1

        related_synsets = synset.also_sees() \
                          + synset.attributes() \
                          + synset.causes() \
                          + synset.entailments() \
                          + synset.hyponyms() \
                          + synset.hypernyms() \
                          + synset.instance_hypernyms() \
                          + synset.instance_hyponyms() \
                          + synset.member_meronyms() \
                          + synset.member_holonyms() \
                          + synset.part_holonyms() \
                          + synset.part_meronyms() \
                          + synset.region_domains() \
                          + synset.substance_meronyms() \
                          + synset.substance_holonyms() \
                          + synset.topic_domains() \
                          + synset.usage_domains() \
                          + synset.verb_groups()

        for lemma in synset.lemmas():
            related_synsets += [x.synset() for x in lemma.derivationally_related_forms()]
            related_synsets += [x.synset() for x in lemma.pertainyms()]

        if use_glossdisambiguated:
            related_synsets += synset_gloss_relation[synset.name()]

        unique_related_synsets = []
        for related_synset in related_synsets:
            if related_synset.name() in [x.name() for x in unique_related_synsets]:
                continue
            unique_related_synsets.append(related_synset)

        expanded_gloss_words = []
        for related_synset in unique_related_synsets:
            expanded_gloss_words += synset_gloss[related_synset.name()]

        for gloss_word in expanded_gloss_words:
            if gloss_word not in gloss_data[synset.name()]:
                gloss_data[synset.name()][gloss_word] = {
                    "freq": 1,
                    "graph_distance": 1,
                }
            else:
                gloss_data[synset.name()][gloss_word]["freq"] += 1

        for gloss_word in [x.name() for x in synset.lemmas()]:
            if gloss_word not in gloss_data[synset.name()]:
                gloss_data[synset.name()][gloss_word] = {
                    "freq": 1,
                    "graph_distance": 1,
                }
            else:
                gloss_data[synset.name()][gloss_word]["freq"] += 1
    
    for synset in synsets:
        for gloss_word in gloss_data[synset.name()].keys():
            igf = 1 + np.log2(len(synsets) / sum([gloss_word in gloss_data[x.name()] for x in synsets]))
            graph_distance_weight = (1 / (1 + gloss_data[synset.name()][gloss_word]["graph_distance"]))
            weight = gloss_data[synset.name()][gloss_word]["freq"] * igf * graph_distance_weight
            gloss_data[synset.name()][gloss_word]["weight"] = weight
    
    synsets_data = []
    for synset in synsets:
        synset_words = set([x.name() for x in synset.lemmas()] + [word])
        sorted_pair = [k for k, v in sorted(gloss_data[synset.name()].items(), key=lambda x: x[1]["weight"], reverse=True)
                       if k not in stwords and k in vocab and v["weight"] > 0.0 and k not in synset_words]
        dict_pair = sorted_pair[:25]
        synsets_data.append({"name": synset.name(), "dict_pair": dict_pair})
    
    return synsets_data


def main():
    stwords = set(stopwords.words('english'))
    
    vocab = set()
    for line in open(sys.argv[1]):
        vocab.add(line.rstrip())

    with open(sys.argv[2], "w") as fout:
        for line in open(sys.argv[1]):
            word = line.rstrip()
            synsets = wn.synsets(word)
            data = defaultdict(list)
            for synset in synsets:
                pos = synset.pos()
                if pos == "s":
                    pos = "a"
                
                if synset in data[pos]:
                    continue
                
                data[pos].append(synset)
            
            for pos, synsets in data.items():
                data[pos] = get_synsets_pair(word, synsets, stwords, vocab)

            for pos, synsets_data in data.items():
                total_freq = 0
                for synset_data in synsets_data:
                    lemmas = [x for x in wn.synset(synset_data["name"]).lemmas() if x.name() == word]
                    if len(lemmas) == 0:
                        count = 0
                    else:
                        count = lemmas[0].count()
                    total_freq += count
                
                for synset_data in synsets_data:
                    lemmas = [x for x in wn.synset(synset_data["name"]).lemmas() if x.name() == word]
                    if len(lemmas) == 0:
                        count = 0
                    else:
                        count = lemmas[0].count()
                    synset_data["prob"] = (count + 1) / (total_freq + len(synsets_data))

            for pos, synsets_data in data.items():
                if len([x for x in synsets_data if len(x["dict_pair"]) >= 3]) == 0:
                    synset_data = synsets_data[0]
                    print(word + "|" + pos + "|" + synset_data["name"], "{:.8f}".format(synset_data["prob"]), ",".join(synset_data["dict_pair"]), sep=" ", file=fout)
                else:
                    for synset_data in synsets_data:
                        if len(synset_data["dict_pair"]) < 3:
                            continue
                        print(word + "|" + pos + "|" + synset_data["name"], "{:.8f}".format(synset_data["prob"]), ",".join(synset_data["dict_pair"]), sep=" ", file=fout)

            print(word)


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print("usage: python sense.py <vocab_file> <output_file>", file=sys.stderr)
        exit()

    main()