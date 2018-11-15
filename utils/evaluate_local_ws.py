import sys
import numpy as np
import scipy as sp
import gensim
from nltk.corpus import wordnet as wn

from sv4d import Model


def main():
    pos_tags = {"a": "a", "s": "a", "n": "n", "v": "v", "r": "r"}
    target_pos1_col = -1
    target_pos2_col = -1
    if len(sys.argv) >= 8:
        target_pos1_col = int(sys.argv[6])
        target_pos2_col = int(sys.argv[7])

    print("Loading weight...")
    model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(sys.argv[1], binary=True)

    print("Calculate similarities...")
    local_similarities = []
    true_similarities = []
    for line in list(open(sys.argv[2]))[1:]:
        line = line.strip()
        if line == "":
            continue

        data = line.rstrip().split("\t")
        word1 = data[int(sys.argv[3])].lower()
        word2 = data[int(sys.argv[4])].lower()
        word1_synsets = [x for x in wn.synsets(word1) if x.name() in model]
        word2_synsets = [x for x in wn.synsets(word2) if x.name() in model]

        if target_pos1_col != -1 and target_pos2_col != -1:
            pos1 = data[target_pos1_col].lower()
            word1_synsets = [x for x in word1_synsets if pos_tags[x.pos()] == pos1]
            pos2 = data[target_pos2_col].lower()
            word2_synsets = [x for x in word2_synsets if pos_tags[x.pos()] == pos2]

        word1_synsets = [x.name() for x in word1_synsets]
        word2_synsets = [x.name() for x in word2_synsets]
        rating_str = data[int(sys.argv[5])]

        # Local Similarity
        similarity = -10000
        for s1 in word1_synsets:
            for s2 in word2_synsets:
                similarity = max(1 - sp.spatial.distance.cosine(model[s1], model[s2]), similarity)
        local_similarities.append(similarity)
        
        # True Similarity
        rating = float(rating_str)
        true_similarities.append(rating)
    
    global_similarity = sp.stats.spearmanr(true_similarities, local_similarities)[0]

    print(f"Global Similarity: {global_similarity * 100.0:.3f}")


if __name__ == "__main__":
    if len(sys.argv) <= 5:
        print("usage: python evaluate_cws.py <weight_model_file> <ws_file> <target_wcol1> <target_wcol2> <target_scol> [<target_poscol1> <target_poscol2>]", file=sys.stderr)
        exit()

    main()
