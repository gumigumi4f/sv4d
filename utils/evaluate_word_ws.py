import sys
import numpy as np
import scipy as sp
import gensim


def main():
    print("Loading weight...")
    model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(sys.argv[1], binary=True)

    print("Calculate similarities...")
    global_similarities = []
    true_similarities = []
    for line in list(open(sys.argv[2]))[1:]:
        line = line.strip()
        if line == "":
            continue

        data = line.rstrip().split("\t")
        word1 = data[int(sys.argv[3])].lower()
        word2 = data[int(sys.argv[4])].lower()
        rating_str = data[int(sys.argv[5])]

        # Global Similarity
        similarity = 1 - sp.spatial.distance.cosine(model[word1], model[word2])
        global_similarities.append(similarity)
        
        # True Similarity
        rating = float(rating_str)
        true_similarities.append(rating)
    
    score = sp.stats.spearmanr(true_similarities, global_similarities)[0]

    print(f"Word-based Score: {score * 100.0:.3f}")


if __name__ == "__main__":
    if len(sys.argv) <= 5:
        print("usage: python evaluate_word_ws.py <weight_model_file> <ws_file> <target_wcol1> <target_wcol2> <target_scol>", file=sys.stderr)
        exit()

    main()
