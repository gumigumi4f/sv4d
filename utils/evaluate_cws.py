import sys
import numpy as np
import scipy as sp
from nltk import tokenize

from sv4d import Model


def extract_feature(document_str, model):
    documents = [x.lower() for x in document_str.split(" ") if x.lower() in model.synset_vocab]
    
    sentences = []
    flag = False
    for word in document_str.split(" "):
        if word == ".":
            if flag:
                break
            sentences = []
        elif word == "<b>":
            flag = True
        else:
            if word.lower() in model.synset_vocab:
                sentences.append(word.lower())
    
    bcontexts = []
    bflag = True
    acontexts = []
    aflag = False
    for word in document_str.split(" "):
        if word == "<b>":
            bflag = False
        elif word == "</b>":
            aflag = True
        else:
            if bflag and word.lower() in model.synset_vocab:
                bcontexts.append(word.lower())
            if aflag and word.lower() in model.synset_vocab:
                acontexts.append(word.lower())
    
    contexts = bcontexts[-5:] + acontexts[:5]
    return (contexts, sentences, documents)


def main():
    model = Model(sys.argv[1])
    print("Loading vocab...")
    model.load_vocab()
    print("Loading weight...")
    model.load_weight()

    print("Calculate similarities...")
    maximum_similarities = []
    average_similarities = []
    global_similarities = []
    true_similarities = []
    for line in open(sys.argv[2]):
        line = line.strip()
        if line == "":
            continue

        num, word1, pos1, word2, pos2, document1, document2, rating_str, _, _, _, _, _, _, _, _, _, _ = line.rstrip().split("\t")
        
        word1 = word1.lower()
        word2 = word2.lower()

        feature1 = extract_feature(document1, model)
        feature2 = extract_feature(document2, model)

        prob1, synsets1 = model.calculate_sense_probability(word1, pos1, feature1[0], feature1[1], feature1[2])
        prob2, synsets2 = model.calculate_sense_probability(word2, pos2, feature2[0], feature2[1], feature2[2])
        
        # Maximum Similarity
        synset1 = synsets1[np.argmax(prob1)]
        synset2 = synsets2[np.argmax(prob2)]
        similarity = 1 - sp.spatial.distance.cosine(model[synset1], model[synset2])
        maximum_similarities.append(similarity)

        # Average Similarity
        similarity = 0.0
        for s1 in range(len(synsets1)):
            for s2 in range(len(synsets2)):
                sim = 1 - sp.spatial.distance.cosine(
                    model[synsets1[s1]],
                    model[synsets2[s2]]
                )
                similarity += prob1[s1] * prob2[s2] * sim
        average_similarities.append(similarity)

        # Glo Similarity
        similarity = 1 - sp.spatial.distance.cosine(model[word1], model[word2])
        global_similarities.append(similarity)
        
        # True Similarity
        rating = float(rating_str)
        true_similarities.append(rating)
    
    maximum_similarity = sp.stats.spearmanr(true_similarities, maximum_similarities)[0]
    average_similarity = sp.stats.spearmanr(true_similarities, average_similarities)[0]
    global_similarity = sp.stats.spearmanr(true_similarities, global_similarities)[0]

    print(f"Maximum Similarity:{maximum_similarity * 100.0:.3f}")
    print(f"Average Similarity:{average_similarity * 100.0:.3f}")
    print(f"Global Similarity:{global_similarity * 100.0:.3f}")


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print("usage: python evaluate_cws.py <model_dir> <cws_file>", file=sys.stderr)
        exit()

    main()
