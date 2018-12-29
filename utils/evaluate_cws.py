import sys
import more_itertools
import numpy as np
import scipy as sp

from sv4d import Model


def extract_feature(document_str, model):
    document = []
    sentence = []
    target_sent_row = 0
    current_row = 0
    for word in document_str.split(" "):
        if word == ".":
            document.append(sentence)
            sentence = []
            current_row += 1
        elif word == "<b>":
            target_sent_row = current_row
        else:
            if word.lower() in model.synset_vocab:
                sentence.append(word.lower())
    else:
        document.append(sentence)
    
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
    return (contexts, document[target_sent_row], document[max(0, target_sent_row - 1):min(len(document), target_sent_row + 1)])


def main():
    use_sense_prob = False
    if len(sys.argv) >= 4:
        use_sense_prob = bool(int(sys.argv[5]))

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

        _, word1, pos1, word2, pos2, document1, document2, rating_str, _, _, _, _, _, _, _, _, _, _ = line.rstrip().split("\t")
        
        word1 = word1.lower()
        word2 = word2.lower()

        feature1 = extract_feature(document1, model)
        feature2 = extract_feature(document2, model)

        prob1, synsets1 = model.calculate_sense_probability(word1, pos1, feature1[0], feature1[1], feature1[2], use_sense_prob=use_sense_prob)
        prob2, synsets2 = model.calculate_sense_probability(word2, pos2, feature2[0], feature2[1], feature2[2], use_sense_prob=use_sense_prob)
        
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

        # Global Similarity
        similarity = 1 - sp.spatial.distance.cosine(model[word1], model[word2])
        global_similarities.append(similarity)
        
        # True Similarity
        rating = float(rating_str)
        true_similarities.append(rating)
    
    maximum_similarity = sp.stats.spearmanr(true_similarities, maximum_similarities)[0]
    average_similarity = sp.stats.spearmanr(true_similarities, average_similarities)[0]
    global_similarity = sp.stats.spearmanr(true_similarities, global_similarities)[0]

    print(f"Maximum Similarity: {maximum_similarity * 100.0:.3f}")
    print(f"Average Similarity: {average_similarity * 100.0:.3f}")
    print(f"Global Similarity: {global_similarity * 100.0:.3f}")


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print("usage: python evaluate_cws.py <model_dir> <cws_file> [<use_sense_prob>]", file=sys.stderr)
        exit()

    main()
