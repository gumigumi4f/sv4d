import sys
import subprocess
import more_itertools
import numpy as np
import scipy as sp
from nltk import tokenize
from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup

from sv4d import Model


def main():
    use_sense_prob = True
    if len(sys.argv) >= 6:
        use_sense_prob = bool(int(sys.argv[5]))

    model = Model(sys.argv[1])
    # print("Loading vocab...")
    model.load_vocab()
    # print("Loading weight...")
    model.load_weight()

    # print("Calculating accuracy...")
    wsd_dataset_pos_tags = {
        "NOUN": "n",
        "VERB": "v",
        "ADJ": "a",
        "ADV": "r"
    }
    with open("/tmp/sv4d_wsd.key", "w") as fout:
        soup = BeautifulSoup(open(sys.argv[2]), "xml")
        corpus = soup.find("corpus")
        for text_element in corpus.findAll("text"):
            document = []
            for sentence_element in text_element.findAll("sentence"):
                sentence = []
                for child in sentence_element.children:
                    if child.name is None:
                        sentence.extend([x.lower() for x in child.title().split("\n") if x.lower() in model.synset_vocab])
                    else:
                        if child.attrs["lemma"].lower() not in model.synset_vocab:
                            continue
                        sentence.append(child.attrs["lemma"].lower())
                document.append(sentence)

            for e, sentence_element in enumerate(text_element.findAll("sentence")):
                sentence = []
                for child in sentence_element.children:
                    if child.name is None:
                        sentence.extend([(x.lower(), "", "") for x in child.title().split("\n") if x.lower() in model.synset_vocab])
                    else:
                        word = child.attrs["lemma"].lower()
                        if child.attrs["lemma"].lower() not in model.synset_vocab:
                            continue
                        pos = wsd_dataset_pos_tags[child.attrs["pos"]] if child.attrs["pos"] in wsd_dataset_pos_tags else child.attrs["pos"]
                        instance_id = child.attrs["id"] if child.name == "instance" else ""
                        sentence.append((word, pos, instance_id))

                for i, (word, pos, instance_id) in enumerate(sentence):
                    if instance_id == "":
                        continue

                    contexts = [x[0] for x in sentence[:i]][-5:] + [x[0] for x in sentence[i + 1:]][:5]
                    sent = [x[0] for x in sentence]
                    doc = document[max(0, e - 1):min(len(document), e + 1)]
                    prob, synsets = model.calculate_sense_probability(word, pos, contexts, sent, doc, use_sense_prob=use_sense_prob)

                    synset = synsets[np.argmax(prob)]
                    if "." not in synset:
                        continue
                    synset_keys = [x.key() for x in wn.synset(name=synset).lemmas() if x.name().lower() == word]
                    if len(synset_keys) != 0:
                        synset_key = synset_keys[0]
                    else:
                        synset_key = [x.key() for x in wn.synset(name=synset).lemmas()][0]
                        _, key = synset_key.split("%")
                        synset_key = "%".join([word, key])

                    print(instance_id, synset_key, file=fout)
    
    p = subprocess.Popen(
        ["java", "-classpath", sys.argv[4], "Scorer", 
        sys.argv[3], "/tmp/sv4d_wsd.key"], 
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    output = p.communicate()[0].decode("utf-8").strip().split("\n")
    precision = float(output[0].split("\t")[1].rstrip("%"))
    recall = float(output[1].split("\t")[1].rstrip("%"))
    f1_value = float(output[2].split("\t")[1].rstrip("%"))

    print(f"Precision: {precision:.1f}  Recall: {recall:.1f}  F1Value: {f1_value:.1f}  ")


if __name__ == "__main__":
    if len(sys.argv) <= 4:
        print("usage: python evaluate_wsd.py <model_dir> <xml_file> <goldkey_file> <scorer_path> [<use_sense_prob>]", file=sys.stderr)
        exit()

    main()
