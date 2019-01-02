# -*- coding: utf-8 -*-

import sys
import os
import glob
import tqdm
from collections import defaultdict


pos_map = defaultdict(lambda: "*", {
    "JJ": "a",
    "JJR": "a",
    "JJS": "a",
    "NN": "n",
    "NNS": "n",
    "RB": "r",
    "RBR": "r",
    "RBS": "r",
    "VB": "v",
    "VBD": "v",
    "VBG": "v",
    "VBN": "v",
    "VBP": "v",
    "VBZ": "v"
})


def main():
    vocab = set()
    for line in open(sys.argv[1]):
        word = line.rstrip()
        vocab.add(word)

    with open(sys.argv[3], "w") as fout:
        for path in tqdm.tqdm(glob.glob(os.path.join(sys.argv[2], "*.possf2"))):
            fout.write("<doc>\n")
            for line in open(path):
                if line.rstrip() == "":
                    continue
                
                sentences = line.rstrip().split("_.")
                for sentence in sentences:
                    if len(sentence) == 0:
                        continue
                    words = [(x.split("_")[0].lower(), pos_map[x.split("_")[-1]]) for x in sentence[:-2].split(" ")]
                    words = [(word, pos) for word, pos in words if word in vocab]
                    fout.write(" ".join([word + "__" + pos for word, pos in words]) + "\n")
            fout.write("</doc>\n")


if __name__ == "__main__":
    if len(sys.argv) <= 3:
        print("usage: python parser.py <vocab_file> <input_file_path> <output_file>", file=sys.stderr)
        exit()

    main()
