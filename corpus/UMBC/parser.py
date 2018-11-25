# -*- coding: utf-8 -*-

import sys
import os
import glob
import tqdm


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
                
                words = [x.split("_")[0].lower() for x in line.rstrip().split(" ")]
                words = [x for x in words if x in vocab]
                fout.write(" ".join(words) + "\n")
            fout.write("</doc>\n")


if __name__ == "__main__":
    if len(sys.argv) <= 3:
        print("usage: python parser.py <vocab_file> <input_file_path> <output_file>", file=sys.stderr)
        exit()

    main()
