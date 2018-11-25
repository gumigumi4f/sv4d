# -*- coding: utf-8 -*-


import sys
import json
import time
import dawg


def main():
    max_length = 1
    compound_words = []
    for line in open(sys.argv[1]):
        word = line.rstrip()
        max_length = max(len(word.split("_")), max_length)

        if len(word.split("_")) <= 1:
            continue

        compound_words.append(word)
    
    compound_words_dawg = dawg.DAWG(compound_words)

    with open(sys.argv[3], "w") as fout:
        start = time.time()
        for e, line in enumerate(open(sys.argv[2])):
            line = line.strip()
            if line == "":
                continue
            elif line == "<doc>":
                print("<doc>", file=fout)
                continue
            elif line == "</doc>":
                print("</doc>", file=fout)
                continue
            
            processed_tokens = []

            tokens = line.split(" ")
            i = 0
            while i < len(tokens):
                chars = "_".join([x for x in tokens[i:i + max_length]])
                common_prefix = compound_words_dawg.prefixes(chars)
                if common_prefix:
                    common_prefix_max = max(common_prefix, key=lambda x: len(x))
                    processed_tokens.append(common_prefix_max)
                    length = len(common_prefix_max.split("_"))
                    i += length
                else:
                    processed_tokens.append(tokens[i])
                    i += 1
            
            print(" ".join(processed_tokens), file=fout)
            
            if (e + 1) % 1000 == 0:
                eta = 40587254 / (e + 1) * (time.time() - start) - (time.time() - start)
                sys.stdout.write("\rsent: %i/%i\tETA: %f" % (e + 1, 40587254, eta))
                sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) <= 3:
        print("usage: python postprocess.py <vocab_file> <input_file> <output_file>", file=sys.stderr)
        exit()

    main()