# -*- coding: utf-8 -*-

import sys
import re
import time
from nltk.parse.corenlp import CoreNLPParser


end_of_document_symbol = "---END.OF.DOCUMENT---".lower()


def main():
    tokenizer = CoreNLPParser(url='http://localhost:42636')

    vocab = set()
    for line in open(sys.argv[1]):
        word = line.rstrip()
        vocab.add(word)

    document_buffer = ""
    token_buffer = []

    with open(sys.argv[2]) as fin, open(sys.argv[3], "w") as fout:
        start = time.time()

        for e, line in enumerate(fin):
            if line.strip() == "":
                continue
            elif line.strip().lower() != end_of_document_symbol:
                document_buffer += line.strip() + " <br> "
                if len(document_buffer) > 90000:
                    token_buffer += list(tokenizer.tokenize(document_buffer))
                    document_buffer = ""
            else:
                token_buffer += list(tokenizer.tokenize(document_buffer))
                
                document = " ".join([x.lower() if x != "." else "<br>" for x in token_buffer if x.lower() in vocab or x in ["<br>", "."]])
                sentences = [x.strip() for x in document.split("<br>") if x.strip()]
                fout.write("<doc>\n" + "\n".join(sentences) + "\n</doc>\n")

                document_buffer = ""
                token_buffer = []
            
            eta = 30749930 / (e + 1) * (time.time() - start) - (time.time() - start)
            if (e + 1) % 1000 == 0:
                sys.stdout.write("\rsent: %i/%i\tETA: %f" % (e + 1, 30749930, eta))
                sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) <= 3:
        print("usage: python parser.py <vocab_file> <input_file> <output_file>", file=sys.stderr)
        exit()

    main()
