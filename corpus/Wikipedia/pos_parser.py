# -*- coding: utf-8 -*-

import sys
import re
import time
from collections import defaultdict
from nltk.parse.corenlp import CoreNLPParser


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

end_of_document_symbol = "---END.OF.DOCUMENT---".lower()


additional_properties = {
    'tokenize.options': 'ptb3Escaping=false, unicodeQuotes=true, splitHyphenated=true, normalizeParentheses=false, normalizeOtherBrackets=false',
    'annotators': 'tokenize, ssplit, pos'
}

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
                    json_result = tokenizer.api_call(document_buffer, properties=additional_properties)
                    for sentence in json_result['sentences']:
                        token_buffer += [(x["originalText"], x["pos"]) for x in sentence['tokens']]
                    document_buffer = ""
            else:
                json_result = tokenizer.api_call(document_buffer, properties=additional_properties)
                for sentence in json_result['sentences']:
                    token_buffer += [(x["originalText"], x["pos"]) for x in sentence['tokens']]
                
                document = " ".join([x.lower() + "__" + pos_map[pos] if x != "." and x != "<br>" else "<br>" for x, pos in token_buffer if x.lower() in vocab or x in ["<br>", "."]])
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
