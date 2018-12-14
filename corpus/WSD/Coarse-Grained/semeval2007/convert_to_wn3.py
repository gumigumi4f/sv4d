# -*- coding: utf-8 -*-

import sys


def main():
    mapping = {}
    for line in open(sys.argv[1]):
        wn2, _, wn3, _ = line.rstrip().split(" ")
        if wn2 not in mapping:
            mapping[wn2] = wn3

    for line in open(sys.argv[3]):
        if wn2 not in mapping:
            mapping[wn2] = wn3

    for line in open(sys.argv[2]):
        data = line.rstrip().split(" ")
        if len(data) < 3:
            continue
        wn2s = data[1]
        wn3s = data[2]
        wn2, _, _ = wn2s.split(";")
        wn3, _, _ = wn3s.split(";")
        if wn2 not in mapping:
            mapping[wn2] = wn3
    
    for line in open(sys.argv[4]):
        data = line.rstrip().split(" ")
        if len(data) < 3:
            continue
        wn2s = data[1]
        wn3s = data[2]
        wn2, _, _ = wn2s.split(";")
        wn3, _, _ = wn3s.split(";")
        if wn2 not in mapping:
            mapping[wn2] = wn3
    
    with open(sys.argv[6], "w") as fout:
        for line in open(sys.argv[5]):
            data = line.rstrip().split(" ")
            instance = data[1]
            keys = [x if x not in mapping else mapping[x] for x in data[2:-2]]
            keys = sorted(list(set(keys)))
            print(instance, " ".join(keys), sep=" ", file=fout)
            
if __name__ == "__main__":
    if len(sys.argv) <= 5:
        print("usage: python convert_to_wn3.py <2.1to3.0.noun.mono> <2.1to3.0.noun.poly> <2.1to3.0.verb.mono> <2.1to3.0.verb.poly> <goldkey_file> <output_file>", file=sys.stderr)
        exit()
    main()