#!/bin/bash

if [ $# -ne 1 ]; then
  echo "usage: evaluate.sh <model_dir>" 1>&2
  exit 1
fi

echo "Evaluate by word similarity"
echo " - MEN-3K"
python evaluate_word_ws.py $1/embedding_in_weight ../corpus/WordSimilarity/MEN-3K.txt 0 1 2
python evaluate_sense_ws.py $1/embedding_in_weight ../corpus/WordSimilarity/MEN-3K.txt 0 1 2
echo " - RG-65"
python evaluate_word_ws.py $1/embedding_in_weight ../corpus/WordSimilarity/RG-65.txt 0 1 2
python evaluate_sense_ws.py $1/embedding_in_weight ../corpus/WordSimilarity/RG-65.txt 0 1 2
echo " - YP-130"
python evaluate_word_ws.py $1/embedding_in_weight ../corpus/WordSimilarity/YP-130.txt 0 1 2
python evaluate_sense_ws.py $1/embedding_in_weight ../corpus/WordSimilarity/YP-130.txt 0 1 2
echo " - SimLex-999"
python evaluate_word_ws.py $1/embedding_in_weight ../corpus/WordSimilarity/SimLex-999.txt 0 1 3 2 2
python evaluate_sense_ws.py $1/embedding_in_weight ../corpus/WordSimilarity/SimLex-999.txt 0 1 3 2 2

echo ""

echo "Evaluate by context word similarity"
echo " - SCWS"
python evaluate_cws.py $1 ../corpus/WordSimilarity/SCWS.txt

echo ""

echo "Evaluate by fine-grained word sense disambiguation (with sense frequency)"
echo " - ALL"
python evaluate_wsd.py $1 ../corpus/WSD/Fine-Grained/ALL/ALL.data.xml ../corpus/WSD/Fine-Grained/ALL/ALL.gold.key.txt ../corpus/WSD/
echo " - senseval2"
python evaluate_wsd.py $1 ../corpus/WSD/Fine-Grained/senseval2/senseval2.data.xml ../corpus/WSD/Fine-Grained/senseval2/senseval2.gold.key.txt ../corpus/WSD/
echo " - senseval3"
python evaluate_wsd.py $1 ../corpus/WSD/Fine-Grained/senseval3/senseval3.data.xml ../corpus/WSD/Fine-Grained/senseval3/senseval3.gold.key.txt ../corpus/WSD/
echo " - semeval2007"
python evaluate_wsd.py $1 ../corpus/WSD/Fine-Grained/semeval2007/semeval2007.data.xml ../corpus/WSD/Fine-Grained/semeval2007/semeval2007.gold.key.txt ../corpus/WSD/
echo " - semeval2013"
python evaluate_wsd.py $1 ../corpus/WSD/Fine-Grained/semeval2013/semeval2013.data.xml ../corpus/WSD/Fine-Grained/semeval2013/semeval2013.gold.key.txt ../corpus/WSD/
echo " - semeval2015"
python evaluate_wsd.py $1 ../corpus/WSD/Fine-Grained/semeval2015/semeval2015.data.xml ../corpus/WSD/Fine-Grained/semeval2015/semeval2015.gold.key.txt ../corpus/WSD/

echo ""

echo "Evaluate by fine-grained word sense disambiguation (without sense frequency)"
echo " - ALL"
python evaluate_wsd.py $1 ../corpus/WSD/Fine-Grained/ALL/ALL.data.xml ../corpus/WSD/Fine-Grained/ALL/ALL.gold.key.txt ../corpus/WSD/ 0
echo " - senseval2"
python evaluate_wsd.py $1 ../corpus/WSD/Fine-Grained/senseval2/senseval2.data.xml ../corpus/WSD/Fine-Grained/senseval2/senseval2.gold.key.txt ../corpus/WSD/ 0
echo " - senseval3"
python evaluate_wsd.py $1 ../corpus/WSD/Fine-Grained/senseval3/senseval3.data.xml ../corpus/WSD/Fine-Grained/senseval3/senseval3.gold.key.txt ../corpus/WSD/ 0
echo " - semeval2007"
python evaluate_wsd.py $1 ../corpus/WSD/Fine-Grained/semeval2007/semeval2007.data.xml ../corpus/WSD/Fine-Grained/semeval2007/semeval2007.gold.key.txt ../corpus/WSD/ 0
echo " - semeval2013"
python evaluate_wsd.py $1 ../corpus/WSD/Fine-Grained/semeval2013/semeval2013.data.xml ../corpus/WSD/Fine-Grained/semeval2013/semeval2013.gold.key.txt ../corpus/WSD/ 0
echo " - semeval2015"
python evaluate_wsd.py $1 ../corpus/WSD/Fine-Grained/semeval2015/semeval2015.data.xml ../corpus/WSD/Fine-Grained/semeval2015/semeval2015.gold.key.txt ../corpus/WSD/ 0

echo ""

echo "Evaluate by coarse-grained word sense disambiguation (without sense frequency)"
echo " - semeval2007"
python evaluate_wsd.py $1 ../corpus/WSD/Coarse-Grained/semeval2007/semeval2007.data.xml ../corpus/WSD/Coarse-Grained/semeval2007/semeval2007.gold.key.txt ../corpus/WSD/ 0

echo "Evaluate by coarse-grained word sense disambiguation (with sense frequency)"
echo " - semeval2007"
python evaluate_wsd.py $1 ../corpus/WSD/Coarse-Grained/semeval2007/semeval2007.data.xml ../corpus/WSD/Coarse-Grained/semeval2007/semeval2007.gold.key.txt ../corpus/WSD/ 1
