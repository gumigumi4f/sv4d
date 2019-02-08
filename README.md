SV4D: Sense Vector for Disambiguation
--

Knowledge-based Word Sense Disambiguation (WSD) and Representation (WSR) tools.
Only we need to training model is unannotated corpus and list of words related to target sense.

Requirements
--

### Pre-processing
- Python 3.5 or newer
  - requests
  - numpy
  - scipy
  - gensim
  - nltk
  - DAWG
  - tqdm
  - py4j
  - more_itertools
  - beautifulsoup4
- Java 1.8.0 or newer
- Stanford CoreNLP 3.9.2 or newer
- BabelNet 4.0.1 indices and BabelNet 4.0.1 API

### Training
- g++ 4.8.5 or newer

### Evaluation
- Python 3.5 or newer
- Java 1.8.0 or newer

Pre-trained models and pre-processed corpus
--
- [https://drive.google.com/drive/folders/1-AYaFJvPnUpTlUQD6BaNMhaYtofuhiry?usp=sharing](https://drive.google.com/drive/folders/1-AYaFJvPnUpTlUQD6BaNMhaYtofuhiry?usp=sharing)

Pre-processing
--

You have to download wikipedia raw corpus from [here](https://www.socher.org/index.php/Main/ImprovingWordRepresentationsViaGlobalContextAndMultipleWordPrototypes) before pre-process corpus.
Also you have to download BabelNet API and Stanford CoreNLP, and locate it to "tools" directory.

```sh
# to export sense.txt using WordNet and BabelNet
cd ./tools/BabelNet-API-4.0.1
cp ../../corpus/Sense.java ./
javac -classpath "lib/*:babelnet-api-4.0.1.jar:config:<your pythob path here>/share/py4j/py4j0.10.7.jar" Sense.java
nohup java -classpath "lib/*:babelnet-api-4.0.1.jar:config:<your pythob path here>/share/py4j/py4j0.10.7.jar:." Sense &
cd ../../corpus
python sense.py vocab.txt sense.txt

# to pre-process wikipedia corpus
cd ./tools/stanford-nlp-toolkit
nohup java -Xmx16g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,parse,depparse -status_port 42636 -port 42636 -timeout 15000 -encoding utf-8 > /dev/null &
cd ../../corpus/Wikipedia
python parser.py ../vocab.txt Wikipedia.Corpus.txt Wikipedia.PosTaggedCorpus.Corpus.txt
python postprocess.py ../sense.txt Wikipedia.PosTaggedCorpus.txt Wikipedia.ProcessedCorpus.txt
```

Training model
--

```sh
cd ./models
mkdir default
cd ../src
make
cd ../bin
./sv4d training -training_corpus ../corpus/Wikipedia/Wikipedia.ProcessedCorpus.txt -synset_data_file ../corpus/sense.txt -model_dir ../models/default -epochs 50
```

Evaluation
--

```sh
# Check trained sense vector
cd ./bin
./sv4d synset_nearest_neighbour -model_dir ../models/default

# Evaluate with Word Similarity dataset and WSD dataset
cd ./utils
./evaluate.sh ../models/default
```

Result
--

- [https://docs.google.com/spreadsheets/d/1CeqtLfqXHudgwRs1VG4YdULs8ezgn1NDjxJUH1EaUlQ/edit?usp=sharing](https://docs.google.com/spreadsheets/d/1CeqtLfqXHudgwRs1VG4YdULs8ezgn1NDjxJUH1EaUlQ/edit?usp=sharing)