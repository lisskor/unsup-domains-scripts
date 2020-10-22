#!/bin/bash

srclang=en
tgtlang=et

mkdir da-corpora

for corpus in Europarl OpenSubtitles JRC-Acquis EMEA
do
 # Download corpora from OPUS
 python opustools_to_documents.py --corpus $corpus --src $srclang --tgt $tgtlang --filename da-corpora/$corpus.$srclang-$tgtlang.docs --minsent 5
 
 # Basic cleaning
 python cleaning_docs.py --input da-corpora/$corpus.$srclang-$tgtlang.docs --output da-corpora/cl-$corpus.$srclang-$tgtlang.docs
 
 # Separate into test, dev and train in a fair way:
 # whole documents are written into sets,
 # the clean test and dev will not contain sentence pairs
 # that have exact matches in other sets
 python separate_test_dev_train.py --input da-corpora/cl-$corpus.$srclang-$tgtlang.docs --test_size 3000 --dev_size 3000 --train_size 500000
done
