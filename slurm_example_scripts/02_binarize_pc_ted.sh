#!/bin/bash

#The name of the job is train
#SBATCH -J binpc

#The job requires 1 compute node
#SBATCH -N 1

#The job requires 1 task per node
#SBATCH --ntasks-per-node=1

#The maximum walltime of the job is 8 days
#SBATCH -t 192:00:00

#SBATCH --mem=20GB

module load python-3.6.3
module load cudnn/7.2.1/cuda-9.2

source activate fairseq-da

srclang=en
tgtlang=et
datapath=single-domain/sys-clusters/unseen

# First, concatenate files from all corpora
# (not necessary in this case because we are using one corpus at a time)

for corpus in ParaCrawl TED
do
for set in train dev test
do
 srclang=en
 tgtlang=et
 echo $set
 echo "src"
 cat $datapath/sp-cl-$corpus.$srclang-$tgtlang.$set.$srclang > $datapath/sp-cl-$corpus.$srclang-$tgtlang.$set.src
 echo "tgt"
 cat $datapath/sp-cl-$corpus.$srclang-$tgtlang.$set.$tgtlang > $datapath/sp-cl-$corpus.$srclang-$tgtlang.$set.tgt
done

# Paste source and target data into one file to shuffle them in parallel
# Shuffle, then cut source and target back into separate files

for set in train dev test
do
 paste $datapath/sp-cl-$corpus.$srclang-$tgtlang.$set.src $datapath/sp-cl-$corpus.$srclang-$tgtlang.$set.tgt | shuf > $datapath/shuf-sp-cl-$corpus.$srclang-$tgtlang.$set.both
 cut -f1 $datapath/shuf-sp-cl-$corpus.$srclang-$tgtlang.$set.both > $datapath/shuf-sp-cl-$corpus.$srclang-$tgtlang.$set.src
 cut -f2 $datapath/shuf-sp-cl-$corpus.$srclang-$tgtlang.$set.both > $datapath/shuf-sp-cl-$corpus.$srclang-$tgtlang.$set.tgt
done

# Copy data into $datapath/fairseq-data-$srclang-$tgtlang-base with filenames like train.en, valid.et, etc.

mkdir -p $datapath/fairseq-data-$srclang-$tgtlang-$corpus-ft
mkdir -p $datapath/bin-data-$srclang-$tgtlang-$corpus-ft

cp $datapath/shuf-sp-cl-$corpus.$srclang-$tgtlang.train.src $datapath/fairseq-data-$srclang-$tgtlang-$corpus-ft/train.en
cp $datapath/shuf-sp-cl-$corpus.$srclang-$tgtlang.train.tgt $datapath/fairseq-data-$srclang-$tgtlang-$corpus-ft/train.et
cp $datapath/shuf-sp-cl-$corpus.$srclang-$tgtlang.dev.src $datapath/fairseq-data-$srclang-$tgtlang-$corpus-ft/valid.en
cp $datapath/shuf-sp-cl-$corpus.$srclang-$tgtlang.dev.tgt $datapath/fairseq-data-$srclang-$tgtlang-$corpus-ft/valid.et
cp $datapath/shuf-sp-cl-$corpus.$srclang-$tgtlang.test.src $datapath/fairseq-data-$srclang-$tgtlang-$corpus-ft/test.en
cp $datapath/shuf-sp-cl-$corpus.$srclang-$tgtlang.test.tgt $datapath/fairseq-data-$srclang-$tgtlang-$corpus-ft/test.et

# Finally, binarize for fairseq

fairseq-preprocess --source-lang $srclang --target-lang $tgtlang \
    --trainpref $datapath/fairseq-data-$srclang-$tgtlang-$corpus-ft/train --validpref $datapath/fairseq-data-$srclang-$tgtlang-$corpus-ft/valid --testpref $datapath/fairseq-data-$srclang-$tgtlang-$corpus-ft/test \
    --destdir $datapath/bin-data-$srclang-$tgtlang-$corpus-ft --joined-dictionary \
    --srcdict single-domain/sys-clusters/segmented/bin-data-$srclang-$tgtlang-base/dict.en.txt

done
