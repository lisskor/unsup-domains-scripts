#!/bin/bash

#The name of the job is tc
#SBATCH -J prepnew

#The job requires 1 compute node
#SBATCH -N 1

#The job requires 1 task per node
#SBATCH --ntasks-per-node=1

#The maximum walltime of the job is 72 h
#SBATCH -t 10:00:00

#SBATCH --mem=20GB

module load python-3.6.3

export LC_ALL=en_US.UTF-8

# set variables
srclang=en
tgtlang=et
datapath=single-domain/sys-clusters/unseen
modelspath=single-domain/preproc-models
scriptspath=../scripts

source activate fairseq-da

echo "SentencePiece"

# the following line is for training a sentencepiece model,
# here i'm applying an existing one (the same one used for concat)

# python3 $scriptspath/word-pieces.py --size 32000 --corpora $datapath/cl-*$srclang-$tgtlang*train.e* --model $modelspath/fs-en-et --action train

# script apply_sentencepiece.py is included in this repo as well

python3 $scriptspath/apply_sentencepiece.py --corpora $datapath/cl-*$srclang-$tgtlang*$srclang --model $modelspath/fs-en-et --action split
python3 $scriptspath/apply_sentencepiece.py --corpora $datapath/cl-*$srclang-$tgtlang*$tgtlang --model $modelspath/fs-en-et --action split

# rename the files so that they only have sp- at the beginning and not the full model name
for corpus in ParaCrawl TED
do
 for set in train dev test
 do
  mv $datapath/fs-en-et-cl-$corpus.$srclang-$tgtlang.$set.$srclang $datapath/sp-cl-$corpus.$srclang-$tgtlang.$set.$srclang
  mv $datapath/fs-en-et-cl-$corpus.$srclang-$tgtlang.$set.$tgtlang $datapath/sp-cl-$corpus.$srclang-$tgtlang.$set.$tgtlang
 done
done


