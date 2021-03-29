#!/bin/bash

#The name of the job is train
#SBATCH -J transl

#The job requires 1 compute node
#SBATCH -N 1

#The job requires 1 task per node
#SBATCH --ntasks-per-node=1

#The maximum walltime of the job is 8 days
#SBATCH -t 192:00:00

#SBATCH --mem=20GB

#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

#SBATCH --exclude=falcon3

module load python/3.6.3/CUDA

source activate fairseq-da

DATA_PATH=single-domain/sys-clusters/segmented
RESULTS_PATH=da-sysclusters/experiments/translations
set=test-cl

srclang=en
tgtlang=et

EXP_NAME=concat
SAVE_DIR=da-sysclusters/experiments/${srclang}_${tgtlang}_${EXP_NAME}
 for corpus in Europarl OpenSubtitles JRC-Acquis EMEA
 do
  # translate
  cat $DATA_PATH/sp-cl-$corpus.$srclang-$tgtlang.docs.$set.$srclang \
    | fairseq-interactive $DATA_PATH/bin-data-$srclang-$tgtlang-base \
      --source-lang $srclang --target-lang $tgtlang \
      --path $SAVE_DIR/checkpoint_best.pt \
      --buffer-size 2000 --batch-size 32 --beam 5 \
    > $RESULTS_PATH/transl_${srclang}_${tgtlang}_${EXP_NAME}_${corpus}.sys
	
  # grep translations from the output file
  grep "^H" $RESULTS_PATH/transl_${srclang}_${tgtlang}_${EXP_NAME}_${corpus}.sys | cut -f3 > $RESULTS_PATH/hyp_${srclang}_${tgtlang}_${EXP_NAME}_${corpus}.txt
  # grep "^T" $RESULTS_PATH/out_${srclang}_${tgtlang}_${EXP_NAME}_${corpus}.sys | cut -f2 > $RESULTS_PATH/tgt_${srclang}_${tgtlang}_${EXP_NAME}_${corpus}.txt
  
  # de-sentencepiece
  python3 ../scripts/apply_sentencepiece.py --corpora $RESULTS_PATH/hyp_${srclang}_${tgtlang}_${EXP_NAME}_${corpus}.txt --model single-domain/preproc-models/fs-en-et --action restore
  
  # calculate bleu w/sacrebleu
  echo $EXP_NAME
  echo $corpus
  cat $RESULTS_PATH/de-fs-en-et-hyp_${srclang}_${tgtlang}_${EXP_NAME}_${corpus}.txt | sacrebleu single-domain/sys-clusters/cl-$corpus.$srclang-$tgtlang.docs.$set.$tgtlang
 done

