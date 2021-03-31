#!/bin/bash

#The name of the job is train
#SBATCH -J concp10

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

EXP_NAME=en_et_concat
SAVE_DIR=da-sysclusters/experiments/$EXP_NAME
DATA_PATH=single-domain/sys-clusters/segmented

mkdir $SAVE_DIR

fairseq-train \
    $DATA_PATH/bin-data-en-et-base \
    --arch transformer --patience 10 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --eval-bleu --eval-bleu-remove-bpe=sentencepiece \
    --max-tokens 15000 \
    --log-format json --seed 321 \
    --save-dir $SAVE_DIR \
    --tensorboard-logdir $SAVE_DIR/log-tb \
    --restore-file $SAVE_DIR/checkpoint_last.pt \
    2>&1 | tee $SAVE_DIR/log.out
