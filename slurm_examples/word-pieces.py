#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Liisa Rätsep, Lisa Korotkova

import os
import subprocess
import logging
import sentencepiece as spm
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def train(arguments):
    logging.info("Creating a training file")
    # Create file with 10,000,000 random lines from the corpora
    subprocess.call("cat " + ' '.join(
        arguments.corpora) + "| shuf | head -n 10000000 > " + arguments.model + ".train",
                    shell=True)
    logging.info("Starting training")
    # Train the model
    spm.SentencePieceTrainer.Train(
        '--input=' + arguments.model + ".train --model_prefix=" + arguments.model + " --vocab_size=" + str(
            arguments.size) + " --input_sentence_size=10000000")
    # Remove training file
    subprocess.call("rm " + arguments.model + ".train", shell=True)


def split(arguments):
    # Initialize SentencePiece processor
    sp = spm.SentencePieceProcessor()

    # Load model
    logging.info("Loading model")
    sp.Load(arguments.model + ".model")

    for corpus in arguments.corpora:
        logging.info("Splitting file {}".format(corpus))
        with open(corpus, 'r', encoding='utf8') as f:
            sentences = f.readlines()
        # Create output file name (add the model prefix)
        out_file = os.path.join(os.path.split(corpus)[0],
                                os.path.split(arguments.model)[1] + '-' +
                                os.path.split(corpus)[1])
        with open(out_file, 'w', encoding='utf8') as f:
            # Process sentences
            for sentence in sentences:
                pieces = sp.EncodeAsPieces(sentence)
                # Check result type, there might be unexpected behavior
                if type(pieces[0]) == str:
                    f.write(' '.join([x for x in sp.EncodeAsPieces(sentence)]))
                    f.write('\n')
                elif type(pieces[0]) == bytes:
                    f.write(' '.join([x.decode('utf-8') for x
                                      in sp.EncodeAsPieces(sentence)]))
                    f.write('\n')


def restore(arguments):
    # Initialize SentencePiece processor
    sp = spm.SentencePieceProcessor()

    # Load model
    logging.info("Loading model")
    sp.Load(arguments.model + ".model")

    for corpus in arguments.corpora:
        logging.info("De-sp file {}".format(corpus))
        with open(corpus, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        # Create output file name (add the model prefix)
        out_file = os.path.join(os.path.split(corpus)[0],
                                'de-' + os.path.split(arguments.model)[1] +
                                '-' + os.path.split(corpus)[1])
        with open(out_file, 'w', encoding='utf8') as f:
            # Process sentences
            for sentence in sentences:
                l = sp.DecodePieces(sentence.split())
                # Check result type, there might be unexpected behavior
                if type(l) == bytes:
                    f.write(l.decode('utf-8'))
                elif type(l) == str:
                    f.write(l)
                f.write('\n')


if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--action", choices=["train", "split", "restore"],
                        default="split", required=True,
                        help="The type of action to be performed: 'train' "
                             "for training a new model, 'split' for splitting "
                             "text using an existing model, 'restore' for "
                             "glueing wordpieces back into plain text")
    parser.add_argument("--size", type=int, dest="size",
                        default=32000, help="Number of wordpieces "
                                            "(for training mode)")
    parser.add_argument("--corpora", dest="corpora", nargs="+",
                        help="File names of all files separated by spaces",
                        required=True)
    parser.add_argument("--model", dest="model",
                        help="Wordpiece model file prefix or prefix "
                             "of an existing model", default="wordpieces")

    args = parser.parse_args()

    # Perform the specified action
    if args.action == 'train':
        train(args)

    if args.action == 'split':
        split(args)

    if args.action == 'restore':
        restore(args)
