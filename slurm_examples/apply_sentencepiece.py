#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import sentencepiece as spm
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def train(arguments):
    logging.info("Starting training")
    spm.SentencePieceTrainer.train(
        input=arguments.corpora,
        model_prefix=arguments.model,
        vocab_size=arguments.size)


def split(arguments):
    # Load model
    sp = spm.SentencePieceProcessor(model_file=arguments.model + ".model")

    # Split each input file
    for corpus in arguments.corpora:
        logging.info("Splitting file {}".format(corpus))
        with open(corpus, 'r', encoding='utf8') as f:
            sentences = [line.strip() for line in f.readlines()]
        # Create output file name (add the model prefix)
        out_file = os.path.join(os.path.split(corpus)[0],
                                os.path.split(arguments.model)[1] + '-' +
                                os.path.split(corpus)[1])
        with open(out_file, 'w', encoding='utf8') as f:
            # Process sentences
            for sentence in sentences:
                pieces = sp.encode(sentence, out_type=str)
                # Check result type, there might be unexpected behavior
                if type(pieces[0]) == str:
                    f.write(' '.join([word for word
                                      in sp.encode(sentence, out_type=str)]))
                elif type(pieces[0]) == bytes:
                    f.write(' '.join([word.decode('utf-8') for word
                                      in sp.encode(sentence, out_type=str)]))
                f.write('\n')


def restore(arguments):
    # Load model
    sp = spm.SentencePieceProcessor(model_file=arguments.model + ".model")

    for corpus in arguments.corpora:
        logging.info("De-sp file {}".format(corpus))
        with open(corpus, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f.readlines()]
        # Create output file name (add the model prefix)
        out_file = os.path.join(os.path.split(corpus)[0],
                                'de-' + os.path.split(arguments.model)[1] +
                                '-' + os.path.split(corpus)[1])
        with open(out_file, 'w', encoding='utf8') as f:
            # Process sentences
            for sentence in sentences:
                glued_sentence = sp.decode(sentence.split())
                # Check result type, there might be unexpected behavior
                if type(glued_sentence) == str:
                    f.write(glued_sentence)
                elif type(glued_sentence) == bytes:
                    f.write(glued_sentence.decode('utf-8'))
                f.write('\n')


if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--action", choices=["train", "split", "restore"],
                        default="split", required=True,
                        help="The type of action to be performed: 'train' "
                             "for training a new model, 'split' for splitting "
                             "text using an existing model, 'restore' for "
                             "glueing subwords back into plain text")
    parser.add_argument("--size", type=int, dest="size",
                        default=32000, help="Vocabulary size "
                                            "(for training mode)")
    parser.add_argument("--corpora", dest="corpora", nargs="+",
                        help="File names of all files separated by spaces",
                        required=True)
    parser.add_argument("--model", dest="model",
                        help="SentencePiece model file prefix or prefix "
                             "of an existing model", default="wordpieces")

    args = parser.parse_args()

    # Perform the specified action
    if args.action == 'train':
        train(args)

    if args.action == 'split':
        split(args)

    if args.action == 'restore':
        restore(args)
