#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Lisa Korotkova, Mark Fishel

import logging
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def filter_lines(docs, src_lines, tgt_lines):
    """
    Discard pairs with empty sentences,
    with sentence length ratio exceeding 9
    or with less than half alphabetical characters
    """
    # Check that files have the same length
    assert len(src_lines) == len(
        tgt_lines), 'Source and target side not parallel'

    raw_result = [sent_pair for sent_pair in zip(docs, src_lines, tgt_lines) if
                  pair_ok(*sent_pair)]
    (docs_result, src_result, tgt_result) = zip(*raw_result)

    return docs_result, src_result, tgt_result


def pair_ok(doc, src_sent, tgt_sent):
    """
    Returns False if either of the sentences is an empty string,
    if length ratio exceeds 9 or if more than half of the characters
    in either sentence are non-alphabetical
    """
    # Check for empty strings
    if len(src_sent) == 0 or len(tgt_sent) == 0:
        return False

    # Check for lines with >100 tokens
    src_len, tgt_len = len(src_sent.split(" ")), len(tgt_sent.split(" "))
    if src_len > 100 or tgt_len > 100:
        return False

    # Calculate length ratio
    ratio = src_len / tgt_len if src_len > tgt_len else tgt_len / src_len
    if ratio > 9:
        return False

    # Check for alphabetic characters
    alpha_ratio_src = sum([c.isalpha() for c in src_sent]) / len(src_sent)
    alpha_ratio_tgt = sum([c.isalpha() for c in tgt_sent]) / len(tgt_sent)
    if alpha_ratio_src < 0.5 or alpha_ratio_tgt < 0.5:
        return False
    else:
        return True

    # return src_len <= 100 and tgt_len <= 100 and ratio < 9


def filter_file(input_file, output_file):
    """
    Read lines from input_file (each line of format
    "doc_id__@delimeter@__src_sent__@delimeter@__tgt_sent"),
    remove bad sentence pairs, write good sentence pairs
    into output_file (format "doc_id\\tsrc_sent\\ttgt_sent")
    """
    logging.info('Cleaning {}'.format(input_file))

    # Read in source and target sentences
    with open(input_file, 'r', encoding='utf-8') as inp_fh:
        inp_lines = [line.strip() for line in inp_fh.readlines()]
        docs, src_lines, tgt_lines = [], [], []
        for line in inp_lines:
            try:
                d, s, t = line.split('__@delimeter@__')
            except ValueError:
                continue

            docs.append(d)
            src_lines.append(s.replace('\t', ' '))
            tgt_lines.append(t.replace('\t', ' '))
    # Filter out bad sentence pairs
    out_docs, out_src, out_tgt = filter_lines(docs, src_lines, tgt_lines)

    # Write result
    with open(output_file, 'w', encoding='utf-8') as out_fh:
        for i, doc in enumerate(out_docs):
            out_fh.write('\t'.join([doc, out_src[i], out_tgt[i]]) + '\n')

    logging.info('Done')


if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--input", help="Input filename, format doc_id"
                                        "__@delimeter@__src_sent"
                                        "__@delimeter@__tgt_sent")
    parser.add_argument("--output", help="Filename for cleaned data")

    args = parser.parse_args()

    # Clean the files
    filter_file(args.input, args.output)
