#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def separate(loc, input_files, src_lang, tgt_lang, n_clusters=4):
    # iterate over input files
    for filename in input_files:
        # create a dictionary of src ang tgt lines in each cluster
        random_sep_lines = {str(i): {'src': [], 'tgt': []}
                            for i in range(n_clusters)}
        # open the input files to read lines from
        # and a file into which to write the cluster indices
        with open(f'{loc}/{filename}.{src_lang}',
                  'r', encoding='utf8') as in_src_fh, \
                open(f'{loc}/{filename}.{tgt_lang}',
                     'r', encoding='utf8') as in_tgt_fh, \
                open(f'{loc}/{filename}.rand_clusters_{n_clusters}.txt',
                     'w', encoding='utf8') as out_cluster_fh:
            # iterate over input lines
            for src_line in in_src_fh:
                tgt_line = in_tgt_fh.readline()
                src, tgt = src_line.strip(), tgt_line.strip()
                # generate random cluster number for current line
                rand_cl = str(np.random.choice(range(n_clusters)))
                # save this cluster number
                out_cluster_fh.write(str(rand_cl) + '\n')
                # add src and tgt lines to the corresponding dictionary
                random_sep_lines[rand_cl]['src'].append(src)
                random_sep_lines[rand_cl]['tgt'].append(tgt)

        # iterate over clusters, write each cluster into a separate file
        for cluster in [str(i) for i in range(n_clusters)]:
            # open the src output file
            with open(
                    f'{loc}/{filename}.rand_clusters_{n_clusters}_cluster'
                    f'{cluster}.{src_lang}',
                    'w', encoding='utf8') as out_src_fh:
                # write all src lines from the dictionary
                out_src_fh.writelines(
                    [line + '\n' for line in random_sep_lines[cluster]['src']])
            # same for tgt
            with open(
                    f'{loc}/{filename}.rand_clusters_{n_clusters}_cluster'
                    f'{cluster}.{tgt_lang}',
                    'w', encoding='utf8') as out_tgt_fh:
                out_tgt_fh.writelines(
                    [line + '\n' for line in random_sep_lines[cluster]['tgt']])


def restore_cluster_order(loc, input_files, src_lang, tgt_lang, n_clusters=4):
    # iterate over input lines
    for filename in input_files:
        # create src and tgt lines where we will save lines in cluster order
        # (first the whole cluster 0, then 1, etc.)
        with open(f'{loc}/{filename}.rand_clusters_{n_clusters}_'
                  f'clusterorder.{src_lang}',
                  'w', encoding='utf8') as src_clusterorder_fh, \
                open(f'{loc}/{filename}.rand_clusters_{n_clusters}_'
                     f'clusterorder.{tgt_lang}',
                     'w', encoding='utf8') as tgt_clusterorder_fh:
            # iterate over clusters
            for cluster in [str(i) for i in range(n_clusters)]:
                # open the cluster file and write all of its lines into
                # the cluster order file
                with open(
                        f'{loc}/{filename}.rand_clusters_{n_clusters}_cluster'
                        f'{cluster}.{src_lang}',
                        'r', encoding='utf8') as cluster_src_fh:
                    for line in cluster_src_fh:
                        src_clusterorder_fh.write(line.strip() + '\n')
                with open(
                        f'{loc}/{filename}.rand_clusters_{n_clusters}_cluster'
                        f'{cluster}.{tgt_lang}',
                        'r', encoding='utf8') as cluster_tgt_fh:
                    for line in cluster_tgt_fh:
                        tgt_clusterorder_fh.write(line.strip() + '\n')


if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--path_to_files", help="Path to input files location")
    parser.add_argument("--input_files", help="Input filename",
                        nargs='+')
    parser.add_argument("--src_lang", help="Source language",
                        default='en')
    parser.add_argument("--tgt_lang", help="Source language",
                        default='et')
    parser.add_argument("--n_clusters", type=int, default=4,
                        help="Numbers of clusters")

    args = parser.parse_args()

    # separate into random clusters
    separate(loc=args.path_to_files, input_files=args.input_files,
             src_lang=args.src_lang, tgt_lang=args.tgt_lang,
             n_clusters=args.n_clusters)
    logging.info(f"Result files saved into {args.path_to_files}/FILENAME."
                 f"rand_clusters_{args.n_clusters}_clusterN.LANG, indices in"
                 f"{args.path_to_files}/FILENAME.rand_clusters_"
                 f"{args.n_clusters}.txt")

    # save files in cluster order
    restore_cluster_order(loc=args.path_to_files, input_files=args.input_files,
                          src_lang=args.src_lang, tgt_lang=args.tgt_lang,
                          n_clusters=args.n_clusters)
    logging.info(f"Lines in cluster order saved into {args.path_to_files}/"
                 f"FILENAME.rand_clusters_{args.n_clusters}_clusterorder."
                 f"LANG")