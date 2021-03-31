#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import json
import logging
from shutil import copyfile
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def duplicate_best_checkpoint(model_path):
    best_epoch = 0
    max_bleu = 0

    # compile regular expressions
    valid_info = re.compile(r"""INFO \| valid \| {.+}""")
    epoch = re.compile(r"""\"epoch\": ([0-9]+)""")
    val_bleu = re.compile(r"""\"valid_bleu\": \"([0-9]+\.?[0-9]*)\"""")

    # iterate over log lines
    with open(model_path + '/log.out', 'r', encoding='utf8') as log_fh:
        for line in log_fh:
            if valid_info.search(line):
                current_epoch = int(epoch.search(line).group(1))
                current_bleu = float(val_bleu.search(line).group(1))
                if current_bleu > max_bleu:
                    max_bleu = current_bleu
                    best_epoch = current_epoch

    # copy best checkpoint into a new file
    copyfile("{0}/checkpoint{1}.pt".format(model_path, str(best_epoch)),
             "{0}/checkpoint_best_dev_bleu.pt".format(model_path))
    logging.info("Copied {0}/checkpoint{1}.pt into \
                  {0}/checkpoint_best_dev_bleu.pt".
                 format(model_path, str(best_epoch)))
    return 0


if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--modeldir", type=str, required=True,
                        help="Model directory containing the log file")
    # parser.add_argument("--output", type=str,
    #                     help="Output file", default="metrics.json")

    args = parser.parse_args()

    if not os.path.isfile(args.modeldir + '/log.out'):
        logging.info(f"File {args.modeldir}/log.out does not exist, exiting")
    else:
        duplicate_best_checkpoint(model_path=args.modeldir)
