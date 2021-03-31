#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import json
import logging
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def extract_metrics(model_path, out_path):
    out_dict = {'epoch': [], 'loss': [], 'ppl': [], 'bleu': []}

    # compile regular expressions
    valid_info = re.compile(r"""INFO \| valid \| {.+}""")
    epoch = re.compile(r"""\"epoch\": ([0-9]+)""")
    val_loss = re.compile(r"""\"valid_loss\": \"([0-9]+\.?[0-9]*)\"""")
    val_ppl = re.compile(r"""\"valid_ppl\": \"([0-9]+\.?[0-9]*)\"""")
    val_bleu = re.compile(r"""\"valid_bleu\": \"([0-9]+\.?[0-9]*)\"""")

    # iterate over log lines
    with open(model_path + '/log.out', 'r', encoding='utf8') as log_fh:
        for line in log_fh:
            if valid_info.search(line):
                out_dict['epoch'].append(epoch.search(line).group(1))
                out_dict['loss'].append(val_loss.search(line).group(1))
                out_dict['ppl'].append(val_ppl.search(line).group(1))
                out_dict['bleu'].append(val_bleu.search(line).group(1))

    # write result into a json file in model directory
    with open(model_path + 'val_metrics.json', 'w', encoding='utf8') as out_fh:
        json.dump(out_dict, out_fh, indent=4)

    if os.path.isfile(out_path):
        with open(out_path, 'r', encoding='utf8') as common_out:
            prior_dicts = json.load(common_out)
    else:
        prior_dicts = []

    prior_dicts.append({model_path: out_dict})

    with open(out_path, 'w', encoding='utf8') as common_out:
        json.dump(prior_dicts, common_out, indent=4)

    return 0


if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--modeldir", type=str, required=True,
                        help="Model directory containing the log file")
    parser.add_argument("--output", type=str,
                        help="Output file", default="metrics.json")

    args = parser.parse_args()

    if not os.path.isfile(args.modeldir + '/log.out'):
        print(f"File {args.modeldir}/log.out does not exist, exiting")
    else:
        extract_metrics(model_path=args.modeldir, out_path=args.output)
