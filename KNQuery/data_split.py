import torch
import logging
import argparse
from tqdm import tqdm
from utils import load_tokenizer
from os.path import basename
from collections import Counter
from glob import glob
import numpy as np
import random


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="")
    parser.add_argument("--nsplit", type=int, default=4)
    parser.add_argument("--data2clst_map_path", default="")
    parser.add_argument("--seed", type=int, default=10086)
    return parser.parse_args()


def gather_data(grouped_data, shard):
    ngroup = len(grouped_data)
    for i in range(ngroup):
        if i == shard:
            continue
        for d in grouped_data[i]:
            yield d


def main(args):
    random.seed(args.seed)
    data = []
    with open(args.input_path) as fin:
        for line in fin:
            data.append(line)
    ntotal = len(data)
    split_size = ntotal // args.nsplit + 1
    index = list(range(ntotal))
    random.shuffle(index)
    grouped_data = [[] for _ in range(args.nsplit)]
    data2clst_map = dict()
    for i in range(args.nsplit):
        s, e = i*split_size, min((i+1) * split_size, ntotal)
        for j in range(s, e):
            grouped_data[i].append(data[index[j]])
            data2clst_map[index[j]] = i
    logger.info("Data is grouped...")
    for i in range(args.nsplit):
        with open("{}.shard{}".format(args.input_path, i), 'w') as fout:
            for d in gather_data(grouped_data, i):
                fout.write(d)
    logger.info("Data is sharded...")
    torch.save(data2clst_map, args.data2clst_map_path)


if __name__ == "__main__":
    main(parse_args())