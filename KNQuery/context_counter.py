import torch
import logging
import argparse
from tqdm import tqdm
from utils import load_tokenizer
from os.path import basename
from collections import Counter
from glob import glob
import numpy as np


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="")
    parser.add_argument("--max_ctxlen", type=int, default=4)
    parser.add_argument("--tokenizer_path", default="")
    parser.add_argument("--output_path", default="")
    return parser.parse_args()


def main(args):
    tokenizer = load_tokenizer(args.tokenizer_path)
    final_ans = []
    for i in range(1, args.max_ctxlen+1):
        ctx_map = dict()
        counter = Counter()
        n = i+1
        logger.info("Counting context with length {}".format(i))
        with open(args.input_path) as fin:
            for line in fin:
                new_line = "</s> {} </s>".format(line)
                word_ids = tokenizer(new_line)
                slen = len(word_ids)
                ngrams = []
                for k in range(n-1, slen):
                    key = tuple(word_ids[k+1-n:k+1])
                    ngrams.append(key)
                counter.update(ngrams)
        for key, freq in counter.most_common():
            ctx = key[:-1]
            if ctx in ctx_map:
                ctx_map[ctx].append((key[-1], freq))
            else:
                ctx_map[ctx] = [(key[-1], freq)]
        new_ctx_map = dict()
        for ctx, cands in ctx_map.items():
            ctx_freq = sum([it[1] for it in cands])
            prob = [it[1] / ctx_freq for it in cands]
            entropy = -sum(it * np.log2(it+1e-9) for it in prob)
            ctx_div = len(cands)
            new_ctx_map[ctx] = (ctx_freq, ctx_div, entropy)
        logger.info("{} contexts with length {} have been found".format(len(new_ctx_map), i))
        final_ans.append(new_ctx_map)
    logger.info("Totally {} contexts have been found".format(sum([len(it) for it in final_ans])))
    logger.info("Saving context information to {}".format(args.output_path))
    torch.save(final_ans, args.output_path)


if __name__ == "__main__":
    main(parse_args())