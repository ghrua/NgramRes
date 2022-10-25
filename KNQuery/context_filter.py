import torch
import logging
import argparse
from tqdm import tqdm
from utils import load_tokenizer
from os.path import basename
from collections import Counter
from glob import glob


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="")
    parser.add_argument("--prune", default="0;3;3;3")
    parser.add_argument("--max_ctxlen", type=int, default=4)
    parser.add_argument("--tokenizer_path", default="")
    parser.add_argument("--output_path", default="")
    return parser.parse_args()


def main(args):
    tokenizer = load_tokenizer(args.tokenizer_path)
    
    ctx_map = dict()
    final_ans = []
    prune_list = [int(it) for it in args.prune.strip().split(";")]
    if len(prune_list) < args.max_ctxlen:
        prune_num = prune_list[-1]
        prune_list = prune_list + [prune_num] * (args.max_ctxlen - len(prune_list))
    
    for i in range(1, args.max_ctxlen+1):
        ans = []
        counter = Counter()
        n = i+1
        prune_num = prune_list[i-1]
        if prune_num <= 0:
            logger.info("Skip context with length {}".format(i))
            continue
        logger.info("Pruning context with length {}".format(i))
        with open(args.input_path) as fin:
            for line in fin:
                new_line = "<s> {} </s>".format(line)
                word_ids = tokenizer(new_line)
                slen = len(word_ids)
                ngrams = []
                for i in range(n-1, slen):
                    key = tuple(word_ids[i+1-n:i+1])
                    ngrams.append(key)
                counter.update(ngrams)
        for key, freq in counter.most_common():
            ctx = key[:-1]
            if ctx in ctx_map:
                ctx_map[ctx].append((key[-1], freq))
            else:
                ctx_map[ctx] = [(key[-1], freq)]
        for ctx, cands in ctx_map.items():
            num = sum([it[1] for it in cands])
            if num <= prune_num:
                ans.append(ctx)
        logger.info("{} {}-grams\t{} contexts\t{} pruned contexts".format(
            len(counter), n, len(ctx_map), len(ans)
        ))
        final_ans += ans
    final_ans = set(final_ans)
    logger.info("{} contexts are pruned".format(len(final_ans)))
    torch.save(final_ans, args.output_path)

if __name__ == "__main__":
    main(parse_args())