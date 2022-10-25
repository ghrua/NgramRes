import torch
import logging
import argparse
from tqdm import tqdm
from utils import load_tokenizer
from os.path import basename


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="")
    parser.add_argument("--tokenizer_path", default="")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--unk_token", default="<unk>")
    return parser.parse_args()


def main(args):
    tokenizer = load_tokenizer(args.tokenizer_path)
    unk_id = tokenizer(args.unk_token)[0]
    with open(args.input_path) as fin, open(args.output_path, 'w') as fout:
        for line in fin:
            word_ids = tokenizer(line)
            words = line.strip().split()
            new_line = []
            assert len(words) == len(word_ids)
            for wi, w in zip(word_ids, words):
                if wi == unk_id:
                    new_line.append("<UNK>")
                else:
                    new_line.append(w)
            fout.write(" ".join(new_line) + "\n")


if __name__ == "__main__":
    main(parse_args())