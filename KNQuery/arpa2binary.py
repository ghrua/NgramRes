import torch
import logging
import argparse
from tqdm import tqdm
from utils import load_tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arpa", default="")
    parser.add_argument("--binary", default="")
    parser.add_argument("--tokenizer_path", default="")
    parser.add_argument("--no_replace", action="store_true")
    parser.add_argument("--eos_token", default="<eos>")
    parser.add_argument("--unk_token", default="<unk>")
    return parser.parse_args()


def read_special_tokens(fin, tokenizer, eos_token):
    """
    merge eos and sos
    """
    ret = []
    unk_line = fin.readline().strip().split("\t")
    sos_line = fin.readline().strip().split("\t")
    eos_line = fin.readline().strip().split("\t")
    ret.append((float(unk_line[0]), tokenizer(unk_line[1]), float(unk_line[2])))
    ret.append((float(eos_line[0]), tokenizer(eos_token), float(sos_line[2])))
    return ret

def replace_special_tokens(ws, eos_token, unk_token):
    if "<s>" in ws:
        ws = ws.replace("<s>", eos_token)
    if "</s>" in ws:
        ws = ws.replace("</s>", eos_token)
    if "<UNK>" in ws:
        ws = ws.replace("<UNK>", unk_token)
    return ws


def main(args):
    replace = not args.no_replace
    if args.tokenizer_path:
        tokenizer = load_tokenizer(args.tokenizer_path)
    else:
        tokenizer = lambda x: x
    data = []
    sizes = []
    with open(args.arpa) as fin:
        line = fin.readline().strip()
        assert line == "\data\\", "Error format"
        while True:
            line = fin.readline().strip()
            if not line:
                break
            assert line.startswith("ngram"), "Error format"
            data.append([])
            sizes.append(int(line.split()[1].split("=")[1]))
        print(sizes)
        sidx = -1
        while True:
            line = fin.readline()
            if not line:
                break
            if not line.strip():
                continue
            if line.startswith("\\"):
                if "gram" in line:
                    sidx += 1
                    if sidx == 0:
                        data[sidx] += read_special_tokens(fin, tokenizer, args.eos_token)
                    logger.info("Loading {}-gram...".format(sidx+1))
                continue
            it = line.strip().split("\t")
            if len(it) > 2: 
                p, ws, b = float(it[0]), it[1], float(it[2])
                ws = replace_special_tokens(ws, args.eos_token, args.unk_token) if replace else ws
                data[sidx].append((p, tokenizer(ws), b))
            else:
                p, ws = float(it[0]), it[1]
                ws = replace_special_tokens(ws, args.eos_token, args.unk_token) if replace else ws
                data[sidx].append((p, tokenizer(ws)))

    logger.info("Saving...")
    torch.save(data, args.binary)


if __name__ == "__main__":
    main(parse_args())
