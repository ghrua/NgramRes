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
    parser.add_argument("--query_out_path", default="")
    return parser.parse_args()


def parse_line(line, word_list, wid_list, length_list, prob_list):
    ws, wids, lengths, probs = [], [], [], []
    for it in line.strip().split('\t')[:-1]:
        idx = 0
        while True:
            if it[idx] == "=" and ('0' <= it[idx+1] <= '9'):
                break
            idx += 1
        w = it[:idx]
        info = it[idx+1:]
        ws.append(w)
        info = [float(j) for j in info.strip().split()]
        wids.append(info[0])
        lengths.append(info[1])
        probs.append(info[2])
    word_list.append(ws)
    wid_list.append(wids)
    length_list.append(lengths)
    prob_list.append(probs)


def read_query_out(fname):
    word_list, wid_list, length_list, prob_list = [], [], [], []
    ppl_with_oov = None
    ppl_wo_oov = None
    oov_num = None
    tokens = None
    line_no = 0
    with open(fname) as fin:
        line = fin.readline()
        while line:
            if line.startswith("Perplexity"):
                ppl_with_oov = float(line.strip().split('\t')[-1])
                ppl_wo_oov = float(fin.readline().strip().split('\t')[-1])
                oov_num = int(fin.readline().strip().split('\t')[-1])
                tokens = int(fin.readline().strip().split('\t')[-1])
                break
            parse_line(line, word_list, wid_list, length_list, prob_list)
            line = fin.readline()
    return word_list, wid_list, length_list, prob_list, ppl_with_oov, ppl_wo_oov, oov_num, tokens
                

def main(args):
    word_list, wid_list, length_list, prob_list, ppl_with_oov, ppl_wo_oov, oov_num, tokens = read_query_out(args.query_out_path)
    num_tokens = 0
    num_words_wo_bpe = 0
    for it in word_list:
        num_tokens += len(it)
        num_words_wo_bpe += len(" ".join(it).replace("@@ ", "").strip().split())
    assert num_tokens == tokens
    
    logprob = 0
    for it in prob_list:
        for p in it:
            logprob += p
    
    print("PPL w/ bpe: {:.2f}".format(10 ** (-logprob / num_tokens)))
    print("PPL w/o bpe: {:.2f}".format(10 ** (-logprob / num_words_wo_bpe)))
    

if __name__ == "__main__":
    main(parse_args())
