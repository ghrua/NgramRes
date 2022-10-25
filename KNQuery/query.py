import torch
import logging
import argparse
from tqdm import tqdm
from utils import load_tokenizer, load_text_data
from model import LanguageModel, GenerationModel
import c_fast_model
import psutil
from os.path import basename


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="")
    parser.add_argument("--lm_path", default="")
    parser.add_argument("--tokenizer_path", default="")
    parser.add_argument("--sos_token", default="<eos>")
    parser.add_argument("--unk_token", default="<unk>")
    parser.add_argument("--cache_path", default="")
    parser.add_argument("--pruneset", default="")
    parser.add_argument("--mode", default="fast")
    return parser.parse_args()


def main(args):
    sys_mem = psutil.virtual_memory().used / (1024 ** 3)
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer_path)
    logger.info("Loading input data...")
    # data = load_text_data(args.input_path, tokenizer)
    logger.info("Loading n-gram LM...")
    sos_id, unk_id = tokenizer("{} {}".format(args.sos_token, args.unk_token))
    # lm = LanguageModel.from_pretrained(args.lm_path, sos_id, unk_id)
    # print("Corpus PPL: {:.3f}".format(""))
    # if args.mode == "fast":
    #     # Faster Model
    #     lm = c_fast_model.CFastLanguageModel.from_pretrained(args.lm_path, sos_id, unk_id, args.cache_path)
    # else:
    #     # Model
    #     lm = LanguageModel.from_pretrained(args.lm_path, sos_id, unk_id)
    # logger.info("Calculating by {} model...".format(args.mode))
    # logger.info("PPL for {}: {:.4f}".format(basename(args.input_path), lm.corpus_ppl(data)))
    # logger.info("Memory ussage for {} model is {:.1f}".format(args.mode, psutil.virtual_memory().used / 1024 ** 3))
    
    ###########
    # DIST
    ###########
    pruneset = None
    if args.pruneset:
        pruneset = torch.load(args.pruneset)
    lm = c_fast_model.CFastGenerationModel.from_scratch(args.lm_path, sos_id, unk_id, args.cache_path, prune_ctxset=pruneset)
    # lm = c_fast_model.CMultiFastGenerationModel.from_cache(args.cache_path, sos_id, unk_id)
    logger.info("Calculating...")
    # for sent in data:
    # ctxlens = [[1, 1, 1, 1, 1]]
    # word3d, prob3d, ctxlen3d = lm.batch_dist(data, dist_size=10, bos=False, require_match_len=True, ctxlens=ctxlens)
    # print(word3d)
    # print(prob3d)
    # print(len(word3d), len(word3d[0]), len(word3d[0][0]))
    # print(word3d, prob3d)
    logger.info("Memory ussage for {} model is {:.1f}".format(args.mode, psutil.virtual_memory().used / 1024 ** 3 - sys_mem))
    # logger.info("PPL for {}: {:.4f}".format(basename(args.input_path), lm.corpus_ppl(data)))

    
if __name__ == "__main__":
    main(parse_args())
