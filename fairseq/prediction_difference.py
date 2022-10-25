#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""

import logging
import math
import os
import sys
from argparse import Namespace
from typing import Iterable, List, Optional

import torch
import fairseq
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter
from fairseq.sequence_scorer import SequenceScorer
from omegaconf import DictConfig


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.eval_lm")


def gather_target_probs(probs, target):
    probs = probs.gather(
        dim=2,
        index=target.unsqueeze(-1),
    )
    return probs


def pred_diff(
    model,
    batch_iterator,
    source_dictionary,
    sample_postprocessor=None,
    ngram_alpha=0
):
    device = next(model.parameters()).device
    model.eval()
    res = []
    max_sample_num = 16
    with torch.no_grad():
        for idx, sample in enumerate(batch_iterator):
            if idx >= max_sample_num:
                break
            if sample_postprocessor is not None:
                sample = sample_postprocessor(sample, is_training=False)
            if "ngram_alpha" in sample:
                sample["ngram_alpha"] = sample["ngram_alpha"].new_ones(1) * ngram_alpha
            sample = utils.move_to_cuda(sample, device=device)
            prev_output_tokens = sample["net_input"]["src_tokens"]
            orig_target = sample["target"]
            bs, slen = prev_output_tokens.size()
            net_output = model(**sample["net_input"])
            orig_prob = gather_target_probs(model.get_normalized_probs(
                net_output, log_probs=False, sample=sample
            ).data, orig_target).squeeze(-1).unsqueeze(1).repeat(1, slen, 1)

            pd_mask_backup = torch.zeros_like(prev_output_tokens).bool()
            for i in range(slen):
                pd_mask = pd_mask_backup.clone()
                pd_mask[:, i] = 1
                sample["net_input"]["pd_mask"] = pd_mask
                net_output = model(**sample["net_input"])
                curr_prob = gather_target_probs(model.get_normalized_probs(
                    net_output, log_probs=False, sample=sample
                ).data, orig_target).squeeze(-1)
                orig_prob[:, i, :] -= curr_prob
            
            for bi in range(bs):
                tgt_word_list = [source_dictionary[orig_target[bi][i].item()] for i in range(slen)]
                src_word_list = [source_dictionary[prev_output_tokens[bi][i].item()] for i in range(slen)]
                rank = []
                # if slen >= 20:
                #     import pdb; pdb.set_trace()
                for sj in range(slen):
                    values, indices = torch.topk(orig_prob[bi, :sj+1, sj], k=1)
                    rank.append(indices.item())
                res.append((sample['id'][bi], orig_prob[bi], tuple(src_word_list), tuple(tgt_word_list), tuple(rank)))

    return res


def main(cfg: DictConfig, **unused_kwargs):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    logger.info(cfg)

    if cfg.eval_lm.context_window > 0:
        # reduce tokens per sample by the required context window size
        cfg.task.tokens_per_sample -= cfg.eval_lm.context_window

    # Initialize the task using the current *cfg*
    task = tasks.setup_task(cfg.task)

    if "ngram_generation_model_cache" in cfg.task:
        sample_postprocessor = task._prepare_ngram
    else:
        sample_postprocessor = None

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=eval(cfg.common_eval.model_overrides),
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
        task=task,
    )

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    # Optimize ensemble for generation and set the source and dest dicts on the model
    # (required by scorer)
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    assert len(models) > 0

    logger.info(
        "num. model params: {:,}".format(sum(p.numel() for p in models[0].parameters()))
    )

    # Load dataset splits
    task.load_dataset(cfg.dataset.gen_subset)
    dataset = task.dataset(cfg.dataset.gen_subset)
    logger.info(
        "{} {} {:,} examples".format(
            cfg.task.data, cfg.dataset.gen_subset, len(dataset)
        )
    )

    itr = task.eval_lm_dataloader(
        dataset=dataset,
        max_tokens=cfg.dataset.max_tokens or 36000,
        batch_size=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            *[model.max_positions() for model in models]
        ),
        num_shards=max(
            cfg.dataset.num_shards,
            cfg.distributed_training.distributed_world_size,
        ),
        shard_id=max(
            cfg.dataset.shard_id,
            cfg.distributed_training.distributed_rank,
        ),
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
        context_window=cfg.eval_lm.context_window,
    )

    itr = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    results = pred_diff(
        model=models[0],
        batch_iterator=itr,
        source_dictionary=task.source_dictionary,
        sample_postprocessor=sample_postprocessor,
        ngram_alpha=cfg.eval_lm.eval_ngram_alpha
    )

    if cfg.eval_lm.pd_save_path:
        torch.save(results, cfg.eval_lm.pd_save_path)

    return results


def cli_main():
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)


if __name__ == "__main__":
    cli_main()
