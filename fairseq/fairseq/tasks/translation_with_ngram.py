# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from omegaconf import II

import numpy as np
from fairseq import metrics, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
import torch
import sys

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@dataclass
class TranslationWithNgramConfig(FairseqDataclass):
    ngram_dist_size: int = field(
        default=100, metadata={"help": "size of ngram distribution"}
    )
    ngram_alpha: float = field(
        default=0.1, metadata={"help": "ngram alpha for evaluation"}
    )
    min_ngram_alpha: float = field(
        default=0.0, metadata={"help": "minimum ngram alpha"}
    )
    ngram_alpha_decay_rate: float = field(
        default=0.0, metadata={"help": "ngram_alpha = max(0, ngram_alpha - dr * update_num)"}
    )
    ngram_min_dist_prob: float = field(
        default=1.0, metadata={"help": "size of ngram distribution"}
    )
    ngram_generation_model_cache: Optional[str] = field(
        default=None, metadata={"help": "path to the cache of ngram generation model"}
    )
    data2clst_map_path: Optional[str] = field(
        default=None, metadata={"help": "path to data2clst_map"}
    )
    ngram_module_path: Optional[str] = field(
        default=None, metadata={"help": "path to the ngram module"}
    )
    eval_ngram_min_ctxlen: Optional[int] = field(
        default=0, metadata={"help": "path to the ngram module"}
    )
    eval_ngram_min_accum_prob: Optional[float] = field(
        default=0.0, metadata={"help": "path to the ngram module"}
    )
    ngram_warmup_updates: Optional[int] = field(
        default=0, metadata={"help": "path to the ngram module"}
    )
    ngram_ctxlen_p: Optional[str] = field(
        default="", metadata={"help": "distribution of context lengths"}
    )
    input_with_ctxlen: bool = field(
        default=False, metadata={"help": "pass the information of ctxlen to the model, which is need by ngram_pointer"}
    )
    weighted_ctxlen: bool = field(
        default=False, metadata={"help": "whether use the weighted context length"}
    )
    train_with_limited_ctxlen: bool = field(
        default=False, metadata={"help": "limit the matched length of context"}
    )
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    load_alignments: bool = field(
        default=False, metadata={"help": "load the binarized alignments"}
    )
    left_pad_source: bool = field(
        default=True, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    upsample_primary: int = field(
        default=-1, metadata={"help": "the amount of upsample primary dataset"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
            "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")

    # options for reporting BLEU during validation
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_args: Optional[str] = field(
        default="{}",
        metadata={
            "help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
            "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None,
        metadata={
            "help": "remove BPE before computing BLEU",
            "argparse_const": "@@ ",
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )


@register_task("translation_with_ngram", dataclass=TranslationWithNgramConfig)
class TranslationWithNgramTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: TranslationWithNgramConfig

    def __init__(self, cfg: TranslationWithNgramConfig, src_dict, tgt_dict, ngram_generation_model):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        if cfg.data2clst_map_path:
            data2clst_map_path_list = cfg.data2clst_map_path.split(";")
            if len(data2clst_map_path_list) > 1:
                self.valid_data2clst_map = torch.load(data2clst_map_path_list[1])
            else:
                self.valid_data2clst_map = None
            self.data2clst_map = torch.load(data2clst_map_path_list[0])
        else:
            self.data2clst_map = None
            self.valid_data2clst_map = None
        self.train_with_limited_ctxlen = cfg.train_with_limited_ctxlen
        self.ngram_generation_model = ngram_generation_model
        self.base_ctxlens = list(range(ngram_generation_model.context_len+1))
        self.ngram_model_num = ngram_generation_model.size
        self.ngram_warmup_updates = cfg.ngram_warmup_updates
        self.ngram_dist_size = cfg.ngram_dist_size
        self.eval_ngram_min_ctxlen = cfg.eval_ngram_min_ctxlen
        self.eval_ngram_min_accum_prob = cfg.eval_ngram_min_accum_prob
        self.input_with_ctxlen = cfg.input_with_ctxlen
        self.ngram_context_len = self.ngram_generation_model.context_len
        self.ngram_min_dist_prob = cfg.ngram_min_dist_prob
        self.weighted_ctxlen = cfg.weighted_ctxlen
        self.ngram_alpha = cfg.ngram_alpha
        self.min_ngram_alpha = cfg.min_ngram_alpha
        self.ngram_alpha_decay_rate = cfg.ngram_alpha_decay_rate
        if cfg.ngram_ctxlen_p:
            self.ngram_ctxlen_p = [0.0] + [float(it) for it in cfg.ngram_ctxlen_p.strip().split(";")]
        else:
            context_len = self.ngram_generation_model.context_len 
            self.ngram_ctxlen_p = [0.0] + [1/context_len for _ in range(context_len)]

    @classmethod
    def setup_task(cls, cfg: TranslationWithNgramConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        sys.path.append(cfg.ngram_module_path)
        import c_fast_model
        ngram_generation_model = c_fast_model.CMultiFastGenerationModel.from_cache(
            cfg.ngram_generation_model_cache, tgt_dict.eos(), tgt_dict.unk())

        return cls(cfg, src_dict, tgt_dict, ngram_generation_model)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def valid_step(self, sample, model, criterion):
        if self.ngram_warmup_updates <= 0 and self.ngram_alpha > 0:
            sample = self._prepare_ngram(sample, is_training=model.training)
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_bleu:

            def sum_logs(key):
                import torch
                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))
            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    try:
                        from sacrebleu.metrics import BLEU
                        comp_bleu = BLEU.compute_bleu
                    except ImportError:
                        # compatibility API for sacrebleu 1.x
                        import sacrebleu
                        comp_bleu = sacrebleu.compute_bleu

                    fn_sig = inspect.getfullargspec(comp_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = comp_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)
            
                metrics.log_derived("bleu", compute_bleu)
            # else:
            #      def zero_bleu(meters):
            #          return 0.0

            #      metrics.log_derived("bleu", zero_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def _prepare_ngram(self, sample, is_training):
        if is_training:
            if self.data2clst_map is None:
                datasize = len(self.datasets["train"])
                shard_size = (datasize // self.ngram_model_num) + 1
                model_ids = [it // shard_size for it in sample['id'].tolist()]
            else:
                model_ids = [self.data2clst_map[it] for it in sample['id'].tolist()]
        else:
            if self.valid_data2clst_map is None:
                model_ids = None
            else:
                model_ids = [self.data2clst_map[it] for it in sample['id'].tolist()]
        input_ids = sample["net_input"]['prev_output_tokens']
        device = input_ids.device
        ctxlens = None
        if is_training and (self.input_with_ctxlen or self.train_with_limited_ctxlen):
            bs, ts = input_ids.size()
            ctxlens = np.random.choice(self.base_ctxlens, bs*ts, replace=True, p=self.ngram_ctxlen_p).reshape((bs, ts)).tolist()
        word3d, prob3d, ctxlen3d = self.ngram_generation_model.batch_dist(
            input_ids.tolist(), dist_size=self.ngram_dist_size, min_dist_prob=self.ngram_min_dist_prob,
            bos=False, model_ids=model_ids, require_match_len=True, ctxlens=ctxlens)
        
        ngram_word = torch.tensor(word3d, device=device, dtype=torch.long)
        ngram_dist = torch.tensor(prob3d, device=device, dtype=torch.float)
        mask = (ngram_dist > 0)
        ngram_dist = torch.pow(10, ngram_dist)
        ngram_dist[mask] = 0
        sample["ngram_word"] = ngram_word
        sample["ngram_dist"] = ngram_dist
        if self.weighted_ctxlen:
            ngram_ctxlen = torch.tensor(ctxlen3d, device=device, dtype=torch.float)
            ngram_ctxlen = torch.sum(ngram_ctxlen * ngram_dist, dim=-1)
        else:
            ngram_ctxlen = torch.tensor([[it[0] for it in sent] for sent in ctxlen3d], device=device, dtype=torch.int)
        if self.input_with_ctxlen:
            sample["net_input"]["ngram_cxtlen"] = ngram_ctxlen
        sample["ngram_ctxlen_mask"] = torch.logical_or(
            ngram_ctxlen < self.eval_ngram_min_ctxlen,
            ngram_dist.sum(dim=-1) < self.eval_ngram_min_accum_prob
        )
        sample["ngram_alpha"] = ngram_dist.new_ones(1) * self.ngram_alpha
        if not is_training:
            tgt = sample["target"].unsqueeze(-1).repeat(1, 1, self.ngram_dist_size)
            gd_match = tgt.eq(ngram_word).sum(dim=-1)
            sample["gd_match"] = gd_match
            sample["ngram_ctxlen"] = ngram_ctxlen
        return sample

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        if self.ngram_alpha_decay_rate > 0:
            self.ngram_alpha = max(self.min_ngram_alpha, self.ngram_alpha - self.ngram_alpha_decay_rate)
        if self.ngram_warmup_updates > 0:
            self.ngram_warmup_updates -= 1
        elif self.ngram_alpha > 0:
            sample = self._prepare_ngram(sample, is_training=model.training)
        return super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad=False)

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens,
                constraints=constraints, ngram_generation_model=self.ngram_generation_model,
                ngram_alpha=self.ngram_alpha, ngram_dist_size=self.ngram_dist_size,
                ngram_min_dist_prob=self.ngram_min_dist_prob
            )