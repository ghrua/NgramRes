# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from torch import Tensor


class FairseqDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.onnx_trace = False
        self.adaptive_softmax = None


    def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def output_layer(self, features, **kwargs):
        """
        Project features to the default output size, e.g., vocabulary size.

        Args:
            features (Tensor): features returned by *extract_features*.
        """
        raise NotImplementedError

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            if "ngram_word" in sample and "ngram_dist" in sample:
                ngram_word = sample["ngram_word"]
                ngram_dist = sample["ngram_dist"]
                ngram_ctxlen_mask = sample["ngram_ctxlen_mask"]
                if isinstance(sample["ngram_alpha"], float):
                    ngram_alpha = sample["ngram_alpha"]
                else:
                    ngram_alpha = sample["ngram_alpha"].item()
                out = self.adaptive_softmax.get_log_prob_with_ngram(
                    net_output[0], target=target, ngram_word=ngram_word,
                    ngram_dist=ngram_dist, ngram_ctxlen_mask=ngram_ctxlen_mask,
                    ngram_alpha=ngram_alpha)
            else:
                out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        ngram_gate = net_output[1]["ngram_gate"] if "ngram_gate" in net_output[1] else None
        if sample is not None and "ngram_word" in sample and "ngram_dist" in sample:
            if isinstance(sample["ngram_alpha"], float):
                ngram_alpha = sample["ngram_alpha"]
            else:
                ngram_alpha = sample["ngram_alpha"].item()
            ngram_logits = torch.scatter_add(torch.zeros_like(logits),
                                    dim=-1, index=sample["ngram_word"],
                                    src=sample["ngram_dist"].to(logits.dtype))
            ngram_logits += ngram_logits.eq(0).to(logits.dtype) * 1e-7
            ngram_logits = torch.log(ngram_logits)
            ngram_logits = ngram_logits - ngram_logits.mean(dim=-1, keepdim=True)
            if "ngram_ctxlen_mask" in sample:
                ngram_logits[sample["ngram_ctxlen_mask"]] = 0
            if ngram_gate is not None:
                ngram_logits = ngram_gate * ngram_logits
            logits = logits + ngram_alpha * ngram_logits

        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return 1e6  # an arbitrary large number

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code."""
        return state_dict

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True
