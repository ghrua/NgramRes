# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import operator

import torch
import torch.nn.functional as F
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import nn


class TiedLinear(nn.Module):
    def __init__(self, weight, transpose):
        super().__init__()
        self.weight = weight
        self.transpose = transpose

    def forward(self, input):
        return F.linear(input, self.weight.t() if self.transpose else self.weight)


class TiedHeadModule(nn.Module):
    def __init__(self, weights, input_dim, num_classes, q_noise, qn_block_size):
        super().__init__()
        tied_emb, _ = weights
        self.num_words, emb_dim = tied_emb.size()

        self.word_proj = quant_noise(
            TiedLinear(tied_emb, transpose=False), q_noise, qn_block_size
        )
        if input_dim != emb_dim:
            self.word_proj = nn.Sequential(
                quant_noise(
                    nn.Linear(input_dim, emb_dim, bias=False), q_noise, qn_block_size
                ),
                self.word_proj,
            )

        self.class_proj = quant_noise(
            nn.Linear(input_dim, num_classes, bias=False), q_noise, qn_block_size
        )
        self.out_dim = self.num_words + num_classes

        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    def forward(self, input):
        inp_sz = functools.reduce(operator.mul, input.shape[:-1], 1)
        out = self._float_tensor.new(inp_sz, self.out_dim)
        out[:, : self.num_words] = self.word_proj(input.view(inp_sz, -1))
        out[:, self.num_words :] = self.class_proj(input.view(inp_sz, -1))
        return out


class AdaptiveSoftmax(nn.Module):
    """
    This is an implementation of the efficient softmax approximation for
    graphical processing units (GPU), described in the paper "Efficient softmax
    approximation for GPUs" (http://arxiv.org/abs/1609.04309).
    """

    def __init__(
        self,
        vocab_size,
        input_dim,
        cutoff,
        dropout,
        factor=4.0,
        adaptive_inputs=None,
        tie_proj=False,
        q_noise=0,
        qn_block_size=8,
    ):
        super().__init__()

        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]
        else:
            assert (
                vocab_size == cutoff[-1]
            ), "cannot specify cutoff larger than vocab size"

        output_dim = cutoff[0] + len(cutoff) - 1

        self.vocab_size = vocab_size
        self.cutoff = cutoff
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.input_dim = input_dim
        self.factor = factor
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size

        self.lsm = nn.LogSoftmax(dim=1)

        if adaptive_inputs is not None:
            self.head = TiedHeadModule(
                adaptive_inputs.weights_for_band(0),
                input_dim,
                len(cutoff) - 1,
                self.q_noise,
                self.qn_block_size,
            )
        else:
            self.head = quant_noise(
                nn.Linear(input_dim, output_dim, bias=False),
                self.q_noise,
                self.qn_block_size,
            )

        self._make_tail(adaptive_inputs, tie_proj)

        def init_weights(m):
            if (
                hasattr(m, "weight")
                and not isinstance(m, TiedLinear)
                and not isinstance(m, TiedHeadModule)
            ):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

        self.register_buffer("version", torch.LongTensor([1]))

    def _make_tail(self, adaptive_inputs=None, tie_proj=False):
        self.tail = nn.ModuleList()
        for i in range(len(self.cutoff) - 1):
            dim = int(self.input_dim // self.factor ** (i + 1))

            tied_emb, tied_proj = (
                adaptive_inputs.weights_for_band(i + 1)
                if adaptive_inputs is not None
                else (None, None)
            )

            if tied_proj is not None:
                if tie_proj:
                    proj = quant_noise(
                        TiedLinear(tied_proj, transpose=True),
                        self.q_noise,
                        self.qn_block_size,
                    )
                else:
                    proj = quant_noise(
                        nn.Linear(tied_proj.size(0), tied_proj.size(1), bias=False),
                        self.q_noise,
                        self.qn_block_size,
                    )
            else:
                proj = quant_noise(
                    nn.Linear(self.input_dim, dim, bias=False),
                    self.q_noise,
                    self.qn_block_size,
                )

            if tied_emb is None:
                out_proj = nn.Linear(
                    dim, self.cutoff[i + 1] - self.cutoff[i], bias=False
                )
            else:
                out_proj = TiedLinear(tied_emb, transpose=False)

            m = nn.Sequential(
                proj,
                nn.Dropout(self.dropout_module.p),
                quant_noise(out_proj, self.q_noise, self.qn_block_size),
            )

            self.tail.append(m)

    def upgrade_state_dict_named(self, state_dict, name):
        version_name = name + ".version"
        if version_name not in state_dict:
            raise Exception("This version of the model is no longer supported")

    def adapt_target(self, target):
        """
        In order to be efficient, the AdaptiveSoftMax does not compute the
        scores for all the word of the vocabulary for all the examples. It is
        thus necessary to call the method adapt_target of the AdaptiveSoftMax
        layer inside each forward pass.
        """

        target = target.view(-1)
        new_target = [target.clone()]
        target_idxs = []

        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
            new_target[0][mask] = self.cutoff[0] + i

            if mask.any():
                target_idxs.append(mask.nonzero(as_tuple=False).squeeze(1))
                new_target.append(target[mask].add(-self.cutoff[i]))
            else:
                target_idxs.append(None)
                new_target.append(None)

        return new_target, target_idxs

    def adapt_target_with_ngram(self, target, ngram_word, ngram_dist, ngram_ctxlen_mask):
        """
        In order to be efficient, the AdaptiveSoftMax does not compute the
        scores for all the word of the vocabulary for all the examples. It is
        thus necessary to call the method adapt_target of the AdaptiveSoftMax
        layer inside each forward pass.
        """

        def process_ngram_info(masked_ngram_word, masked_ngram_dist, lower_bound=0, upper_bound=1):
            dist_mask = masked_ngram_word.ge(lower_bound).mul(masked_ngram_word.lt(upper_bound))
            masked_ngram_word[torch.logical_not(dist_mask)] = 0
            masked_ngram_dist = masked_ngram_dist * dist_mask.float()
            masked_ngram_dist = masked_ngram_dist / (torch.sum(masked_ngram_dist, dim=-1, keepdim=True) + 1e-7)
            return masked_ngram_word, masked_ngram_dist

        batch_size, seq_len, dist_size = ngram_word.size()
        ngram_word = ngram_word.view(-1, dist_size)
        ngram_dist = ngram_dist.view(-1, dist_size)
        ngram_ctxlen_mask = ngram_ctxlen_mask.view(-1)
        target = target.view(-1)
        new_target = [target.clone()]
        target_idxs = []

        num_classes = len(self.cutoff) - 1
        masked_ngram_word, masked_ngram_dist =  process_ngram_info(
            ngram_word.clone(), ngram_dist.clone(), upper_bound=self.cutoff[0])
        new_ngram_word_list = [masked_ngram_word]
        new_ngram_dist_list = [masked_ngram_dist]
        new_ngram_ctxlen_mask_list = [ngram_ctxlen_mask]
        class_word = ngram_word.new(ngram_word.size(0), num_classes)
        class_dist = ngram_dist.new(ngram_dist.size(0), num_classes)
        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
            new_target[0][mask] = self.cutoff[0] + i
            class_word[:, i] = self.cutoff[0] + i
            class_dist_mask = ngram_word.ge(self.cutoff[i]).mul(ngram_word.lt(self.cutoff[i + 1]))
            class_dist[:, i] = torch.sum(ngram_dist * class_dist_mask.float(), dim=-1)
            if mask.any():
                target_idxs.append(mask.nonzero(as_tuple=False).squeeze(1))
                new_target.append(target[mask].add(-self.cutoff[i]))
                ngram_mask = mask.unsqueeze(-1).expand_as(ngram_word)
                masked_ngram_word, masked_ngram_dist = process_ngram_info(
                    ngram_word[ngram_mask].reshape(-1, dist_size).contiguous().add(-self.cutoff[i]) ,
                    ngram_dist[ngram_mask].reshape(-1, dist_size).contiguous(),
                    upper_bound=self.cutoff[i + 1] - self.cutoff[i]
                )
                new_ngram_ctxlen_mask_list.append(ngram_ctxlen_mask[mask])
                new_ngram_word_list.append(masked_ngram_word)
                new_ngram_dist_list.append(masked_ngram_dist)
            else:
                target_idxs.append(None)
                new_target.append(None)
                new_ngram_word_list.append(None)
                new_ngram_dist_list.append(None)
                new_ngram_ctxlen_mask_list.append(None)
        new_ngram_word_list[0] = torch.cat([new_ngram_word_list[0], class_word], dim=-1)
        new_ngram_dist_list[0] = torch.cat([new_ngram_dist_list[0], class_dist], dim=-1)
        return new_target, target_idxs, new_ngram_word_list, new_ngram_dist_list, new_ngram_ctxlen_mask_list

    def forward(self, input, target):
        """
        Args:
            input: (b x t x d)
            target: (b x t)
        Returns:
            2 lists: output for each cutoff section and new targets by cut off
        """

        input = input.contiguous().view(-1, input.size(-1))
        input = self.dropout_module(input)

        new_target, target_idxs = self.adapt_target(target)
        output = [self.head(input)]

        for i in range(len(target_idxs)):
            if target_idxs[i] is not None:
                output.append(self.tail[i](input.index_select(0, target_idxs[i])))
            else:
                output.append(None)

        return output, new_target

    def forward_with_ngram(self, input, target, ngram_word, ngram_dist, ngram_ctxlen_mask, ngram_alpha=0.1):
        """
        Args:
            input: (b x t x d)
            target: (b x t)
        Returns:
            2 lists: output for each cutoff section and new targets by cut off
        """
        dtype = input.dtype
        input = input.contiguous().view(-1, input.size(-1))
        input = self.dropout_module(input)

        new_target, target_idxs, new_ngram_word_list, new_ngram_dist_list, ngram_ctxlen_mask_list = self.adapt_target_with_ngram(target, ngram_word, ngram_dist, ngram_ctxlen_mask)

        logits_list = []

        def process_ngram_logits(new_ngram_word, new_ngram_dist, partial_logits, ngram_ctxlen_mask):
            ngram_logits = torch.scatter_add(torch.zeros_like(partial_logits),
                                             dim=-1, index=new_ngram_word,
                                             src=new_ngram_dist.to(dtype))
            ngram_logits += ngram_logits.eq(0).to(dtype) * 1e-7
            ngram_logits = torch.log(ngram_logits)
            ngram_logits = ngram_logits - ngram_logits.mean(dim=-1, keepdim=True)
            ngram_logits[ngram_ctxlen_mask] = 0
            return ngram_logits

        for i in range(len(new_ngram_word_list)):
            new_ngram_word = new_ngram_word_list[i]
            new_ngram_dist = new_ngram_dist_list[i]
            new_ngram_ctxlen_mask = ngram_ctxlen_mask_list[i]
            if i == 0:
                partial_logits = self.head(input)
                partial_logits = partial_logits + ngram_alpha * process_ngram_logits(new_ngram_word, new_ngram_dist, partial_logits, new_ngram_ctxlen_mask)
            else:
                if target_idxs[i-1] is not None:
                    partial_logits = self.tail[i-1](input.index_select(0, target_idxs[i-1]))
                    partial_logits = partial_logits + ngram_alpha * process_ngram_logits(new_ngram_word, new_ngram_dist, partial_logits, new_ngram_ctxlen_mask)
                else:
                    partial_logits = None
            logits_list.append(partial_logits)

        return logits_list, new_target

    def get_log_prob_with_ngram(self, input, target, ngram_word, ngram_dist, ngram_ctxlen_mask, ngram_alpha=0.1):
        """
        Computes the log probabilities for all the words of the vocabulary,
        given a 2D tensor of hidden vectors.
        """
        dtype = input.dtype
        def process_ngram_logits(new_ngram_word, new_ngram_dist, partial_logits, ngram_ctxlen_mask):
            ngram_logits = torch.scatter_add(torch.zeros_like(partial_logits),
                                             dim=-1, index=new_ngram_word,
                                             src=new_ngram_dist.to(dtype))
            ngram_logits += ngram_logits.eq(0).to(dtype) * 1e-7
            ngram_logits = torch.log(ngram_logits)
            ngram_logits = ngram_logits - ngram_logits.mean(dim=-1, keepdim=True)
            ngram_logits[ngram_ctxlen_mask] = 0
            return ngram_logits
            
        bsz, length, dim = input.size()
        input = input.contiguous().view(-1, dim)

        if target is not None:
            new_target, target_idxs, new_ngram_word_list, new_ngram_dist_list, ngram_ctxlen_mask_list = self.adapt_target_with_ngram(target, ngram_word, ngram_dist, ngram_ctxlen_mask)
        else:
            target_idxs = None

        head_y = self.head(input)
        log_probs = head_y.new_zeros(input.size(0), self.vocab_size)

        head_sz = self.cutoff[0] + len(self.tail)
        log_probs[:, :head_sz] = self.lsm(head_y + ngram_alpha * process_ngram_logits(new_ngram_word_list[0], new_ngram_dist_list[0], head_y, ngram_ctxlen_mask_list[0]))
        tail_priors = log_probs[:, self.cutoff[0] : head_sz].clone()

        for i in range(len(self.tail)):
            new_ngram_word = new_ngram_word_list[i+1]
            new_ngram_dist = new_ngram_dist_list[i+1]
            new_ngram_ctxlen_mask = ngram_ctxlen_mask_list[i+1]
            start = self.cutoff[i]
            end = self.cutoff[i + 1]

            if target_idxs is None:
                # TODO: process ngram without target
                tail_out = log_probs[:, start:end]
                partial_logits = self.tail[i](self.tail[i](input))
                tail_out.copy_(self.tail[i](input))
                log_probs[:, start:end] = self.lsm(tail_out).add_(
                    tail_priors[:, i, None]
                )
            elif target_idxs[i] is not None:
                idxs = target_idxs[i]
                tail_out = log_probs[idxs, start:end]
                partial_logits = self.tail[i](input[idxs])
                partial_logits = partial_logits +  ngram_alpha * process_ngram_logits(new_ngram_word, new_ngram_dist, partial_logits, new_ngram_ctxlen_mask)
                tail_out.copy_(partial_logits)
                log_probs[idxs, start:end] = self.lsm(tail_out).add_(
                    tail_priors[idxs, i, None]
                )

        log_probs = log_probs.view(bsz, length, -1)
        return log_probs

    def get_log_prob(self, input, target):
        """
        Computes the log probabilities for all the words of the vocabulary,
        given a 2D tensor of hidden vectors.
        """

        bsz, length, dim = input.size()
        input = input.contiguous().view(-1, dim)

        if target is not None:
            _, target_idxs = self.adapt_target(target)
        else:
            target_idxs = None

        head_y = self.head(input)
        log_probs = head_y.new_zeros(input.size(0), self.vocab_size)

        head_sz = self.cutoff[0] + len(self.tail)
        log_probs[:, :head_sz] = self.lsm(head_y)
        tail_priors = log_probs[:, self.cutoff[0] : head_sz].clone()

        for i in range(len(self.tail)):
            start = self.cutoff[i]
            end = self.cutoff[i + 1]

            if target_idxs is None:
                tail_out = log_probs[:, start:end]
                tail_out.copy_(self.tail[i](input))
                log_probs[:, start:end] = self.lsm(tail_out).add_(
                    tail_priors[:, i, None]
                )
            elif target_idxs[i] is not None:
                idxs = target_idxs[i]
                tail_out = log_probs[idxs, start:end]
                tail_out.copy_(self.tail[i](input[idxs]))
                log_probs[idxs, start:end] = self.lsm(tail_out).add_(
                    tail_priors[idxs, i, None]
                )

        log_probs = log_probs.view(bsz, length, -1)
        return log_probs
