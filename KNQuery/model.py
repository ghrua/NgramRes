import torch
import numpy as np
from utils import remove_duplicate


class LanguageModel:

    def __init__(self, n, prob_dict, backoff_dict, sos_id, unk_id) -> None:
        self.prob = prob_dict
        self.backoff = backoff_dict
        self.n = n
        self._sos_id = sos_id
        self._unk_id = unk_id

    @classmethod
    def from_pretrained(cls, model_file, sos_id, unk_id):
        model_state = torch.load(model_file)
        n = len(model_state)
        prob_dict = dict()
        backoff_dict = dict()
        
        for k, ngram_list in enumerate(model_state):
            if k == n-1:
                for p, wids in ngram_list:
                    prob_dict[wids] = p
            else:
                for p, wids, b in ngram_list:
                    prob_dict[wids] = p
                    backoff_dict[wids] = b
        del model_state
        return cls(n, prob_dict, backoff_dict, sos_id, unk_id)

    @property
    def unk_id(self,):
        return self._unk_id

    @property
    def sos_id(self,):
        return self._sos_id

    def hist_backoff(self, ctx, pos):
        b = 0
        for j in range(pos):
            key = ctx[j:]
            if key in self.backoff:
                b += self.backoff[key]
        return b

    def ngram_logprob(self, wids):
        """
        log10
        """
        i, length = 0, len(wids)
        while i < length:
            key = wids[i:]
            if key in self.prob:
                b = self.hist_backoff(wids[:-1], i)
                return self.prob[key] + b
            else:
                i += 1
        b = self.hist_backoff(wids[:-1], length-1)
        return self.prob[(self.unk_id,)] + b

    def sent_logprob(self, wids, bos=True):
        """
        log10
        """
        if bos:
            wids = tuple([self.sos_id] + list(wids))
        elif not isinstance(wids, tuple):
            wids = tuple(wids)
        length = len(wids)
        ans = []
        for i in range(2, length+1):
            s, e = max(0, i-self.n), i
            ans.append(self.ngram_logprob(wids[s:e]))
        return ans

    def sent_ppl(self, sent):
        probs = self.sent_logprob(sent)
        return 10 ** (-sum(probs) / len(probs))

    def corpus_ppl(self, corpus):
        probs = []
        for sent in corpus:
            probs += self.sent_logprob(sent)
        return 10 ** (-sum(probs) / len(probs))


class GenerationModel:

    def __init__(self, n, dist_dict, backoff_dict, sos_id, unk_id) -> None:
        self.dist = dist_dict
        self.backoff = backoff_dict
        self.n = n
        self._sos_id = sos_id
        self._unk_id = unk_id

    # @classmethod
    # def from_pretrained(cls, model_file, sos_id, unk_id, group_ngram=False):
    #     model_state = torch.load(model_file)
    #     n = len(model_state)
    #     context_map = dict()
    #     context_num = sum([len(it) for it in model_state[:-1]])+1
    #     word_list = [[] for i in range(context_num)]
    #     dist_list = [[] for i in range(context_num)]
    #     backoff_list = np.zeros(context_num)
    #     ci = 1
    #     for k, ngram_list in enumerate(model_state):
    #         ngram_list = sorted(ngram_list, key=lambda x: -x[0])
    #         if k == n-1:
    #             for p, wids in ngram_list:
    #                 ctx, cand = wids[:-1], wids[-1]
    #                 idx = context_map[ctx]
    #                 word_list[idx].append(cand)
    #                 dist_list[idx].append(p)
    #         elif k == 0:
    #             for p, wids, b in ngram_list:
    #                 assert wids not in context_map, "{} alraedy in".format(wids)
    #                 context_map[wids] = ci
    #                 word_list[0].append(wids[0])
    #                 dist_list[0].append(p)
    #                 backoff_list[ci] = b
    #                 ci += 1
    #         else:
    #             for p, wids, b in ngram_list:
    #                 ctx, cand = wids[:-1], wids[-1]
    #                 idx = context_map[ctx]
    #                 word_list[idx].append(cand)
    #                 dist_list[idx].append(p)
    #                 context_map[wids] = ci
    #                 backoff_list[ci] = b 
    #                 ci += 1
    #     word_list = [np.array(it) for it in word_list]
    #     dist_list = [np.array(it) for it in dist_list]
    #     return cls(n, dist_dict, backoff_dict, sos_id, unk_id, group_ngram)

    @classmethod
    def from_pretrained(cls, model_file, sos_id, unk_id):
        model_state = torch.load(model_file)
        n = len(model_state)
        dist_dict = dict()
        backoff_dict = dict()

        for k, ngram_list in enumerate(model_state):
            # ngram_list = sorted(ngram_list, key=lambda x: -x[0])
            if k == n-1:
                for p, wids in ngram_list:
                    ctx, cand = wids[:-1], wids[-1]
                    if ctx in dist_dict:
                        dist_dict[ctx].append((cand, p))
                    else:
                        dist_dict[ctx] = [(cand, p)]
            else:
                for p, wids, b in ngram_list:
                    ctx, cand = wids[:-1], wids[-1]
                    if ctx in dist_dict:
                        dist_dict[ctx].append((cand, p))
                    else:
                        dist_dict[ctx] = [(cand, p)]
                    backoff_dict[wids] = b 
                    
        return cls(n, dist_dict, backoff_dict, sos_id, unk_id)

    @property
    def unk_id(self,):
        return self._unk_id

    @property
    def sos_id(self,):
        return self._sos_id

    def hist_backoff(self, ctx, pos):
        b = 0
        for j in range(pos):
            key = ctx[j:]
            if key in self.backoff:
                b += self.backoff[key]
        return b

    def ngram_dist(self, ctx, dist_size=100):
        ans = []
        i, n, max_dist = 0, len(ctx), self.n * dist_size
        while i < n:
            key = ctx[i:]
            if key in self.dist:
                b = self.hist_backoff(ctx, i)
                ans += [(wid, b+p) for wid, p in self.dist[key][:dist_size]]
            i += 1
            # if len(ans) >= max_dist:
            #     return remove_duplicate(ans)[:dist_size]
            if len(ans) >= dist_size:
                return ans[:dist_size]
        b = self.hist_backoff(ctx, n)
        ans += [(wid, b+p) for wid, p in self.dist[ctx[n:]][:dist_size]]
        # return remove_duplicate(ans)[:dist_size]
        return ans[:dist_size]

    def sent_dist(self, wids, dist_size=100, bos=True):
        if bos:
            wids = tuple([self.sos_id] + list(wids))
        elif not isinstance(wids, tuple):
            wids = tuple(wids)
        length = len(wids)
        ans = []
        for i in range(1, length):
            s, e = max(0, i+1-self.n), i
            ans.append(self.ngram_dist(wids[s:e], dist_size))
        return ans

    def batch_dist(self, ctx, dist_size=100):
        pass
        