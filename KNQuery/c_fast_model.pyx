#cython: language_level=3
# distutils: language = c++

from fast_model cimport FastLanguageModel, FastGenerationModel, WordList, WordInfo, WordInfoPro
from fast_model cimport one, two, three, four, five, six
from libcpp.vector cimport vector
import torch
import logging
from os.path import basename, isfile

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


cdef class CWordInfo:
    cdef int w
    cdef float p

    def __cinit__ (self, w, p):
        self.w = w
        self.p = p


cdef class CFastLanguageModel:
    cdef FastLanguageModel[one] _flm1
    cdef FastLanguageModel[two] _flm2
    cdef FastLanguageModel[three] _flm3
    cdef FastLanguageModel[four] _flm4
    cdef FastLanguageModel[five] _flm5
    cdef FastLanguageModel[six] _flm6
    cdef int n # ngram length
    
    def __cinit__(self, nragms, probs, backoffs, unk_id, sos_id, n):
        self.n = n
        if self.n == 6:
            self._flm6 = FastLanguageModel[six](nragms, probs, backoffs, unk_id, sos_id, n)
        elif self.n == 5:
            self._flm5 = FastLanguageModel[five](nragms, probs, backoffs, unk_id, sos_id, n)
        elif self.n == 4:
            self._flm4 = FastLanguageModel[four](nragms, probs, backoffs, unk_id, sos_id, n)
        elif self.n == 3:
            self._flm3 = FastLanguageModel[three](nragms, probs, backoffs, unk_id, sos_id, n)
        elif self.n == 2:
            self._flm2 = FastLanguageModel[two](nragms, probs, backoffs, unk_id, sos_id, n)
        elif self.n == 1:
            self._flm1 = FastLanguageModel[one](nragms, probs, backoffs, unk_id, sos_id, n)
        else:
            raise NotImplemented

    @classmethod
    def from_pretrained(cls, model_file, sos_id, unk_id, cache_path=""):
        """
        The model_file is exported by `arpa2binary`
        """
        try:
            logger.info("Loading from cache...")
            ngrams, probs, backoffs, n = torch.load(cache_path)
        except Exception:
            logger.info("No accessible cache found...")
            logger.info("Loading from pretrained model...")
            model_state = torch.load(model_file)
            n = len(model_state)
            ngrams = []
            probs = []
            backoffs = []
            
            for k, ngram_list in enumerate(model_state):
                if k == n-1:
                    for p, wids in ngram_list:
                        ngrams.append(wids)
                        probs.append(p)
                else:
                    for p, wids, b in ngram_list:
                        ngrams.append(wids)
                        probs.append(p)
                        backoffs.append(b)
            del model_state
            if cache_path:
                logger.info("Saving cache to {}...".format(basename(cache_path)))
                torch.save([ngrams, probs, backoffs, n], cache_path)

        return cls(ngrams, probs, backoffs, unk_id, sos_id, n)

    def ngram_logprob(self, wids):
        """
        log10
        """
        if self.n == 6:
            return self._flm6.ngramLogprob(wids)
        elif self.n == 5:
            return self._flm5.ngramLogprob(wids)
        elif self.n == 4:
            return self._flm4.ngramLogprob(wids)
        elif self.n == 3:
            return self._flm3.ngramLogprob(wids)
        elif self.n == 2:
            return self._flm2.ngramLogprob(wids)
        elif self.n == 1:
            return self._flm1.ngramLogprob(wids)
        else:
            raise NotImplemented

    def sent_logprob(self, wids, bos=True):
        """
        log10
        """
        if self.n == 6:
            return self._flm6.sentLogprob(wids, bos)
        elif self.n == 5:
            return self._flm5.sentLogprob(wids, bos)
        elif self.n == 4:
            return self._flm4.sentLogprob(wids, bos)
        elif self.n == 3:
            return self._flm3.sentLogprob(wids, bos)
        elif self.n == 2:
            return self._flm2.sentLogprob(wids, bos)
        elif self.n == 1:
            return self._flm1.sentLogprob(wids, bos)
        else:
            raise NotImplemented
        
    def sent_ppl(self, sent):
        probs = self.sent_logprob(sent)
        return 10 ** (-sum(probs) / len(probs))

    def corpus_ppl(self, corpus):
        probs = []
        for sent in corpus:
            probs += self.sent_logprob(sent)
        return 10 ** (-sum(probs) / len(probs))


cdef class CFastGenerationModel:
    cdef FastGenerationModel[one] _fgm1
    cdef FastGenerationModel[two] _fgm2
    cdef FastGenerationModel[three] _fgm3
    cdef FastGenerationModel[four] _fgm4
    cdef FastGenerationModel[five] _fgm5
    cdef int n # context length
    
    def __cinit__(self, contexts, cands, probs, backoffs, unk_id, sos_id, context_len):
        self.n = context_len
        if self.n == 5:
            self._fgm5 = FastGenerationModel[five](contexts, cands, probs, backoffs, unk_id, sos_id, context_len)
        elif self.n == 4:
            self._fgm4 = FastGenerationModel[four](contexts, cands, probs, backoffs, unk_id, sos_id, context_len)
        elif self.n == 3:
            self._fgm3 = FastGenerationModel[three](contexts, cands, probs, backoffs, unk_id, sos_id, context_len)
        elif self.n == 2:
            self._fgm2 = FastGenerationModel[two](contexts, cands, probs, backoffs, unk_id, sos_id, context_len)
        elif self.n == 1:
            self._fgm1 = FastGenerationModel[one](contexts, cands, probs, backoffs, unk_id, sos_id, context_len)
        else:
            raise NotImplemented

    @classmethod
    def from_scratch(cls, model_file, sos_id, unk_id, cache_path="", shard_size=1000000, prune_ctxset=None):
        """
        The model_file is exported by `arpa2binary`
        """
        contexts, cands, probs, backoffs, context_len = CFastGenerationModel.load_model_state(model_file, prune_ctxset)
        print("Loaded {} contexts from model file".format(len(contexts)))
        if cache_path:
            CFastGenerationModel.save_cache(contexts, cands, probs, backoffs, context_len, cache_path, shard_size)

        return cls(contexts, cands, probs, backoffs, unk_id, sos_id, context_len)

    @classmethod
    def from_cache(cls, cache_path, sos_id, unk_id):
        """
        The model_file is exported by `arpa2binary`
        """
        from glob import glob
        model = None
        total_size = 0
        if isfile(cache_path):
            contexts, cands, probs, backoffs, context_len = torch.load(cache_path)
            model = cls(contexts, cands, probs, backoffs, unk_id, sos_id, context_len)
            print("Loaded {} contexts from the cache file".format(len(contexts)))
        else:
            cache_files = sorted(glob(cache_path + "*"))
            for idx, cache_file in enumerate(cache_files):
                contexts, cands, probs, backoffs, context_len = torch.load(cache_file)
                total_size += len(contexts)
                if model is None:
                    model = cls(contexts, cands, probs, backoffs, unk_id, sos_id, context_len)
                else:
                    model.update_model(contexts, cands, probs, backoffs)
            print("Loaded {} contexts from {} cache files".format(total_size, len(cache_files)))
        return model

    def update_model(self, contexts, cands, probs, backoffs):
        if self.n == 5:
            self._fgm5.updateModel(contexts, cands, probs, backoffs)
        elif self.n == 4:
            self._fgm4.updateModel(contexts, cands, probs, backoffs)
        elif self.n == 3:
            self._fgm3.updateModel(contexts, cands, probs, backoffs)
        elif self.n == 2:
            self._fgm2.updateModel(contexts, cands, probs, backoffs)
        elif self.n == 1:
            self._fgm1.updateModel(contexts, cands, probs, backoffs)
        else:
            raise NotImplemented        

    @staticmethod
    def load_model_state(model_file, prune_ctxset=None):
        model_state = torch.load(model_file)
        n = len(model_state)
        context_len = n-1
        contexts, cands, probs, backoffs = [], [], [], []
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
        for context, info in dist_dict.items():
            if prune_ctxset is not None and context in prune_ctxset:
                continue
            contexts.append(context)
            info = sorted(info, key=lambda x: -x[1])
            cands.append([it[0] for it in info])
            probs.append([it[1] for it in info])
            if context in backoff_dict:
                backoffs.append(backoff_dict[context])
            else:
                logger.info("No backoff value for `{}`".format(context))
                backoffs.append(0.0)  
        return contexts, cands, probs, backoffs, context_len
    
    @staticmethod
    def save_cache(contexts, cands, probs, backoffs, context_len,  cache_path, shard_size=1000000):
        if shard_size < 0:
            torch.save([contexts, cands, probs, backoffs, context_len], "{}.pt".format(cache_path))
            return
        num = len(contexts)
        if shard_size >= num:
            torch.save([contexts, cands, probs, backoffs, context_len], "{}.pt".format(cache_path))
            return
        s, e = 0, shard_size
        shard_idx = 0
        while s < num:
            torch.save([contexts[s:e], cands[s:e], probs[s:e], backoffs[s:e], context_len], "{}.{}.pt".format(cache_path, shard_idx))
            s += shard_size
            e += shard_size
            shard_idx += 1

    def ngram_dist(self, wids, dist_size=100, min_dist_prob=1.0, require_match_len=False):
        """
        log10
        """
        cdef vector[WordInfo] ret
        cdef vector[WordInfoPro] ret_pro
        word, prob, ctxlen = [], [], []

        if require_match_len:
            if self.n == 5:
                ret_pro = self._fgm5.ngramDistWithMatch(wids, dist_size, min_dist_prob)
            elif self.n == 4:
                ret_pro = self._fgm4.ngramDistWithMatch(wids, dist_size, min_dist_prob)
            elif self.n == 3:
                ret_pro = self._fgm3.ngramDistWithMatch(wids, dist_size, min_dist_prob)
            elif self.n == 2:
                ret_pro = self._fgm2.ngramDistWithMatch(wids, dist_size, min_dist_prob)
            elif self.n == 1:
                ret_pro = self._fgm1.ngramDistWithMatch(wids, dist_size, min_dist_prob)
            else:
                ret_pro = NotImplemented
            for jt in ret:
                word.append(jt.w)
                prob.append(jt.p)
                ctxlen.append(jt.ctxlen)
            return word, prob, ctxlen
        else:
            if self.n == 5:
                ret = self._fgm5.ngramDist(wids, dist_size)
            elif self.n == 4:
                ret = self._fgm4.ngramDist(wids, dist_size)
            elif self.n == 3:
                ret = self._fgm3.ngramDist(wids, dist_size)
            elif self.n == 2:
                ret = self._fgm2.ngramDist(wids, dist_size)
            elif self.n == 1:
                ret = self._fgm1.ngramDist(wids, dist_size)
            else:
                ret = NotImplemented
            for it in ret:
                word.append(it.w)
                prob.append(it.p)
        return word, prob

    def sent_dist(self, wids, dist_size=100, min_dist_prob=1.0, bos=True, require_match_len=False, ctxlens=None):
        """
        log10
        """
        word2d, prob2d, ctxlen2d = [], [], []
        cdef vector[vector[WordInfoPro]] sent_ret_pro
        cdef vector[vector[WordInfo]] sent_ret
        if require_match_len:
            if ctxlens is None:
                ctxlens = [self.n] * len(wids)
            if self.n == 5:
                sent_ret_pro = self._fgm5.sentDistWithMatch(wids, ctxlens, dist_size, min_dist_prob, bos)
            elif self.n == 4:
                sent_ret_pro = self._fgm4.sentDistWithMatch(wids, ctxlens, dist_size, min_dist_prob, bos)
            elif self.n == 3:
                sent_ret_pro = self._fgm3.sentDistWithMatch(wids, ctxlens, dist_size, min_dist_prob, bos)
            elif self.n == 2:
                sent_ret_pro = self._fgm2.sentDistWithMatch(wids, ctxlens, dist_size, min_dist_prob, bos)
            elif self.n == 1:
                sent_ret_pro = self._fgm1.sentDistWithMatch(wids, ctxlens, dist_size, min_dist_prob, bos)
            else:
                raise NotImplemented
            for ret_pro in sent_ret_pro:
                word, prob, ctxlen = [], [], []
                for jt in ret_pro:
                    word.append(jt.w)
                    prob.append(jt.p)
                    ctxlen.append(jt.ctxlen)
                word2d.append(word)
                prob2d.append(prob)
                ctxlen2d.append(ctxlen)
            return word2d, prob2d, ctxlen2d
        else:
            if self.n == 5:
                sent_ret = self._fgm5.sentDist(wids, dist_size, bos)
            elif self.n == 4:
                sent_ret = self._fgm4.sentDist(wids, dist_size, bos)
            elif self.n == 3:
                sent_ret = self._fgm3.sentDist(wids, dist_size, bos)
            elif self.n == 2:
                sent_ret = self._fgm2.sentDist(wids, dist_size, bos)
            elif self.n == 1:
                sent_ret = self._fgm1.sentDist(wids, dist_size, bos)
            else:
                raise NotImplemented
            for ret in sent_ret:
                word, prob = [], []
                for it in ret:
                    word.append(it.w)
                    prob.append(it.p)
                word2d.append(word)
                prob2d.append(prob)
            return word2d, prob2d


    def batch_dist(self, batch, dist_size=100, min_dist_prob=1.0, bos=True, require_match_len=False, ctxlens=None):
        word3d, prob3d, ctxlen3d = [], [], []
        if require_match_len:
            if ctxlens is None:
                ctxlens = [[self.n] * len(sent) for sent in batch]
            for sent, ctl in zip(batch, ctxlens):
                word2d, prob2d, ctxlen2d = self.sent_dist(sent, dist_size, min_dist_prob, bos, require_match_len, ctxlens=ctl)
                word3d.append(word2d)
                prob3d.append(prob2d)
                ctxlen3d.append(ctxlen2d)
            return word3d, prob3d, ctxlen3d
        else:
            for sent in batch:
                word2d, prob2d = self.sent_dist(sent, dist_size, min_dist_prob, bos)
                word3d.append(word2d)
                prob3d.append(prob2d)
            return word3d, prob3d

    @property
    def context_len(self, ):
        return self.n


class CMultiFastGenerationModel:

    def __init__(self, fgm_vec):
        self._fgm_vec = fgm_vec
        self._size = len(self._fgm_vec)

    @property
    def size(self):
        return self._size

    @property
    def context_len(self):
        return self._fgm_vec[0].context_len   

    @classmethod
    def from_scratch(cls, model_files, sos_id, unk_id):
        """
        The model_file is exported by `arpa2binary`
        """
        path_list = model_files.split(";")
        fgm_vec = []
        for path in path_list:
            fgm_vec.append(CFastGenerationModel.from_scratch(path, sos_id, unk_id))

        return cls(fgm_vec)

    @classmethod
    def from_cache(cls, cache_path, sos_id, unk_id):
        """
        The model_file is exported by `arpa2binary`
        """
        path_list = cache_path.split(";")
        fgm_vec = []
        for path in path_list:
            fgm_vec.append(CFastGenerationModel.from_cache(path, sos_id, unk_id))

        return cls(fgm_vec)

    def ngram_dist(self, wids, dist_size=100, min_dist_prob=1.0 , model_id=None, require_match_len=False):
        """
        log10
        """
        if model_id is None:
            model_id = self.size - 1
        word, prob = self._fgm_vec[model_id].ngram_dist(wids, dist_size, min_dist_prob, require_match_len)
        return word, prob

    def sent_dist(self, wids, dist_size=100, min_dist_prob=1.0, bos=True, model_id=None, require_match_len=False, ctxlens=None):
        """
        log10
        """
        if model_id is None:
            model_id = self.size - 1
        return self._fgm_vec[model_id].sent_dist(wids, dist_size, min_dist_prob, bos, require_match_len, ctxlens)

    def batch_dist(self, batch, dist_size=100, min_dist_prob=1.0, bos=True, model_ids=None, require_match_len=False, ctxlens=None):
        word3d, prob3d, ctxlen3d = [], [], []
        if model_ids is None:
            model_id = self.size - 1
            model_ids = [model_id] * len(batch)
        if require_match_len:
            if ctxlens is None:
                ctxlens = [[self.context_len] * len(sent) for sent in batch]
            for sent, model_id, ctl in zip(batch, model_ids, ctxlens):
                word2d, prob2d, ctxlen2d = self.sent_dist(sent, dist_size, min_dist_prob, bos, model_id, require_match_len, ctxlens=ctl)
                word3d.append(word2d)
                prob3d.append(prob2d)
                ctxlen3d.append(ctxlen2d)
            return word3d, prob3d, ctxlen3d
        else:
            for sent, model_id in zip(batch, model_ids):
                word2d, prob2d = self.sent_dist(sent, dist_size, min_dist_prob, bos, model_id)
                word3d.append(word2d)
                prob3d.append(prob2d)
            return word3d, prob3d
