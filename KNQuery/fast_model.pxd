from libcpp.vector cimport vector
from libcpp cimport bool
from os.path import basename

cdef extern from "fast_model.cpp":
    pass

cdef extern from "<array>" namespace "std" nogil :
    cdef cppclass one "1":
        pass
    cdef cppclass two "2":
        pass
    cdef cppclass three "3":
        pass
    cdef cppclass four "4":
        pass
    cdef cppclass five "5":
        pass
    cdef cppclass six "6":
        pass

cdef extern from "fast_model.h" namespace "model":
    ctypedef vector[int] WordList


cdef extern from "fast_model.h" namespace "model":
    cdef struct WordInfo:
        int w
        float p

cdef extern from "fast_model.h" namespace "model":
    cdef struct WordInfoPro:
        int w
        float p
        int ctxlen


# Declare the class with cdef
cdef extern from "fast_model.h" namespace "model":
    cdef cppclass FastLanguageModel[N]:
        FastLanguageModel() except +
        FastLanguageModel(const vector[WordList] &ngrams, const vector[float] &probs, const vector[float] &backoffs, int unk_id, int sos_id, int n) except +
        float histBackoff(const WordList &context, int pos)
        float ngramLogprob(const WordList &ngram)
        vector[float] sentLogprob(const WordList &sent, bool bos)


cdef extern from "fast_model.h" namespace "model":
    cdef cppclass FastGenerationModel[N]:
        int n
        FastGenerationModel() except +
        FastGenerationModel(int unk_id, int sos_id, int n) except +
        FastGenerationModel(const vector[WordList] &contexts, const vector[vector[int]] &cands, const vector[vector[float]] &probs, const vector[float] &backoffs, int unk_id, int sos_id, int n) except +
        float histBackoff(const WordList &context, int pos)
        vector[WordInfo] ngramDist(const WordList &ngram, int dist_size)
        vector[vector[WordInfo]] sentDist(const WordList &sent, int dist_size, bool bos)
        vector[WordInfoPro] ngramDistWithMatch(const WordList &ngram, int dist_size, float min_dist_prob)
        vector[vector[WordInfoPro]] sentDistWithMatch(const WordList &sent, int dist_size, float min_dist_prob, bool bos)
        vector[vector[WordInfoPro]] sentDistWithMatch(const WordList &sent, const vector[int] ctxlens, int dist_size, float min_dist_prob, bool bos)
        void updateModel(const vector[WordList] &contexts, const vector[vector[int]] &cands, const vector[vector[float]] &probs, const vector[float] &backoffs)
        
