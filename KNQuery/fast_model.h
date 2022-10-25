#ifndef FASTLANGUAGEMODEL_H
#define FASTLANGUAGEMODEL_H

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>


namespace model {

    typedef std::vector<int> WordList;

    template<size_t N>
    struct ArrayHasher {
        int operator()(const std::array<int, N> &vec) const {
            int hash = vec.size();
            for(auto &i : vec) {
                hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };

    template <class T>
    void print_vector(const std::vector<T> &v) {
        for (const auto &it : v) {
            std::cout << it << '\t';
        }
        std::cout << '\n';
    }

    struct ContextInfo {
        unsigned int si;
        unsigned int ei;
        float b;
    };

    struct WordInfo {
        int w;
        float p;
    };

    struct WordInfoPro {
        int w;
        float p;
        int ctxlen;
    };

    // typedef std::unordered_map<WordList, float, WordListHasher> NgramFloatMap;


    template <class T, size_t N>
    void print_array(const std::array<T, N> &v) {
        for (const auto &it : v) {
            std::cout << it << ", ";
        }
        std::cout << '\n';
    }

    template<size_t N>
    class FastLanguageModel {
        private:
            inline std::array<int, N> prepare_array(const std::vector<int> &vec) {
                int vec_size = vec.size();
                assert (vec_size <= n);
                std::array<int, N> arr;
                std::move(vec.begin(), vec.end(), arr.begin());
                if (vec_size < n) {
                    std::fill(arr.begin() + vec_size, arr.end(), -1);
                }
                return arr;
            }

            inline std::array<int, N> shorten_array(const std::array<int, N> &arr, int start_idx, int end_idx) {
                int new_size = end_idx - start_idx;
                std::array<int, N> new_arr;
                std::move(arr.begin()+start_idx, arr.begin()+end_idx, new_arr.begin());
                std::fill(new_arr.begin() + new_size, new_arr.end(), -1);
                return new_arr;
            }

        public:
            int unk_id, sos_id, n;
            // std::unordered_map<std::array<int, N>, float, ArrayHasher<N> >  prob_map;
            // std::unordered_map<std::array<int, N>, float, ArrayHasher<N> > backoff_map;
            std::unordered_map<std::array<int, N>, std::array<float, 2>, ArrayHasher<N> >  ngram_map;
            FastLanguageModel(){}
            FastLanguageModel(const std::vector<WordList> &ngrams, const std::vector<float> &probs, const std::vector<float> &backoffs, int unk_id, int sos_id, int n) : unk_id(unk_id), sos_id(sos_id), n(n) {
                int sn = ngrams.size(), sb = backoffs.size();
                // for (int i=0; i < sn; i++) {
                //     auto key = prepare_array(ngrams[i]);
                //     if (i < sb)
                //         backoff_map[key] = backoffs[i];
                //     prob_map[key] = probs[i];
                // }
                for (int i=0; i < sn; i++) {
                    float b = 0;
                    auto key = prepare_array(ngrams[i]);
                    if (i < sb)
                        b = backoffs[i];
                    ngram_map[key] = {probs[i], b};
                }                
            }
            // ~FastLanguageModel();

            float histBackoff(const WordList &context, int pos) {
                float b = 0;
                int key_len = context.size();
                std::array<int, N> key = prepare_array(context);
                for (int i=0; i < pos; i++) {
                    auto search = ngram_map.find(key);
                    if (search != ngram_map.end()) {
                        b = b + search -> second[1];
                    }
                    if (key_len == 1) break;
                    key = shorten_array(key, 1, key_len);
                    key_len--;
                }
                return b;
            }

            float ngramLogprob(const WordList &ngram) {
                // log10
                int i = 0, ngram_len = ngram.size();
                WordList context(ngram.begin(), ngram.end()-1);
                std::array<int, N> key = prepare_array(ngram);
                while (i < ngram_len) {
                    auto search = ngram_map.find(key);
                    if (search != ngram_map.end()) {
                        return search -> second[0] + histBackoff(context, i);
                        
                    } else {
                        key = shorten_array(key, 1, ngram_len-i);
                        i++;
                    }
                }
                key = prepare_array({unk_id});
                return ngram_map.find(key) -> second[0] + histBackoff(context, ngram_len-1);
            }

            std::vector<float> sentLogprob(const WordList &sent, bool bos=true) {
                WordList input(sent.begin(), sent.end());
                if (bos) {
                    auto it = input.begin();
                    input.insert(it, sos_id);
                }
                int length = input.size()+1;
                auto it = input.begin();

                std::vector<float> ans;
                for (int i=2; i < length; i++) {
                    int s = i-n, e = i;
                    if (s < 0) s = 0;
                    ans.emplace_back(ngramLogprob(WordList(it+s, it+e)));
                }
                return ans;
            }
            // float sentPPL(const WordList &sent);
            // float corpusPPL(const std::vector<WordList> &corpus);
    };

    template<size_t N>
    class FastGenerationModel {
        private:
            inline std::array<int, N> prepare_array(const std::vector<int> &vec) {
                int vec_size = vec.size();
                assert (vec_size <= n);
                std::array<int, N> arr;
                std::move(vec.begin(), vec.end(), arr.begin());
                if (vec_size < n) {
                    std::fill(arr.begin() + vec_size, arr.end(), -1);
                }
                return arr;
            }

            inline std::array<int, N> shorten_array(const std::array<int, N> &arr, int start_idx, int end_idx) {
                int new_size = end_idx - start_idx;
                std::array<int, N> new_arr;
                std::move(arr.begin()+start_idx, arr.begin()+end_idx, new_arr.begin());
                std::fill(new_arr.begin() + new_size, new_arr.end(), -1);
                return new_arr;
            }

            inline std::vector<WordInfoPro> remove_duplicates(const std::vector<WordInfoPro> &vec, int dist_size) {
                std::vector<WordInfoPro> ans;
                std::unordered_set<int> visited;
                int min_ctxlen = 999;
                int vec_size = vec.size(), ans_size = 0;
                for (int i=0; i < vec_size; i++) {
                    if (vec[i].p > 0)   break;
                    if (visited.count(vec[i].w)) continue;
                    visited.emplace(vec[i].w);
                    if (vec[i].ctxlen < min_ctxlen)
                        min_ctxlen = vec[i].ctxlen;
                    ans.emplace_back(std::move(vec[i]));
                    ans_size++;
                    if (ans_size >= dist_size) break;
                }
                
                while (ans_size < dist_size) {
                    WordInfoPro tmp = {0, 1, min_ctxlen};
                    ans.emplace_back(std::move(tmp));
                    ans_size++;
                }

                return ans;
            }

            inline std::vector<WordInfo> remove_duplicates(const std::vector<WordInfo> &vec, int dist_size) {
                std::vector<WordInfo> ans;
                std::unordered_set<int> visited;
                int vec_size = vec.size(), ans_size = 0;
                for (int i=0; i < vec_size; i++) {
                    if (vec[i].p > 0)   break;
                    if (visited.count(vec[i].w)) continue;
                    visited.emplace(vec[i].w);
                    ans.emplace_back(std::move(vec[i]));
                    ans_size++;
                    if (ans_size >= dist_size) break;
                }
                
                while (ans_size < dist_size) {
                    WordInfo tmp = {0, 1};
                    ans.emplace_back(std::move(tmp));
                    ans_size++;
                }

                return ans;
            }
            
        public:
            int unk_id, sos_id, n;
            std::unordered_map<std::array<int, N>, ContextInfo, ArrayHasher<N> >  context_map;
            std::vector<WordInfo> wordinfo_vec;
            FastGenerationModel(){}
            FastGenerationModel(int unk_id, int sos_id, int n) : unk_id(unk_id), sos_id(sos_id), n(n){}
            FastGenerationModel(const std::vector<WordList> &contexts, const std::vector<std::vector<int>> &cands, const std::vector<std::vector<float>> &probs, const std::vector<float> &backoffs, int unk_id, int sos_id, int n) : unk_id(unk_id), sos_id(sos_id), n(n) {
                int sn = contexts.size();
                for (int i=0; i < sn; i++) {
                    unsigned int start_idx = wordinfo_vec.size();
                    auto key = prepare_array(contexts[i]);
                    unsigned int num_cands = cands[i].size();
                    context_map[key] = {start_idx, start_idx+num_cands, backoffs[i]};
                    for (size_t j=0; j < num_cands; j++) {
                        WordInfo tmp = {cands[i][j], probs[i][j]};
                        wordinfo_vec.emplace_back(std::move(tmp));
                    }
                }                
            }
            // ~FastLanguageModel();
            void updateModel(const std::vector<WordList> &contexts, const std::vector<std::vector<int>> &cands, const std::vector<std::vector<float>> &probs, const std::vector<float> &backoffs) {
                int sn = contexts.size();
                for (int i=0; i < sn; i++) {
                    unsigned int start_idx = wordinfo_vec.size();
                    auto key = prepare_array(contexts[i]);
                    unsigned int num_cands = cands[i].size();
                    context_map[key] = {start_idx, start_idx+num_cands, backoffs[i]};
                    for (size_t j=0; j < num_cands; j++) {
                        WordInfo tmp = {cands[i][j], probs[i][j]};
                        wordinfo_vec.emplace_back(std::move(tmp));
                    }
                }                  
            }

            float histBackoff(const WordList &context, int pos) {
                float b = 0;
                int key_len = context.size();
                std::array<int, N> key = prepare_array(context);
                for (int i=0; i < pos; i++) {
                    auto search = context_map.find(key);
                    if (search != context_map.end()) {
                        b = b + search -> second.b;
                    }
                    if (key_len == 1) break;
                    key = shorten_array(key, 1, key_len);
                    key_len--;
                }
                return b;
            }

            std::vector<WordInfoPro> ngramDistWithMatch(const WordList &context, int dist_size=100, float min_dist_prob=1.0) {
                // log10
                int index = 0, max_size = dist_size;
                std::vector<WordInfoPro> ans(max_size);
                std::array<int, N> key = prepare_array(context);
                int i = 0, context_len = context.size();
                float accum_prob = 0.0;
                std::unordered_set<int> visited;
                while (i < context_len) {
                    auto search = context_map.find(key);
                    if (search != context_map.end()) {
                        ContextInfo cinfo = search -> second;
                        float b = histBackoff(context, i);
                        for (size_t j=cinfo.si; j < cinfo.ei && index < max_size && accum_prob < min_dist_prob; j++) {
                            if (visited.count(wordinfo_vec[j].w)) continue;
                            visited.emplace(wordinfo_vec[j].w);
                            float cur_prob = wordinfo_vec[j].p + b;
                            accum_prob = accum_prob + std::pow(10, cur_prob);
                            ans[index] = {wordinfo_vec[j].w, cur_prob, context_len-i};
                            index++;
                        }
                        
                    }
                    if (index >= max_size) {
                        return ans;
                    } else if (accum_prob >= min_dist_prob) {
                        while (index < max_size) {
                            WordInfoPro tmp = {0, 1, 0};
                            ans[index] = std::move(tmp);
                            index++;
                        }
                        return ans;
                    }
                    key = shorten_array(key, 1, context_len-i);
                    i++;
                }
                float b = histBackoff(context, context_len);
                auto search = context_map.find(key);
                ContextInfo cinfo = search -> second;
                for (size_t j=cinfo.si; j < cinfo.ei && index < max_size && accum_prob < min_dist_prob; j++) {
                    if (visited.count(wordinfo_vec[j].w)) continue;
                    visited.emplace(wordinfo_vec[j].w);
                    float cur_prob = wordinfo_vec[j].p + b;
                    accum_prob = accum_prob + std::pow(10, cur_prob);
                    ans[index] = {wordinfo_vec[j].w, cur_prob, 0};
                    index++;
                }
                if (accum_prob >= min_dist_prob) {
                    while (index < max_size) {
                        WordInfoPro tmp = {0, 1, 0};
                        ans[index] = std::move(tmp);
                        index++;
                    }
                }
                return ans;
            }

            std::vector<std::vector<WordInfoPro>> sentDistWithMatch(const WordList &sent, int dist_size=100, float min_dist_prob=1.0, bool bos=true) {
                // std::cout << context_map.size() << ' ' << wordinfo_vec.size() << '\n';
                WordList input(sent.begin(), sent.end());
                if (bos) {
                    auto it = input.begin();
                    input.insert(it, sos_id);
                }
                int length = input.size()+1;
                auto it = input.begin();

                std::vector<std::vector<WordInfoPro>> ans;
                for (int i=1; i < length; i++) {
                    int s = i-n, e = i;
                    if (s < 0) s = 0;
                    auto ret = ngramDistWithMatch(WordList(it+s, it+e), dist_size, min_dist_prob);
                    ans.emplace_back(std::move(ret));
                    // ans.emplace_back(remove_duplicates(std::move(ret), dist_size));
                }
                return ans;
            }

            std::vector<std::vector<WordInfoPro>> sentDistWithMatch(const WordList &sent, const std::vector<int> ctxlens, int dist_size=100, float min_dist_prob=1.0, bool bos=true) {
                // std::cout << context_map.size() << ' ' << wordinfo_vec.size() << '\n';
                WordList input(sent.begin(), sent.end());
                std::vector<int> ctxlens_new(ctxlens.begin(), ctxlens.end());
                if (bos) {
                    input.insert(input.begin(), sos_id);
                    ctxlens_new.insert(ctxlens_new.begin(), n);
                }
                int length = input.size()+1;
                auto it = input.begin();

                std::vector<std::vector<WordInfoPro>> ans;
                for (int i=1; i < length; i++) {
                    int s = i-ctxlens_new[i-1], e = i;
                    if (s < 0) s = 0;
                    auto ret = ngramDistWithMatch(WordList(it+s, it+e), dist_size, min_dist_prob);
                    ans.emplace_back(std::move(ret));
                    // ans.emplace_back(remove_duplicates(std::move(ret), dist_size));
                }
                return ans;
            }
            
            std::vector<WordInfo> ngramDist(const WordList &context, int dist_size=100) {
                // log10
                int index = 0, max_size = dist_size;
                std::vector<WordInfo> ans(max_size);
                std::array<int, N> key = prepare_array(context);
                int i = 0, context_len = context.size();
                while (i < context_len) {
                    auto search = context_map.find(key);
                    if (search != context_map.end()) {
                        ContextInfo cinfo = search -> second;
                        float b = histBackoff(context, i);
                        for (size_t j=cinfo.si; j < cinfo.ei && index < max_size; j++) {
                            ans[index] = {wordinfo_vec[j].w, wordinfo_vec[j].p + b};
                            index++;
                        }
                    }
                    if (index >= max_size) return ans;
                    key = shorten_array(key, 1, context_len-i);
                    i++;
                }
                float b = histBackoff(context, context_len);
                auto search = context_map.find(key);
                ContextInfo cinfo = search -> second;
                for (size_t j=cinfo.si; j < cinfo.ei && index < max_size; j++) {
                    ans[index] = {wordinfo_vec[j].w, wordinfo_vec[j].p + b};
                    index++;
                }
                return ans;
            }

            std::vector<std::vector<WordInfo>> sentDist(const WordList &sent, int dist_size=100, bool bos=true) {
                // std::cout << context_map.size() << ' ' << wordinfo_vec.size() << '\n';
                WordList input(sent.begin(), sent.end());
                if (bos) {
                    auto it = input.begin();
                    input.insert(it, sos_id);
                }
                int length = input.size()+1;
                auto it = input.begin();

                std::vector<std::vector<WordInfo>> ans;
                for (int i=1; i < length; i++) {
                    int s = i-n, e = i;
                    if (s < 0) s = 0;
                    auto ret = ngramDist(WordList(it+s, it+e), dist_size);
                    ans.emplace_back(remove_duplicates(std::move(ret), dist_size));
                }
                return ans;
            }
            // float sentPPL(const WordList &sent);
            // float corpusPPL(const std::vector<WordList> &corpus);
    };

}

#endif