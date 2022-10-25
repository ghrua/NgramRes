#include <iostream>
#include "fast_model.h"
#include <cassert> 


namespace model {


//     float FastLanguageModel<N>::histBackoff(const WordList &context, int pos) {
//         float b = 0;
//         int context_len = context.size();
//         std::array<int, N> key = prepare_array(context.begin(), context.end());
//         int key_len = context_len;
//         for (int i=0; i < pos; i++) {
//             auto search = backoff_map.find(key);
//             if (search != backoff_map.end()) {
//                 b = b + search -> second;
//             }
//             key_len--;
//             if (key_len == 0) break;
//             key = shorten_array(context, context_len-key_len, context_len);
//         }
//         return b;
//     }

//     float FastLanguageModel<N>::ngramLogprob(const WordList &ngram) {
//         // log10
//         int i = 0, ngram_len = ngram.size();
//         WordList context(ngram.begin(), ngram.end()-1);
//         std::array<int, N> key = prepare_array(ngram);
//         while (i < ngram_len) {
//             auto search = prob_map.find(key);
//             if (search != prob_map.end()) {
//                 return search -> second + histBackoff(context, i);
//             } else {
//                 i++;
//                 key = shorten_array(key, i, ngram_len);
//             }
//         }
//         key = prepare_array({unk_id});
//         return prob_map.find(key) -> second + histBackoff(context, length-1);
//     }

//     std::vector<float> FastLanguageModel<N>::sentLogprob(const WordList &sent, bool bos) {
//         WordList input(sent.begin(), sent.end());
//         if (bos) {
//             auto it = input.begin();
//             input.insert(it, sos_id);
//         }
//         int length = input.size()+1;
//         auto it = input.begin();

//         std::vector<float> ans;
//         for (int i=2; i < length; i++) {
//             int s = i-n, e = i;
//             if (s < 0) s = 0;
//             ans.emplace_back(ngramLogprob(WordList(it+s, it+e)));
//         }
//         return ans;
// }
} // namespace model


