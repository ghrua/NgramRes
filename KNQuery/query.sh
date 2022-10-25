DEST=/mnt/task_wrapper/user_output/artifacts/


# python query.py --tokenizer_path $DEST/huggingface/tokenizers/wikitext-103/space/space_tokenizer.json \
#                       --lm_path ./wt103.kenlm.pt \
#                       --input_path ./tmp.txt \
#                       --mode fast

# python query.py --tokenizer_path $DEST/huggingface/tokenizers/wikitext-103/space/space_tokenizer.json \
#                       --lm_path ./wt103.kenlm.5gram.pt \
#                       --input_path ./train.witheos.txt \
#                       --mode normal

# mkdir -p ./cache_5gram_multi_domain_fairseq/

# python query.py --tokenizer_path $DEST/wmt19.en/dict.txt \
#                       --input_path ./tmp.txt \
#                       --lm_path ./law.train.wt103vocab.kenlm.5gram.arpa.fairseq.pt \
#                       --cache_path ./cache_5gram_multi_domain_fairseq/law.train.wt103vocab.cache \
#                       --mode fast


python query.py --tokenizer_path $DEST/data-bin/wikitext-103-bpe/dict.txt \
                      --input_path ./tmp.txt \
		              --lm_path ./wiki.train.tokens.bpe.repunk.arpa.pt \
                      --pruneset ./wiki.train.tokens.bpe.repunk.pruneset.pt \
                      --cache_path ./cache_5gram_wt103_bpe/wiki.train.tokens.bpe.repunk.arpa.prunectx.cache \
                      --mode fast

# python query.py --tokenizer_path $DEST/data-bin/wikitext-103/dict.txt \
#                       --input_path ./tmp.txt \
#                       --lm_path ./wt103.train.fairseq.arpa.pt \
#                       --cache_path ./cache_5gram_fairseq/fgm.wt103.train.fairseq.arpa.cache \
#                       --mode fast
