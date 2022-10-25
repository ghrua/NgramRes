DEST=/mnt/task_wrapper/user_output/artifacts/

# python context_counter.py --tokenizer_path $DEST/data-bin/wikitext-103-bpe/dict.txt \
#                       --input_path $DEST/data/wikitext-103/wiki.train.tokens.bpe.repunk \
#                       --max_ctxlen 4 \
#                       --output_path ./wiki.train.tokens.bpe.repunk.ctxinfo.went.pt

python context_counter.py --tokenizer_path $DEST/data-bin/multidomain.law.wmt19.bpe/dict.txt \
                      --input_path $DEST/data/newscrawl_wmt19/multi_domain/law/train.en.norm.print.tok.bpe \
                      --max_ctxlen 4 \
                      --output_path ./newscrawl.wmt19.law.train.en.norm.print.tok.bpe.ctxinfo.went.pt
