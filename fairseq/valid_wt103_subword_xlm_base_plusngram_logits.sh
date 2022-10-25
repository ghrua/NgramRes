DEST=/mnt/task_wrapper/user_output/artifacts/
# SUB=test

# gpuid=0
# MODEL=$DEST/results/xfm.base.drop0.3.warm4k.wt103.bpe.fp16/checkpoint_best.pt
# for SUB in valid test
# do
#     CUDA_VISIBLE_DEVICES=$gpuid nohup python eval_lm.py \
#         $DEST/data-bin/wikitext-103-bpe \
#         --gen-subset $SUB \
#         --task language_modeling \
#         --path $MODEL \
#         --distributed-world-size 1 \
#         --batch-size 1  \
#         --context-window 1536 \
#         --tokens-per-sample 2048 \
#         --post-process "@@ " > $MODEL.$SUB.log &
#     let "gpuid=gpuid+1"
# done

gpuid=0
dist=100
SUB=valid
# alpha=0.2
for TRAIN_ALPHA in 0.4 0.1
do
    # MODEL=$DEST/results/xfm.base.drop0.3.warm4k.wt103.bpe.fp16/checkpoint_best.pt
    # CUDA_VISIBLE_DEVICES=$gpuid nohup python eval_lm.py \
    # $DEST/data-bin/wikitext-103-bpe \
    # --gen-subset $SUB \
    # --task language_modeling_with_ngram \
    # --path $MODEL \
    # --distributed-world-size 1 \
    # --batch-size 1  \
    # --context-window 1536 \
    # --tokens-per-sample 2048 \
    # --output-word-probs \
    # --prob-save-path $DEST/results/xfm.base.drop0.3.warm4k.wt103.bpe.fp16/prob.xfm.ctxwin1536.$SUB.pt \
    # --ngram-dist-size $dist \
    # --eval-ngram-alpha 0 \
    # --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_wt103_bpe/wiki.train.tokens.bpe.repunk.arpa.cache" \
    # --ngram-module-path $DEST/KNQuery/ > $MODEL.$SUB.ctxwin1536.prob.log &
    # let "gpuid=gpuid+1"

    for EVAL_ALPHA in 0.0 0.05 0.1 0.15 0.2 0.25 0.35 0.4
    do
        MODEL=$DEST/results/xfm.base.drop0.3.warm4k.wt103.bpe.fp16.5gram.$TRAIN_ALPHA/checkpoint_best.pt
        CUDA_VISIBLE_DEVICES=$gpuid nohup python eval_lm.py \
        $DEST/data-bin/wikitext-103-bpe \
        --gen-subset $SUB \
        --task language_modeling_with_ngram \
        --path $MODEL \
        --distributed-world-size 1 \
        --batch-size 1  \
        --context-window 1536 \
        --tokens-per-sample 2048 \
        --output-word-probs \
        --prob-save-path $DEST/results/xfm.base.drop0.3.warm4k.wt103.bpe.fp16.5gram.$TRAIN_ALPHA/prob.xfm.$SUB.ctxwin1536.prob.wctxlen.brute.$EVAL_ALPHA.pt \
        --weighted-ctxlen \
        --ngram-dist-size $dist \
        --eval-ngram-alpha $EVAL_ALPHA \
        --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_wt103_bpe/wiki.train.tokens.bpe.repunk.arpa.cache" \
        --ngram-module-path $DEST/KNQuery/ > $MODEL.$SUB.ctxwin1536.prob.wctxlen.brute.$EVAL_ALPHA.log &
        let "gpuid=gpuid+1"
    done
    exit 0
done
# --prob-save-path ./prob.5gram.$SUB.$alpha.pt \
# --output-word-probs \

# dist=5000
# for SUB in valid test
# do
#     MODEL=$DEST/results/xfm.base.drop0.3.warm4k.wt103.bpe.fp16.5gram.0.2/checkpoint_best.pt
#     CUDA_VISIBLE_DEVICES=$gpuid nohup python eval_lm.py \
#     $DEST/data-bin/wikitext-103-bpe \
#     --gen-subset $SUB \
#     --task language_modeling_with_ngram \
#     --path $MODEL \
#     --distributed-world-size 1 \
#     --batch-size 1  \
#     --context-window 1536 \
#     --tokens-per-sample 2048 \
#     --post-process "@@ " \
#     --ngram-dist-size $dist \
#     --eval-ngram-alpha $alpha \
#     --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_wt103_bpe/wiki.train.tokens.bpe.repunk.arpa.cache" \
#     --ngram-module-path $DEST/KNQuery/ > $MODEL.$SUB.$dist.$alpha.log &
#     let "gpuid=gpuid+1"
# done
