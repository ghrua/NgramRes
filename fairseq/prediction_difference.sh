DEST=/mnt/task_wrapper/user_output/artifacts/

alpha=0.6
dist=100
MODEL=$DEST/results/xfm.base.drop0.3.warm4k.wt103.bpe.fp16.5gram.$alpha/checkpoint_best.pt
SUB=valid
CUDA_VISIBLE_DEVICES=0 nohup python prediction_difference.py \
    $DEST/data-bin/wikitext-103-bpe \
    --gen-subset $SUB \
    --task language_modeling_with_ngram \
    --path $MODEL \
    --distributed-world-size 1 \
    --batch-size 1  \
    --context-window 0 \
    --tokens-per-sample 2048 \
    --ngram-dist-size $dist \
    --eval-ngram-alpha $alpha \
    --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_wt103_bpe/wiki.train.tokens.bpe.repunk.arpa.cache" \
    --ngram-module-path $DEST/KNQuery/ \
    --pd-save-path $MODEL.$SUB.pd.pt > $MODEL.$SUB.pd.log &


SUB=train
CUDA_VISIBLE_DEVICES=1 nohup python prediction_difference.py \
    $DEST/data-bin/wikitext-103-bpe \
    --gen-subset $SUB \
    --task language_modeling_with_ngram \
    --path $MODEL \
    --distributed-world-size 1 \
    --batch-size 1  \
    --context-window 0 \
    --tokens-per-sample 2048 \
    --ngram-dist-size $dist \
    --eval-ngram-alpha $alpha \
    --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_wt103_bpe/wiki.train.tokens.bpe.repunk.arpa.cache" \
    --ngram-module-path $DEST/KNQuery/ \
    --pd-save-path $MODEL.$SUB.pd.pt > $MODEL.$SUB.pd.log &


# MODEL=$DEST/results/xfm.base.drop0.3.warm4k.wt103.bpe.fp16/checkpoint_best.pt
# SUB=valid
# CUDA_VISIBLE_DEVICES=2 nohup python prediction_difference.py \
#     $DEST/data-bin/wikitext-103-bpe \
#     --gen-subset $SUB \
#     --task language_modeling \
#     --path $MODEL \
#     --distributed-world-size 1 \
#     --batch-size 1  \
#     --context-window 0 \
#     --tokens-per-sample 2048 \
#     --pd-save-path $MODEL.$SUB.pd.pt > $MODEL.$SUB.pd.log &


# SUB=train
# CUDA_VISIBLE_DEVICES=3 nohup python prediction_difference.py \
#     $DEST/data-bin/wikitext-103-bpe \
#     --gen-subset $SUB \
#     --task language_modeling \
#     --path $MODEL \
#     --distributed-world-size 1 \
#     --batch-size 1  \
#     --context-window 0 \
#     --tokens-per-sample 2048 \
#     --pd-save-path $MODEL.$SUB.pd.pt > $MODEL.$SUB.pd.log &
