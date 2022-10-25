#!/usr/bin/env bash


########################################
# Adaptive Input Repr
########################################

DEST=/mnt/task_wrapper/user_output/artifacts/


########
# XFM_LM
########
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
SAVE_DIR=$DEST/results/transformer_lm.wmt19.en.ngramgpt.inter.alpha0.2
mkdir -p $SAVE_DIR

# CUDA_VISIBLE_DEVICES=0 python train.py \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python train.py \
    --task language_modeling_with_ngram \
    $DEST/data-bin/newscrawl.wmt19.bpe \
    --save-dir $SAVE_DIR \
    --sample-break-mode eos \
    --max-target-positions 512 \
    --arch transformer_lm_gbw  \
    --relu-dropout 0.1 \
    --decoder-embed-dim 1536 \
    --decoder-output-dim 1024 \
    --decoder-input-dim 1024 \
    --decoder-ffn-embed-dim 6144 \
    --decoder-layers 20 \
    --base-shuffle 1 \
    --max-target-positions 512 \
    --criterion cross_entropy \
    --dropout 0.1 \
    --optimizer nag --momentum 0.99 --weight-decay 0.00 --clip-norm 0.0 \
    --lr 5e-05 --lr-scheduler cosine --warmup-updates 16000 --warmup-init-lr 1e-07 \
    --min-lr 0.0 --t-mult 2.0 --lr-period-updates 959000.0 --lr-shrink 0.6 \
    --restore-file $DEST/wmt19.en/model.pt \
    --tokens-per-sample 512 \
    --max-tokens 2048 --update-freq 16 \
    --fp16 \
    --max-update 1500000 \
    --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_newscrawl_wmt19/newscrawl.wmt19.min2.repunk.arpa.cache" \
    --ngram-alpha 0.2 \
    --ngram-module-path $DEST/KNQuery/ \
    --ngram-warmup-updates 0 > $SAVE_DIR/log.train &


