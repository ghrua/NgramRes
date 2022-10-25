#!/usr/bin/env bash


########################################
# Adaptive Input Repr
########################################

DEST=/mnt/task_wrapper/user_output/artifacts/


########
# ADP
########
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
# CUDA_VISIBLE_DEVICES=0 python train.py \
# mkdir -p $DEST/results/adaptive.input.wt103.5gram.fp16
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python train.py \
#     --task language_modeling_with_ngram \
#     $DEST/data-bin/wikitext-103 \
#     --save-dir $DEST/results/adaptive.input.wt103.5gram.fp16 \
#     --arch transformer_lm_wiki103 \
#     --max-update 286000 --lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
#     --warmup-updates 16000 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --optimizer nag --min-lr 0.0001 --clip-norm 0.1 \
#     --criterion adaptive_loss_with_ngram --max-tokens 8192 --update-freq 1 --tokens-per-sample 2048 --seed 1 \
#     --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp \
#     --fp16 \
#     --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_fairseq/fgm.wt103.train.fairseq.arpa.cache" \
#     --ngram-alpha 0.1 \
#     --ngram-module-path $DEST/KNQuery/ \
#     --ngram-warmup-updates 0  > $DEST/results/adaptive.input.wt103.5gram.fp16/log.train.continue &

SAVE_DIR=$DEST/results/adaptive.input.wt103.fp16
mkdir -p $SAVE_DIR
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python train.py \
    --task language_modeling \
    $DEST/data-bin/wikitext-103 \
    --save-dir $SAVE_DIR \
    --arch transformer_lm_wiki103 \
    --max-update 286000 --lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --optimizer nag --min-lr 1e-5 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 8192 --update-freq 1 --tokens-per-sample 2048 --seed 1 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp \
    --restore-file $DEST/results/adaptive.input.wt103/checkpoint3.pt \
    --fp16 \
    --reset-optimizer \
    --reset-lr-scheduler > $SAVE_DIR/log.train &


########
# XFM_LM
########
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
# SAVE_DIR=$DEST/results/transformer_lm.wmt19.en.ft.ngramgpt.alpha1.0
# mkdir -p $SAVE_DIR
# 
# # CUDA_VISIBLE_DEVICES=0 python train.py \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python train.py \
#     --task language_modeling_with_ngram \
#     $DEST/data-bin/newscrawl.wmt19.bpe \
#     --save-dir $SAVE_DIR \
#     --sample-break-mode eos \
#     --max-target-positions 512 \
#     --arch transformer_lm_gbw  \
#     --relu-dropout 0.1 \
#     --decoder-embed-dim 1536 \
#     --decoder-output-dim 1024 \
#     --decoder-input-dim 1024 \
#     --decoder-ffn-embed-dim 6144 \
#     --decoder-layers 20 \
#     --base-shuffle 1 \
#     --max-target-positions 512 \
#     --criterion cross_entropy \
#     --dropout 0.1 \
#     --optimizer nag --momentum 0.99 --weight-decay 0.00 --clip-norm 0.0 \
#     --lr 5e-05 --lr-scheduler cosine --warmup-updates 16000 --warmup-init-lr 1e-07 \
#     --min-lr 0.0 --t-mult 2.0 --lr-period-updates 959000.0 --lr-shrink 0.6 \
#     --tokens-per-sample 512 \
#     --max-tokens 2048 --update-freq 16 \
#     --restore-file $DEST/wmt19.en/model.pt \
#     --reset-optimizer \
#     --reset-lr-scheduler \
#     --reset-dataloader \
#     --fp16 \
#     --max-update 1500000 \
#     --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_newscrawl_wmt19/newscrawl.wmt19.min2.repunk.arpa.cache" \
#     --ngram-alpha 1.0 \
#     --ngram-module-path $DEST/KNQuery/ \
#     --ngram-warmup-updates 0 \
#     --input-with-ctxlen \
#     --ngram-ctxlen-p "0.1;0.2;0.3;0.4" > $SAVE_DIR/log.train &
