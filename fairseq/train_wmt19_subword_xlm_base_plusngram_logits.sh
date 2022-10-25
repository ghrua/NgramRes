#!/usr/bin/env bash


########################################
# Adaptive Input Repr
########################################

DEST=/mnt/task_wrapper/user_output/artifacts/

# SAVE_DIR=$DEST/results/xfm.base.wmt19.bpe.debug
# mkdir -p $SAVE_DIR
# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --task language_modeling \
#     $DEST/data-bin/newscrawl.wmt19.bpe.fixbug/ \
#     --save-dir $SAVE_DIR \
#     --arch transformer_lm_big --share-decoder-input-output-embed \
#     --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.00 --clip-norm 0.0 \
#     --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 16000 --warmup-init-lr 1e-07 \
#     --max-tokens 4096 --update-freq 1 --tokens-per-sample 512 --seed 1 \
#     --max-target-positions 512 \
#     --max-update 150000 \
#     --sample-break-mode eos  \
#     --fp16

######
# base+gpt with different alpha
######

# alpha=0.6
# SAVE_DIR=$DEST/results/xfm.base.wmt19.bpe.fp16.5gram.$alpha
# mkdir -p $SAVE_DIR
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
#     --task language_modeling_with_ngram \
#     $DEST/data-bin/newscrawl.wmt19.bpe.fixbug/ \
#     --save-dir $SAVE_DIR \
#     --arch transformer_lm_big --share-decoder-input-output-embed \
#     --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.00 --clip-norm 0.0 \
#     --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 16000 --warmup-init-lr 1e-07 \
#     --max-tokens 4096 --update-freq 1 --tokens-per-sample 512 --seed 1 \
#     --max-target-positions 512 \
#     --max-update 150000 \
#     --sample-break-mode eos  \
#     --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_newscrawl_wmt19/newscrawl.wmt19.min2.repunk.arpa.cache" \
#     --ngram-alpha $alpha \
#     --ngram-module-path $DEST/KNQuery/ \
#     --ngram-warmup-updates 0 \
#     --fp16 > $SAVE_DIR/log.train

######
# base+gpt with 4 clusters
######

alpha=0.6
SAVE_DIR=$DEST/results/xfm.base.drop0.3.wmt19.bpe.fp16.4clst5gram.$alpha
mkdir -p $SAVE_DIR
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python train.py \
    --task language_modeling_with_ngram \
    $DEST/data-bin/newscrawl.wmt19.bpe.fixbug/ \
    --save-dir $SAVE_DIR \
    --arch transformer_lm_big --share-decoder-input-output-embed \
    --dropout 0.3 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.00 --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 16000 --warmup-init-lr 1e-07 \
    --max-tokens 4096 --update-freq 1 --tokens-per-sample 512 --seed 1 \
    --max-target-positions 512 \
    --max-update 1000000 \
    --sample-break-mode eos  \
    --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_newscrawl_wmt19_clst/newscrawl.wmt19.train.repunk.txt.clst0.min1.arpa.cache;${DEST}/KNQuery/cache_5gram_newscrawl_wmt19_clst/newscrawl.wmt19.train.repunk.txt.clst1.min1.arpa.cache;${DEST}/KNQuery/cache_5gram_newscrawl_wmt19_clst/newscrawl.wmt19.train.repunk.txt.clst2.min1.arpa.cache;${DEST}/KNQuery/cache_5gram_newscrawl_wmt19_clst/newscrawl.wmt19.train.repunk.txt.clst3.min1.arpa.cache" \
    --data2clst-map-path $DEST/wmt19_repr/mbert_avg_6/faiss_index/data2clst_map.pt \
    --ngram-alpha $alpha \
    --ngram-module-path $DEST/KNQuery/ \
    --ngram-warmup-upsdates 0 \
    --fp16 > $SAVE_DIR/log.train &

