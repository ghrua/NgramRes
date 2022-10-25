#!/usr/bin/env bash


########################################
# Base model
########################################

DEST=/mnt/task_wrapper/user_output/artifacts/

SAVE_DIR=$DEST/results/multitask.xfm.base.drop0.3.wmt19.bpe.fp16
mkdir -p $SAVE_DIR
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --task language_modeling \
    $DEST/data-bin/multi-task-lm-en-bin/  \
    --save-dir $SAVE_DIR \
    --arch transformer_lm_big --share-decoder-input-output-embed \
    --dropout 0.3 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.00 --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --max-tokens 4096 --update-freq 1 --tokens-per-sample 512 --seed 1 \
    --max-target-positions 512 \
    --max-epoch 50 \
    --sample-break-mode eos \
    --fp16 > $SAVE_DIR/log.train

######
# base+5gram
######

for alpha in 0.1 0.2 0.4 0.6
do
    DATADIR=$DEST/data/multi_domain_new_split_multitask_version
    SAVE_DIR=$DEST/results/multitask.xfm.base.drop0.3.wmt19.bpe.fp16.5gram.$alpha
    mkdir -p $SAVE_DIR
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
        --task language_modeling_with_ngram \
        $DEST/data-bin/multi-task-lm-en-bin/ \
        --save-dir $SAVE_DIR \
        --arch transformer_lm_big --share-decoder-input-output-embed \
        --dropout 0.3 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.00 --clip-norm 0.0 \
        --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
        --max-tokens 4096 --update-freq 1 --tokens-per-sample 512 --seed 1 \
        --max-target-positions 512 \
        --max-epoch 50 \
        --sample-break-mode eos  \
        --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_multitask_it/train.en.tok.bpe32k.5gram.arpa.cache;${DEST}/KNQuery/cache_5gram_multitask_koran/train.en.tok.bpe32k.5gram.arpa.cache;${DEST}/KNQuery/cache_5gram_multitask_law/train.en.tok.bpe32k.5gram.arpa.cache;${DEST}/KNQuery/cache_5gram_multitask_medical/train.en.tok.bpe32k.5gram.arpa.cache;${DEST}/KNQuery/cache_5gram_multitask_subtitles/train.en.tok.bpe32k.5gram.arpa.cache" \
        --data2clst-map-path "${DATADIR}/data2clst_map_train_en.pt;${DATADIR}/data2clst_map_dev_en.pt" \
        --ngram-alpha $alpha \
        --ngram-module-path $DEST/KNQuery/ \
        --ngram-warmup-updates 0 \
        --fp16 > $SAVE_DIR/log.train
done
