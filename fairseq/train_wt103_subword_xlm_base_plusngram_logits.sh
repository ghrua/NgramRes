#!/usr/bin/env bash


########################################
# Adaptive Input Repr
########################################

DEST=/mnt/task_wrapper/user_output/artifacts/


######
# base
######
# SAVE_DIR=$DEST/results/xfm.base.drop0.3.warm4k.wt103.bpe.fp16.plot
# mkdir -p $SAVE_DIR
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python train.py \
#     --task language_modeling \
#     $DEST/data-bin/wikitext-103-bpe \
#     --save-dir $SAVE_DIR \
#     --arch transformer_lm_big --share-decoder-input-output-embed \
#     --dropout 0.3 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.00 --clip-norm 0.0 \
#     --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
#     --max-tokens 8192 --update-freq 1 --tokens-per-sample 2048 --seed 1 \
#     --max-update 50000 \
#     --sample-break-mode none  \
#     --fp16 > $SAVE_DIR/log.train &


######
# base+ngram with different dist size & min dist prob
######

min_dist_prob=1.0
dist_size=100
for alpha in 0.1 0.2 0.3
do
    SAVE_DIR=$DEST/results/xfm.base.drop0.3.warm4k.wt103.bpe.fp16.5gram.alpha$alpha.dp1.0.ds$dist_size.ngramgate
    mkdir -p $SAVE_DIR
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python train.py \
        --task language_modeling_with_ngram \
        $DEST/data-bin/wikitext-103-bpe \
        --save-dir $SAVE_DIR \
        --arch transformer_lm_big --share-decoder-input-output-embed \
        --dropout 0.3 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.00 --clip-norm 0.0 \
        --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
        --max-tokens 8192 --update-freq 1 --tokens-per-sample 2048 --seed 1 \
        --max-update 40000 \
        --sample-break-mode none  \
        --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_wt103_bpe/wiki.train.tokens.bpe.repunk.arpa.cache" \
        --ngram-alpha $alpha \
        --ngram-dist-size $dist_size \
        --ngram-min-dist-prob $min_dist_prob \
        --ngram-module-path $DEST/KNQuery/ \
        --ngram-warmup-updates 0 \
        --ngram-gate \
        --fp16 > $SAVE_DIR/log.train
done

######
# base+gpt with different alpha
######

# for alpha in 0.2 0.4 0.6 0.8 1.0
# do
#     SAVE_DIR=$DEST/results/xfm.base.drop0.3.warm4k.wt103.bpe.fp16.5gram.$alpha
#     mkdir -p $SAVE_DIR
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
#         --task language_modeling_with_ngram \
#         $DEST/data-bin/wikitext-103-bpe \
#         --save-dir $SAVE_DIR \
#         --arch transformer_lm_big --share-decoder-input-output-embed \
#         --dropout 0.3 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.00 --clip-norm 0.0 \
#         --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
#         --max-tokens 8192 --update-freq 1 --tokens-per-sample 2048 --seed 1 \
#         --max-update 50000 \
#         --sample-break-mode none  \
#         --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_wt103_bpe/wiki.train.tokens.bpe.repunk.arpa.cache" \
#         --ngram-alpha $alpha \
#         --ngram-module-path $DEST/KNQuery/ \
#         --ngram-warmup-updates 0 \
#         --fp16 > $SAVE_DIR/log.train.continue
# done



######
# base+gpt with limited lengths of context 
######

# for alpha in 0.2 0.4 0.6 0.8 1.0
# do
#     SAVE_DIR=$DEST/results/xfm.base.wt103.bpe.fp16.5gram.limctxlen.$alpha
#     mkdir -p $SAVE_DIR
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
#         --task language_modeling_with_ngram \
#         $DEST/data-bin/wikitext-103-bpe \
#         --save-dir $SAVE_DIR \
#         --arch transformer_lm_big --share-decoder-input-output-embed \
#         --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.00 --clip-norm 0.0 \
#         --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 16000 --warmup-init-lr 1e-07 \
#         --max-tokens 8192 --update-freq 1 --tokens-per-sample 2048 --seed 1 \
#         --max-update 50000 \
#         --sample-break-mode none  \
#         --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_wt103_bpe/wiki.train.tokens.bpe.repunk.arpa.cache" \
#         --ngram-alpha $alpha \
#         --ngram-module-path $DEST/KNQuery/ \
#         --ngram-warmup-updates 0 \
#         --train-with-limited-ctxlen \
#         --ngram-ctxlen-p "0.1;0.2;0.3;0.4" \
#         --fp16 > $SAVE_DIR/log.train
# done



######
# base+gpt with pruned ngram
######

# for alpha in 0.2 0.4 0.6 0.8 1.0
# do
#     SAVE_DIR=$DEST/results/xfm.base.wt103.bpe.fp16.min3.5gram.$alpha
#     mkdir -p $SAVE_DIR
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
#         --task language_modeling_with_ngram \
#         $DEST/data-bin/wikitext-103-bpe \
#         --save-dir $SAVE_DIR \
#         --arch transformer_lm_big --share-decoder-input-output-embed \
#         --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.00 --clip-norm 0.0 \
#         --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 16000 --warmup-init-lr 1e-07 \
#         --max-tokens 8192 --update-freq 1 --tokens-per-sample 2048 --seed 1 \
#         --max-update 50000 \
#         --sample-break-mode none  \
#         --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_wt103_bpe/wiki.train.tokens.bpe.repunk.min3.arpa.cache" \
#         --ngram-alpha $alpha \
#         --ngram-module-path $DEST/KNQuery/ \
#         --ngram-warmup-updates 0 \
#         --fp16 > $SAVE_DIR/log.train
# done



######
# base+gpt with larger dist size
######

# for alpha in 0.2 0.4 0.6 0.8 1.0
# do
#     SAVE_DIR=$DEST/results/xfm.base.wt103.bpe.fp16.5gram.dist500.$alpha
#     mkdir -p $SAVE_DIR
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
#         --task language_modeling_with_ngram \
#         $DEST/data-bin/wikitext-103-bpe \
#         --save-dir $SAVE_DIR \
#         --arch transformer_lm_big --share-decoder-input-output-embed \
#         --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.00 --clip-norm 0.0 \
#         --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 16000 --warmup-init-lr 1e-07 \
#         --max-tokens 8192 --update-freq 1 --tokens-per-sample 2048 --seed 1 \
#         --max-update 50000 \
#         --sample-break-mode none  \
#         --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_wt103_bpe/wiki.train.tokens.bpe.repunk.arpa.cache" \
#         --ngram-alpha $alpha \
#         --ngram-module-path $DEST/KNQuery/ \
#         --ngram-warmup-updates 0 \
#         --ngram-dist-size 500 \
#         --fp16 > $SAVE_DIR/log.train
# done
