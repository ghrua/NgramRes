DEST=/mnt/task_wrapper/user_output/artifacts/

########################################
# Base model
########################################
L1=en
L2=de


# SAVE_DIR=$DEST/results/multitask.${L1}${L2}.base
# mkdir -p $SAVE_DIR
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup fairseq-train \
#     --task translation \
#     $DEST/data-bin/multi-task-mt-${L1}${L2}-bin/ \
#     --save-dir $SAVE_DIR \
#     --arch transformer_wmt_en_de \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#     --weight-decay 0.0 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 4096 \
#     --update-freq 1 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-tokenized-bleu \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#     --max-epoch 30 > $SAVE_DIR/log.train

########################################
# Base model + 5gram
########################################

alpha=0.1
dist_size=100
min_dist_prob=1.0
DATADIR=$DEST/data/multi_domain_new_split_multitask_version

for alpha in 0.05 0.075 0.1
do
    SAVE_DIR=$DEST/results/multitask.${L1}${L2}.5gram.alpha$alpha.dist$dist_size
    mkdir -p $SAVE_DIR
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train \
        --task translation_with_ngram \
        $DEST/data-bin/multi-task-mt-${L1}${L2}-bin/ \
        --save-dir $SAVE_DIR \
        --arch transformer_wmt_en_de \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
        --weight-decay 0.0 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-tokens 4096 \
        --update-freq 1 \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-tokenized-bleu \
        --eval-bleu-remove-bpe \
        --eval-bleu-print-samples \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --max-epoch 30 \
        --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_multitask_it/train.${L2}.tok.bpe32k.5gram.arpa.cache;${DEST}/KNQuery/cache_5gram_multitask_koran/train.${L2}.tok.bpe32k.5gram.arpa.cache;${DEST}/KNQuery/cache_5gram_multitask_law/train.${L2}.tok.bpe32k.5gram.arpa.cache;${DEST}/KNQuery/cache_5gram_multitask_medical/train.${L2}.tok.bpe32k.5gram.arpa.cache;${DEST}/KNQuery/cache_5gram_multitask_subtitles/train.${L2}.tok.bpe32k.5gram.arpa.cache" \
        --data2clst-map-path "${DATADIR}/data2clst_map_train_${L2}.pt;${DATADIR}/data2clst_map_dev_${L2}.pt" \
        --ngram-alpha $alpha \
        --ngram-dist-size $dist_size \
        --ngram-min-dist-prob $min_dist_prob \
        --ngram-module-path $DEST/KNQuery/ > $SAVE_DIR/log.train
done