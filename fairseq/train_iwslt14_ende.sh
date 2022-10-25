DEST=/mnt/task_wrapper/user_output/artifacts/

gpuid=0


# L1=en
# L2=de
# SAVE_DIR=$DEST/results/iwslt14.$L1-$L2.base
# mkdir -p $SAVE_DIR
# CUDA_VISIBLE_DEVICES=$gpuid nohup fairseq-train \
#     --task translation \
#     $DEST/data-bin/iwslt14.$L1-$L2/ \
#     --save-dir $SAVE_DIR \
#     --arch transformer_iwslt_de_en \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#     --dropout 0.3 --weight-decay 0.0001 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 4096 \
#     --max-epoch 30 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok space \
#     --eval-tokenized-bleu \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric > $SAVE_DIR/log.train.continue &
# let "gpuid=gpuid+1"


# L1=de
# L2=en
# SAVE_DIR=$DEST/results/iwslt14.$L1-$L2.base
# mkdir -p $SAVE_DIR
# CUDA_VISIBLE_DEVICES=$gpuid nohup fairseq-train \
#     --task translation \
#     $DEST/data-bin/iwslt14.$L1-$L2/ \
#     --save-dir $SAVE_DIR \
#     --arch transformer_iwslt_de_en \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#     --dropout 0.3 --weight-decay 0.0001 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 4096 \
#     --max-epoch 30 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok space \
#     --eval-tokenized-bleu \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric > $SAVE_DIR/log.train.continue &
# let "gpuid=gpuid+1"

L1=en
L2=de
dist_size=100
min_dist_prob=1.0
for alpha in 0.05 0.07 0.09 0.11 0.13 0.15 0.17 0.19
do
    SAVE_DIR=$DEST/results/iwslt14.$L1-$L2.5gram.4shards.alpha$alpha.dist$dist_size.nowd
    mkdir -p $SAVE_DIR
    CUDA_VISIBLE_DEVICES=$gpuid nohup fairseq-train \
        --task translation_with_ngram \
        $DEST/data-bin/iwslt14.$L1-$L2/ \
        --save-dir $SAVE_DIR \
        --arch transformer_iwslt_de_en \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
        --dropout 0.3 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-tokens 4096 \
        --max-epoch 30 \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok space \
        --eval-tokenized-bleu \
        --eval-bleu-remove-bpe \
        --eval-bleu-print-samples \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_iwslt14_en${L2}/train.${L2}.shard0.5gram.arpa.cache;${DEST}/KNQuery/cache_5gram_iwslt14_en${L2}/train.${L2}.shard1.5gram.arpa.cache;${DEST}/KNQuery/cache_5gram_iwslt14_en${L2}/train.${L2}.shard2.5gram.arpa.cache;${DEST}/KNQuery/cache_5gram_iwslt14_en${L2}/train.${L2}.shard3.5gram.arpa.cache" \
        --data2clst-map-path $DEST/data/iwslt14_en$L2/iwslt14.tokenized.$L2-en/data2clst_4shard_map_$L2.pt \
        --ngram-alpha $alpha \
        --ngram-dist-size $dist_size \
        --ngram-min-dist-prob $min_dist_prob \
        --ngram-module-path $DEST/KNQuery/ > $SAVE_DIR/log.train &
    let "gpuid=gpuid+1"
done


# L1=en
# L2=de
# alpha=0.1
# dist_size=100
# min_dist_prob=1.0
# base_dr=0.000025
# for alpha in 0.2 0.4 0.6 0.8
# do
#     dr=$(echo "$alpha * 0.000025 * 5" | bc)
#     SAVE_DIR=$DEST/results/iwslt14.$L1-$L2.5gram.alpha$alpha.dist$dist_size.dr$dr
#     mkdir -p $SAVE_DIR
#     CUDA_VISIBLE_DEVICES=$gpuid nohup fairseq-train \
#         --task translation_with_ngram \
#         $DEST/data-bin/iwslt14.$L1-$L2/ \
#         --save-dir $SAVE_DIR \
#         --arch transformer_iwslt_de_en \
#         --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#         --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#         --dropout 0.3 --weight-decay 0.0001 \
#         --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#         --max-tokens 4096 \
#         --max-epoch 30 \
#         --eval-bleu \
#         --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#         --eval-bleu-detok space \
#         --eval-tokenized-bleu \
#         --eval-bleu-remove-bpe \
#         --eval-bleu-print-samples \
#         --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#         --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_iwslt14_ende/train.${L2}.5gram.arpa.cache" \
#         --ngram-alpha $alpha \
#         --ngram-alpha-decay-rate $dr \
#         --ngram-dist-size $dist_size \
#         --ngram-min-dist-prob $min_dist_prob \
#         --ngram-module-path $DEST/KNQuery/ > $SAVE_DIR/log.train &
#     let "gpuid=gpuid+1"
# done



# L1=de
# L2=en
# alpha=0.1
# dist_size=100
# min_dist_prob=1.0
# base_dr=0.000025
# for alpha in 0.2 0.4 0.6 0.8
# do
#     dr=$(echo "$alpha * 0.000025 * 5" | bc)
#     SAVE_DIR=$DEST/results/iwslt14.$L1-$L2.5gram.alpha$alpha.dist$dist_size.dr$dr
#     mkdir -p $SAVE_DIR
#     CUDA_VISIBLE_DEVICES=$gpuid nohup fairseq-train \
#         --task translation_with_ngram \
#         $DEST/data-bin/iwslt14.$L1-$L2/ \
#         --save-dir $SAVE_DIR \
#         --arch transformer_iwslt_de_en \
#         --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#         --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#         --dropout 0.3 --weight-decay 0.0001 \
#         --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#         --max-tokens 4096 \
#         --max-epoch 30 \
#         --eval-bleu \
#         --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#         --eval-bleu-detok space \
#         --eval-tokenized-bleu \
#         --eval-bleu-remove-bpe \
#         --eval-bleu-print-samples \
#         --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#         --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_iwslt14_ende/train.${L2}.5gram.arpa.cache" \
#         --ngram-alpha $alpha \
#         --ngram-alpha-decay-rate $dr \
#         --ngram-dist-size $dist_size \
#         --ngram-min-dist-prob $min_dist_prob \
#         --ngram-module-path $DEST/KNQuery/ > $SAVE_DIR/log.train &
#     let "gpuid=gpuid+1"
# done


# L1=en
# L2=de
# alpha=0.1
# dist_size=100
# min_dist_prob=1.0
# base_dr=0.000025
# min_alpha=0.01
# for alpha in 0.6 0.8
# do
#     dr=$(echo "$alpha * 0.000025 * 5" | bc)
#     SAVE_DIR=$DEST/results/iwslt14.$L1-$L2.5gram.4shards.alpha$alpha.minalpha$min_alpha.dist$dist_size.dr$dr
#     mkdir -p $SAVE_DIR
#     CUDA_VISIBLE_DEVICES=$gpuid nohup fairseq-train \
#         --task translation_with_ngram \
#         $DEST/data-bin/iwslt14.$L1-$L2/ \
#         --save-dir $SAVE_DIR \
#         --arch transformer_iwslt_de_en \
#         --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#         --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#         --dropout 0.3 --weight-decay 0.0001 \
#         --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#         --max-tokens 4096 \
#         --max-epoch 30 \
#         --eval-bleu \
#         --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#         --eval-bleu-detok space \
#         --eval-tokenized-bleu \
#         --eval-bleu-remove-bpe \
#         --eval-bleu-print-samples \
#         --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#         --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_iwslt14_en${L2}/train.${L2}.shard0.5gram.arpa.cache;${DEST}/KNQuery/cache_5gram_iwslt14_en${L2}/train.${L2}.shard1.5gram.arpa.cache;${DEST}/KNQuery/cache_5gram_iwslt14_en${L2}/train.${L2}.shard2.5gram.arpa.cache;${DEST}/KNQuery/cache_5gram_iwslt14_en${L2}/train.${L2}.shard3.5gram.arpa.cache" \
#         --data2clst-map-path $DEST/data/iwslt14_en$L2/iwslt14.tokenized.$L2-en/data2clst_4shard_map_$L2.pt \
#         --ngram-alpha $alpha \
#         --min-ngram-alpha $min_alpha \
#         --ngram-alpha-decay-rate $dr \
#         --ngram-dist-size $dist_size \
#         --ngram-min-dist-prob $min_dist_prob \
#         --ngram-module-path $DEST/KNQuery/ > $SAVE_DIR/log.train &
#     let "gpuid=gpuid+1"
# done


