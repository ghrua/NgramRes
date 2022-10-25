DEST=/mnt/task_wrapper/user_output/artifacts/

gpuid=0
L1=en
L2=de
SUBSET=test


# MODEL=$DEST/results/iwslt14.$L1-$L2.base/checkpoint_best.pt
# CUDA_VISIBLE_DEVICES=$gpuid python generate.py $DEST/data-bin/iwslt14.$L1-$L2/ \
#                 --task translation \
#                 --path  $MODEL \
#                 --gen-subset $SUBSET --max-len-a 1.2 --max-len-b 10 \
#                 --beam 5 --batch-size 128 --remove-bpe > $MODEL.$SUBSET.out

# echo $MODEL.$SUBSET.out
# grep ^D $MODEL.$SUBSET.out | cut -f3- > $MODEL.$SUBSET.out.sys
# grep ^T $MODEL.$SUBSET.out | cut -f2- > $MODEL.$SUBSET.out.ref
# echo "==================================================="
# python score.py --sys $MODEL.$SUBSET.out.sys --ref $MODEL.$SUBSET.out.ref
# echo "==================================================="

# dist_size=100
# min_dist_prob=1.0
# SUBSET=test
# for alpha in 0.05 0.07
# do
#     MODEL=$DEST/results/iwslt14.$L1-$L2.5gram.alpha$alpha.dist$dist_size/checkpoint_best.pt
#     CUDA_VISIBLE_DEVICES=$gpuid python generate.py $DEST/data-bin/iwslt14.$L1-$L2/ \
#                     --task translation_with_ngram \
#                     --path  $MODEL \
#                     --gen-subset $SUBSET --max-len-a 1.2 --max-len-b 10 \
#                     --beam 5 --batch-size 128 --remove-bpe \
#                     --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_iwslt14_en${L2}/train.${L2}.5gram.arpa.cache" \
#                     --ngram-alpha $alpha \
#                     --ngram-dist-size $dist_size \
#                     --ngram-min-dist-prob $min_dist_prob \
#                     --ngram-module-path $DEST/KNQuery/ > $MODEL.$SUBSET.out

#     echo $MODEL.$SUBSET.out
#     grep ^D $MODEL.$SUBSET.out | cut -f3- > $MODEL.$SUBSET.out.sys
#     grep ^T $MODEL.$SUBSET.out | cut -f2- > $MODEL.$SUBSET.out.ref
#     echo "==================================================="
#     python score.py --sys $MODEL.$SUBSET.out.sys --ref $MODEL.$SUBSET.out.ref
#     echo "==================================================="
# done
# dist_size=100
# min_dist_prob=1.0
# SUBSET=test
# for alpha in 0.07 0.09 0.11 0.13 0.15 0.17 0.19
# do
#     MODEL=$DEST/results/iwslt14.$L1-$L2.5gram.4shards.alpha$alpha.dist$dist_size/checkpoint_best.pt
#     CUDA_VISIBLE_DEVICES=$gpuid python generate.py $DEST/data-bin/iwslt14.$L1-$L2/ \
#                     --task translation_with_ngram \
#                     --path  $MODEL \
#                     --gen-subset $SUBSET --max-len-a 1.2 --max-len-b 10 \
#                     --beam 5 --batch-size 128 --remove-bpe \
#                     --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_iwslt14_en${L2}/train.${L2}.5gram.arpa.cache" \
#                     --ngram-alpha $alpha \
#                     --ngram-dist-size $dist_size \
#                     --ngram-min-dist-prob $min_dist_prob \
#                     --ngram-module-path $DEST/KNQuery/ > $MODEL.$SUBSET.out

#     echo $MODEL.$SUBSET.out
#     grep ^D $MODEL.$SUBSET.out | cut -f3- > $MODEL.$SUBSET.out.sys
#     grep ^T $MODEL.$SUBSET.out | cut -f2- > $MODEL.$SUBSET.out.ref
#     echo "==================================================="
#     python score.py --sys $MODEL.$SUBSET.out.sys --ref $MODEL.$SUBSET.out.ref
#     echo "==================================================="
# done

dist_size=100
min_dist_prob=1.0
SUBSET=test
min_alpha=0.01
alpha=0.8
for alpha in 0.6 0.8
do
    for CKPT in 27 28 29 30
    do
        dr=$(echo "$alpha * 0.000025 * 5" | bc)
        MODEL=$DEST/results/iwslt14.$L1-$L2.5gram.4shards.alpha$alpha.minalpha$min_alpha.dist$dist_size.dr$dr/checkpoint$CKPT.pt
        # CUDA_VISIBLE_DEVICES=$gpuid nohup python generate.py $DEST/data-bin/iwslt14.$L1-$L2/ \
        #                 --task translation_with_ngram \
        #                 --path  $MODEL \
        #                 --gen-subset $SUBSET --max-len-a 1.2 --max-len-b 10 \
        #                 --beam 5 --batch-size 128 --remove-bpe \
        #                 --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_iwslt14_en${L2}/train.${L2}.5gram.arpa.cache" \
        #                 --ngram-alpha $min_alpha \
        #                 --ngram-dist-size 1000 \
        #                 --ngram-min-dist-prob $min_dist_prob \
        #                 --ngram-module-path $DEST/KNQuery/ > $MODEL.$SUBSET.out &
        # let "gpuid=gpuid+1"
        echo $MODEL.$SUBSET.out
        grep ^D $MODEL.$SUBSET.out | cut -f3- > $MODEL.$SUBSET.out.sys
        grep ^T $MODEL.$SUBSET.out | cut -f2- > $MODEL.$SUBSET.out.ref
        echo "==================================================="
        python score.py --sys $MODEL.$SUBSET.out.sys --ref $MODEL.$SUBSET.out.ref
        echo "==================================================="
    done
done