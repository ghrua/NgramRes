DEST=/mnt/task_wrapper/user_output/artifacts/

gpuid=0
L1=vi
L2=en

# MODEL=$DEST/results/iwslt15.$L1-$L2.base/checkpoint_best.pt
# for SUBSET in valid test
# do
#     CUDA_VISIBLE_DEVICES=$gpuid python generate.py $DEST/data-bin/iwslt15.$L1-$L2/ \
#                     --task translation \
#                     --path  $MODEL \
#                     --gen-subset $SUBSET --max-len-a 1.2 --max-len-b 10 \
#                     --beam 5 --batch-size 128 --remove-bpe > $MODEL.$SUBSET.out

#     echo $MODEL.$SUBSET.out
#     grep ^D $MODEL.$SUBSET.out | cut -f3- > $MODEL.$SUBSET.out.sys
#     grep ^T $MODEL.$SUBSET.out | cut -f2- > $MODEL.$SUBSET.out.ref
#     echo "==================================================="
#     python score.py --sys $MODEL.$SUBSET.out.sys --ref $MODEL.$SUBSET.out.ref
#     echo "==================================================="
# done

# MODEL=$DEST/results/iwslt15.$L2-$L1.base/checkpoint_best.pt
# for SUBSET in valid test
# do
#     CUDA_VISIBLE_DEVICES=$gpuid python generate.py $DEST/data-bin/iwslt15.$L2-$L1/ \
#                     --task translation \
#                     --path  $MODEL \
#                     --gen-subset $SUBSET --max-len-a 1.2 --max-len-b 10 \
#                     --beam 5 --batch-size 128 --remove-bpe > $MODEL.$SUBSET.out

#     echo $MODEL.$SUBSET.out
#     grep ^D $MODEL.$SUBSET.out | cut -f3- > $MODEL.$SUBSET.out.sys
#     grep ^T $MODEL.$SUBSET.out | cut -f2- > $MODEL.$SUBSET.out.ref
#     echo "==================================================="
#     python score.py --sys $MODEL.$SUBSET.out.sys --ref $MODEL.$SUBSET.out.ref
#     echo "==================================================="
# done


# alpha=0.0
dist_size=100
min_dist_prob=1.0

for alpha in 0.2 0.4 0.6 0.8
do
    dr=$(echo "$alpha * 0.000025 * 5" | bc)
    MODEL=$DEST/results/iwslt15.$L1-$L2.5gram.alpha$alpha.dist$dist_size.dr$dr/checkpoint_best.pt
    for SUBSET in valid test
    do
        CUDA_VISIBLE_DEVICES=$gpuid python generate.py $DEST/data-bin/iwslt15.$L1-$L2/ \
                        --task translation_with_ngram \
                        --path  $MODEL \
                        --gen-subset $SUBSET --max-len-a 1.2 --max-len-b 10 \
                        --beam 5 --batch-size 128 --remove-bpe \
                        --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_iwslt15_envi/train.${L2}.5gram.arpa.cache" \
                        --ngram-alpha 0.0 \
                        --ngram-dist-size $dist_size \
                        --ngram-min-dist-prob $min_dist_prob \
                        --ngram-module-path $DEST/KNQuery/ > $MODEL.$SUBSET.out

        echo $MODEL.$SUBSET.out
        grep ^D $MODEL.$SUBSET.out | cut -f3- > $MODEL.$SUBSET.out.sys
        grep ^T $MODEL.$SUBSET.out | cut -f2- > $MODEL.$SUBSET.out.ref
        echo "==================================================="
        python score.py --sys $MODEL.$SUBSET.out.sys --ref $MODEL.$SUBSET.out.ref
        echo "==================================================="
    done
done

# L1=en
# L2=vi
# dist_size=100
# min_dist_prob=1.0
# for alpha in 0.05 0.1 0.2
# do
#     # dr=$(echo "$alpha * 0.000025 * 5" | bc)
#     MODEL=$DEST/results/iwslt15.$L1-$L2.5gram.alpha$alpha.dist100/checkpoint_best.pt
#     for SUBSET in valid test
#     do
#         CUDA_VISIBLE_DEVICES=$gpuid python generate.py $DEST/data-bin/iwslt15.$L1-$L2/ \
#                         --task translation_with_ngram \
#                         --path  $MODEL \
#                         --gen-subset $SUBSET --max-len-a 1.2 --max-len-b 10 \
#                         --beam 5 --batch-size 128 --remove-bpe \
#                         --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_iwslt15_ende/train.${L2}.5gram.arpa.cache" \
#                         --ngram-alpha $alpha \
#                         --ngram-dist-size $dist_size \
#                         --ngram-min-dist-prob $min_dist_prob \
#                         --ngram-module-path $DEST/KNQuery/ > $MODEL.$SUBSET.out

#         echo $MODEL.$SUBSET.out
#         grep ^D $MODEL.$SUBSET.out | cut -f3- > $MODEL.$SUBSET.out.sys
#         grep ^T $MODEL.$SUBSET.out | cut -f2- > $MODEL.$SUBSET.out.ref
#         echo "==================================================="
#         python score.py --sys $MODEL.$SUBSET.out.sys --ref $MODEL.$SUBSET.out.ref
#         echo "==================================================="
#     done
# done

# L1=vi
# L2=en
# for alpha in 0.05 0.1 0.2
# do
#     # dr=$(echo "$alpha * 0.000025 * 5" | bc)
#     MODEL=$DEST/results/iwslt15.$L1-$L2.5gram.alpha$alpha.dist100/checkpoint_best.pt
#     for SUBSET in valid test
#     do
#         CUDA_VISIBLE_DEVICES=$gpuid python generate.py $DEST/data-bin/iwslt15.$L1-$L2/ \
#                         --task translation_with_ngram \
#                         --path  $MODEL \
#                         --gen-subset $SUBSET --max-len-a 1.2 --max-len-b 10 \
#                         --beam 5 --batch-size 128 --remove-bpe \
#                         --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_iwslt15_ende/train.${L2}.5gram.arpa.cache" \
#                         --ngram-alpha $alpha \
#                         --ngram-dist-size $dist_size \
#                         --ngram-min-dist-prob $min_dist_prob \
#                         --ngram-module-path $DEST/KNQuery/ > $MODEL.$SUBSET.out

#         echo $MODEL.$SUBSET.out
#         grep ^D $MODEL.$SUBSET.out | cut -f3- > $MODEL.$SUBSET.out.sys
#         grep ^T $MODEL.$SUBSET.out | cut -f2- > $MODEL.$SUBSET.out.ref
#         echo "==================================================="
#         python score.py --sys $MODEL.$SUBSET.out.sys --ref $MODEL.$SUBSET.out.ref
#         echo "==================================================="
#     done
# done