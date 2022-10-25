DEST=/mnt/task_wrapper/user_output/artifacts/

gpuid=0
alpha=0.1
dist_size=100
min_dist_prob=1.0
DATADIR=$DEST/data/multi_domain_new_split_multitask_version
SUBSET=test
L1=en
L2=de

# for DOMAIN in it koran law medical subtitles
# do
#     SAVE_DIR=$DEST/results/multitask.${L1}${L2}.base
#     MODEL=$SAVE_DIR/checkpoint_best.pt
#     CUDA_VISIBLE_DEVICES=$gpuid python generate.py $DEST/data-bin/multi-task-mt-${L1}${L2}-$DOMAIN-bin/ \
#                     --task translation \
#                     --path  $MODEL \
#                     --gen-subset $SUBSET --max-len-a 1.2 --max-len-b 10 \
#                     --beam 5 --batch-size 128 --remove-bpe > $MODEL.$SUBSET.$DOMAIN.out

#     echo $MODEL.$SUBSET.$DOMAIN.out
#     grep ^D $MODEL.$SUBSET.$DOMAIN.out | cut -f3- > $MODEL.$SUBSET.$DOMAIN.out.sys
#     grep ^T $MODEL.$SUBSET.$DOMAIN.out | cut -f2- > $MODEL.$SUBSET.$DOMAIN.out.ref
#     echo "==================================================="
#     python score.py --sys $MODEL.$SUBSET.$DOMAIN.out.sys --ref $MODEL.$SUBSET.$DOMAIN.out.ref
#     echo "==================================================="
# done

for alpha in 0.05 0.075 0.1
do
    for DOMAIN in it koran law medical subtitles
    do
        SAVE_DIR=$DEST/results/multitask.${L1}${L2}.5gram.alpha$alpha.dist$dist_size
        MODEL=$SAVE_DIR/checkpoint_best.pt
        CUDA_VISIBLE_DEVICES=$gpuid python generate.py $DEST/data-bin/multi-task-mt-${L1}${L2}-$DOMAIN-bin/ \
                        --task translation_with_ngram \
                        --path  $MODEL \
                        --gen-subset $SUBSET --max-len-a 1.2 --max-len-b 10 \
                        --beam 5 --batch-size 128 --remove-bpe \
                        --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_multitask_${DOMAIN}/train.${L2}.tok.bpe32k.5gram.arpa.cache" \
                        --ngram-alpha $alpha \
                        --ngram-dist-size $dist_size \
                        --ngram-min-dist-prob $min_dist_prob \
                        --ngram-module-path $DEST/KNQuery/ > $MODEL.$SUBSET.$DOMAIN.out
        echo $MODEL.$SUBSET.$DOMAIN.out
        grep ^D $MODEL.$SUBSET.$DOMAIN.out | cut -f3- > $MODEL.$SUBSET.$DOMAIN.out.sys
        grep ^T $MODEL.$SUBSET.$DOMAIN.out | cut -f2- > $MODEL.$SUBSET.$DOMAIN.out.ref
        echo "==================================================="
        python score.py --sys $MODEL.$SUBSET.$DOMAIN.out.sys --ref $MODEL.$SUBSET.$DOMAIN.out.ref
        echo "==================================================="
    done
done