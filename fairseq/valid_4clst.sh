DEST=/mnt/task_wrapper/user_output/artifacts/
# SUB=test

MODEL=$DEST/results/adaptive.input.wt103.5gram.fp16/checkpoint_best.pt
# CUDA_VISIBLE_DEVICES=0 python eval_lm.py \
#     $DEST/data-bin/wikitext-103 \
#     --gen-subset "valid;test" \
#     --task language_modeling_with_ngram \
#     --path $MODEL \
#     --distributed-world-size 1 \
#     --batch-size 2  \
#     --context-window 1536 \
#     --tokens-per-sample 2048 \
#     --ngram-generation-model-cache "${DEST}/KNQuery/cache_3gram_fairseq/fgm.wt103.kenlm.3gram.cache.fairseq.shard" \
#     --ngram-module-path $DEST/KNQuery/

# gpuid=0
# dist=100
# SUB=test
# alpha=0.1
# for ctxlen in 2 3
# do
#     CUDA_VISIBLE_DEVICES=$gpuid python eval_lm.py \
#     $DEST/data-bin/wikitext-103 \
#     --gen-subset $SUB \
#     --task language_modeling_with_ngram \
#     --path $MODEL \
#     --distributed-world-size 1 \
#     --batch-size 2  \
#     --context-window 1536 \
#     --tokens-per-sample 2048 \
#     --ngram-dist-size $dist \
#     --eval-ngram-alpha $alpha \
#     --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_fairseq/fgm.wt103.train.fairseq.arpa.cache" \
#     --ngram-min-ctxlen $ctxlen \
#     --ngram-min-accum-prob 0.0 \
#     --ngram-module-path $DEST/KNQuery/
# done

    #  > ./log.valid.wt103.debug &

# for alpha in 0.1 0.2 0.3
# do
#     for SUB in valid test
#     do
#         CUDA_VISIBLE_DEVICES=$gpuid nohup python eval_lm.py \
#             $DEST/data-bin/wikitext-103 \
#             --gen-subset $SUB \
#             --task language_modeling_with_ngram \
#             --path $MODEL \
#             --distributed-world-size 1 \
#             --batch-size 2  \
#             --context-window 1536 \
#             --tokens-per-sample 2048 \
#             --ngram-dist-size $dist \
#             --eval-ngram-alpha $alpha \
#             --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_fairseq/fgm.wt103.train.fairseq.arpa.cache" \
#             --ngram-module-path $DEST/KNQuery/ > ./log.valid.wt103.$SUB.$dist.$alpha &

#         gpuid=$((gpuid+1))
#     done
# done


# 
# alpha=1.0
# SUB=valid
# CUDA_VISIBLE_DEVICES=0 python eval_lm.py \
#     $DEST/data-bin/multi_domain_law/ \
#     --gen-subset $SUB \
#     --task language_modeling_with_ngram \
#     --path $MODEL \
#     --distributed-world-size 1 \
#     --batch-size 2  \
#     --context-window 1536 \
#     --tokens-per-sample 2048 \
#     --eval-ngram-alpha $alpha \
#     --ngram-dist-size 5000 \
#     --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_multi_domain_fairseq/law.train.wt103vocab.cache" \
#     --ngram-module-path $DEST/KNQuery/


# for SUB in valid test
# do

# SUB=valid
# gpuid=0
# dist=100
# for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
# do
#     CUDA_VISIBLE_DEVICES=$gpuid nohup python eval_lm.py \
#         $DEST/data-bin/multi_domain_law/ \
#         --gen-subset $SUB \
#         --task language_modeling_with_ngram \
#         --path $MODEL \
#         --distributed-world-size 1 \
#         --batch-size 2  \
#         --context-window 1536 \
#         --tokens-per-sample 2048 \
#         --eval-ngram-alpha $alpha \
#         --ngram-dist-size $dist \
#         --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_multi_domain_fairseq/law.train.wt103vocab.cache" \
#         --ngram-module-path $DEST/KNQuery/ > log.law.$SUB.$dist.$alpha &
#     let "gpuid=gpuid+1"
# done
# # done

# SUB=test
# dist=100
# alpha=0.1
# MODEL=$DEST/results/transformer_lm.wmt19.en.ft.5gram/checkpoint29.pt
# CUDA_VISIBLE_DEVICES=0 python eval_lm.py \
#     $DEST/data-bin/multidomain.law.wmt19.bpe/ \
#     --gen-subset $SUB \
#     --task language_modeling_with_ngram \
#     --path $MODEL \
#     --distributed-world-size 1 \
#     --batch-size 32  \
#     --context-window 0 \
#     --sample-break-mode eos \
#     --tokens-per-sample 512 \
#     --eval-ngram-alpha $alpha \
#     --ngram-dist-size $dist \
#     --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_newscrawl_wmt19/newscrawl.wmt19.min2.repunk.arpa.cache" \
#     --ngram-module-path $DEST/KNQuery/ \
#     --post-process "@@ " > log/log.law.wmt19.$SUB.$dist.$alpha &
    # --output-word-probs
    #  > log.law.wmt19.$SUB &


# SUB=test
# gpuid=0
# dist=5000
# MODEL=$DEST/results/transformer_lm.wmt19.en.ft.5gram/checkpoint29.pt
# for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
# do
#     CUDA_VISIBLE_DEVICES=$gpuid nohup python eval_lm.py \
#         $DEST/data-bin/multidomain.law.wmt19.bpe/ \
#         --gen-subset $SUB \
#         --task language_modeling_with_ngram \
#         --path $MODEL \
#         --distributed-world-size 1 \
#         --batch-size 32  \
#         --sample-break-mode eos \
#         --tokens-per-sample 512 \
#         --eval-ngram-alpha $alpha \
#         --ngram-dist-size $dist \
#         --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_law_wmt19/law.train.en.wmt19.arpa.cache" \
#         --ngram-module-path $DEST/KNQuery/ \
#         --post-process "@@ " > log/log.law.wmt19.$SUB.$dist.$alpha &
#     let "gpuid=gpuid+1"
# done

# alpha=0.9
# CUDA_VISIBLE_DEVICES=$gpuid nohup python eval_lm.py \
#     $DEST/data-bin/multidomain.law.wmt19.bpe/ \
#     --gen-subset $SUB \
#     --task language_modeling_with_ngram \
#     --path $MODEL \
#     --distributed-world-size 1 \
#     --batch-size 32  \
#     --sample-break-mode eos \
#     --tokens-per-sample 512 \
#     --eval-ngram-alpha $alpha \
#     --ngram-dist-size $dist \
#     --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_law_wmt19/law.train.en.wmt19.arpa.cache" \
#     --ngram-module-path $DEST/KNQuery/ \
#     --post-process "@@ " > log/log.law.wmt19.$SUB.$dist.$alpha &
# let "gpuid=gpuid+1"

SUB=valid
gpuid=0
dist=1000
MODEL=$DEST/results/xfm.base.drop0.3.wmt19.bpe.fp16.4clst5gram.0.6/checkpoint_best.pt
LOG_DIR=$DEST/results/xfm.base.drop0.3.wmt19.bpe.fp16.4clst5gram.0.6/log
echo $LOG_DIR
mkdir -p $LOG_DIR

for alpha in 0.8 0.9 1.0
    do
        CUDA_VISIBLE_DEVICES=$gpuid nohup python eval_lm.py \
            $DEST/data-bin/multidomain.law.wmt19.bpe/ \
            --gen-subset $SUB \
            --task language_modeling_with_ngram \
            --path $MODEL \
            --distributed-world-size 1 \
            --batch-size 32  \
            --sample-break-mode eos \
            --tokens-per-sample 512 \
            --output-word-probs \
            --prob-save-path $LOG_DIR/prob.xfm.$SUB.$alpha.pt \
            --weighted-ctxlen \
            --eval-ngram-alpha $alpha \
            --ngram-dist-size $dist \
            --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_law_wmt19/law.train.en.wmt19.arpa.cache" \
            --ngram-module-path $DEST/KNQuery/ > $LOG_DIR/log.law.wmt19.inter.4clst.$SUB.$dist.$alpha.prob.brute &
        let "gpuid=gpuid+1"
done
