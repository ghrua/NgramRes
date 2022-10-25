DEST=/mnt/task_wrapper/user_output/artifacts/

TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=500
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=4
BART_PATH=$DEST/fairseq_hub/bart.large/model.pt


#########
# base
#########

# SAVE_DIR=$DEST/results/cnndm.bart.large.ft
# mkdir -p $SAVE_DIR

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup fairseq-train $DEST/data-bin/cnn_dm-bin \
#     --save-dir $SAVE_DIR \
#     --restore-file $BART_PATH \
#     --max-tokens $MAX_TOKENS \
#     --task translation \
#     --source-lang source --target-lang target \
#     --truncate-source \
#     --layernorm-embedding \
#     --share-all-embeddings \
#     --share-decoder-input-output-embed \
#     --reset-optimizer --reset-dataloader --reset-meters \
#     --required-batch-size-multiple 1 \
#     --arch bart_large \
#     --criterion label_smoothed_cross_entropy \
#     --label-smoothing 0.1 \
#     --dropout 0.1 --attention-dropout 0.1 \
#     --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
#     --clip-norm 0.1 \
#     --dropout 0.1 --attention-dropout 0.1 \
#     --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
#     --clip-norm 0.1 \
#     --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
#     --fp16 --update-freq $UPDATE_FREQ \
#     --skip-invalid-size-inputs-valid-test \
#     --find-unused-parameters > $SAVE_DIR/log.train &


#########
# base
#########

alpha=0.1
dist_size=100
min_dist_prob=1.0

for alpha in 0.1 0.2
do
    SAVE_DIR=$DEST/results/cnndm.bart.large.ft.cat5gram.alpha$alpha.size$dist_size
    mkdir -p $SAVE_DIR
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup fairseq-train $DEST/data-bin/cnn_dm-bin \
        --save-dir $SAVE_DIR \
        --restore-file $BART_PATH \
        --max-tokens $MAX_TOKENS \
        --task translation_with_ngram \
        --source-lang source --target-lang target \
        --truncate-source \
        --layernorm-embedding \
        --share-all-embeddings \
        --share-decoder-input-output-embed \
        --reset-optimizer --reset-dataloader --reset-meters \
        --required-batch-size-multiple 1 \
        --arch bart_large \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --dropout 0.1 --attention-dropout 0.1 \
        --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
        --clip-norm 0.1 \
        --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
        --max-update $TOTAL_NUM_UPDATES \
        --fp16 --update-freq $UPDATE_FREQ \
        --skip-invalid-size-inputs-valid-test \
        --find-unused-parameters \
        --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_cnn_dm/train.bpe.cat.5gram.min1.arpa.cache" \
        --ngram-alpha $alpha \
        --ngram-dist-size $dist_size \
        --ngram-min-dist-prob $min_dist_prob \
        --ngram-module-path $DEST/KNQuery/ > $SAVE_DIR/log.train
done
