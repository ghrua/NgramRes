DEST=/mnt/task_wrapper/user_output/artifacts/
alpha=0.1
dist_size=100
min_dist_prob=1.0

SAVE_DIR=$DEST/results/wmt14.ende.5gram.alpha$alpha.dist$dist_size
mkdir -p $SAVE_DIR
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup fairseq-train \
    --task translation_with_ngram \
    $DEST/data-bin/wmt14.tokenized.en-de/ \
    --save-dir $SAVE_DIR \
    --arch transformer_wmt_en_de --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --update-freq 1 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok space \
    --eval-tokenized-bleu \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-interval-updates 1000 --max-update 100000 \
    --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_wmt14_ende/train.bpe32k.de.5gram.arpa.cache" \
    --ngram-alpha $alpha \
    --ngram-dist-size $dist_size \
    --ngram-min-dist-prob $min_dist_prob \
    --ngram-module-path $DEST/KNQuery/ > $SAVE_DIR/log.train &
