DEST=/mnt/task_wrapper/user_output/artifacts/

# ln -s examples/bart/summarize.py summarize.py

alpha=0.05
dist_size=100
min_dist_prob=1.0
SUB=test
gpuid=0


SAVE_DIR=$DEST/results/cnndm.bart.large.ft.cat5gram.alpha$alpha.size$dist_size
cp $DEST/data-bin/cnn_dm-bin/dict.source.txt $SAVE_DIR
cp $DEST/data-bin/cnn_dm-bin/dict.target.txt $SAVE_DIR
MODEL_NAME=checkpoint_best.pt
SRC_FILE=$DEST/data/bart_data/cnn_dm/$SUB.source

CUDA_VISIBLE_DEVICES=$gpuid python examples/bart/summarize.py  \
    --model-dir $SAVE_DIR \
    --model-file $MODEL_NAME \
    --src $SRC_FILE \
    --out $SRC_FILE.debug.hypo.$alpha

# for SRC_FILE in `ls $DEST/data/bart_data/cnn_dm/$SUB.source.shard.a[a-z]`
# do
    # CUDA_VISIBLE_DEVICES=$gpuid nohup python examples/bart/summarize.py $DEST/data-bin/cnn_dm-bin/ \
    #     --model-dir $SAVE_DIR \
    #     --model-file $MODEL_NAME \
    #     --src $SRC_FILE \
    #     --out $SRC_FILE.hypo.$alpha \
    #     --task translation_with_ngram \
    #     --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_cnn_dm/train.bpe.cat.5gram.min1.arpa.cache" \
    #     --ngram-alpha $alpha \
    #     --ngram-dist-size $dist_size \
    #     --ngram-min-dist-prob $min_dist_prob \
    #     --ngram-module-path $DEST/KNQuery/ > $SRC_FILE.log.$alpha &
    # let "gpuid=gpuid+1"


        # exit 0
        # --task translation_with_ngram \
        # --ngram-generation-model-cache "${DEST}/KNQuery/cache_5gram_cnn_dm/train.bpe.cat.5gram.min1.arpa.cache" \
        # --ngram-alpha $alpha \
        # --ngram-dist-size $dist_size \
        # --ngram-min-dist-prob $min_dist_prob \
        # --ngram-module-path $DEST/KNQuery/ > $SRC_FILE.log.$alpha
# done




# MERGE_FILE=$DEST/data/bart_data/cnn_dm/$SUB.source.hypo.$alpha
# rm -rf $MERGE_FILE
# touch $MERGE_FILE

# for SRC_FILE in `ls $DEST/data/bart_data/cnn_dm/$SUB.source.shard.*.hypo.$alpha`
# do
#     cat $SRC_FILE >> $MERGE_FILE
# done

# # export CLASSPATH=$DEST/corenlp/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar
# # # Tokenize hypothesis and target files.
# cat $MERGE_FILE | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $MERGE_FILE.tokenized
# cat $DEST/data/bart_data/cnn_dm/test.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $DEST/data/bart_data/cnn_dm/test.target.tokenized
# files2rouge $DEST/data/bart_data/cnn_dm/test.target.tokenized $MERGE_FILE.tokenized
# Expected output: (ROUGE-2 Average_F: 0.21238) 
