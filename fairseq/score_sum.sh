DEST=/mnt/task_wrapper/user_output/artifacts/


alpha=0.05
dist_size=100
min_dist_prob=1.0
SUB=test
gpuid=0

MERGE_FILE=$DEST/data/bart_data/cnn_dm/$SUB.source.debug.hypo.$alpha
# rm -rf $MERGE_FILE
# touch $MERGE_FILE

# for SRC_FILE in `ls $DEST/data/bart_data/cnn_dm/$SUB.source.shard.*.hypo.$alpha`
# do
#     cat $SRC_FILE >> $MERGE_FILE
# done

# # export CLASSPATH=$DEST/corenlp/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar
# # # Tokenize hypothesis and target files.
cat $MERGE_FILE | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $MERGE_FILE.tokenized
cat $DEST/data/bart_data/cnn_dm/test.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $DEST/data/bart_data/cnn_dm/test.target.tokenized
files2rouge $DEST/data/bart_data/cnn_dm/test.target.tokenized $MERGE_FILE.tokenized
# Expected output: (ROUGE-2 Average_F: 0.21238) 