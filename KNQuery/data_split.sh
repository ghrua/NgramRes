DEST=/mnt/task_wrapper/user_output/artifacts/

IN_DIR=$DEST/data/iwslt14_ende/iwslt14.tokenized.de-en/
IN_FILE=$IN_DIR/train.de
echo $IN_FILE
python data_split.py --input_path $IN_FILE \
                     --nsplit 4 \
                     --data2clst_map_path $IN_DIR/data2clst_4shard_map.pt