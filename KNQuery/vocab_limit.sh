DEST=/mnt/task_wrapper/user_output/artifacts/


python vocab_limit.py --input_path $DEST/data/iwslt15_envi/train.vi \
                      --tokenizer_path $DEST/data-bin/iwslt15.en-vi/dict.vi.txt \
                      --output_path $DEST/data/iwslt15_envi/train.repunk.vi

python vocab_limit.py --input_path $DEST/data/iwslt15_envi/train.en \
                      --tokenizer_path $DEST/data-bin/iwslt15.en-vi/dict.en.txt \
                      --output_path $DEST/data/iwslt15_envi/train.repunk.en