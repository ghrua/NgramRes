DEST=/project/nlp-work4/huayang-l/residual_with_ngram/
DATADIR=$DEST/data/
LANG=de
ARPA_FNAME=train.bpe32k.$LANG.prn0001.arpa
ARPA=$DATADIR/$DOMAIN/$ARPA_FNAME
CACHEDIR=./cache_5gram_wmt_$LANG/
# TKZ=$DEST/data-bin/wmt14.tokenized.de-en/dict.$LANG.txt
TKZ=$DEST/data-bin/wmt14.tokenized.de-en/dict.$LANG.txt

mkdir -p $CACHEDIR
python arpa2binary.py --tokenizer_path $TKZ \
                    --arpa $ARPA \
                    --eos_token "</s>" \
                    --unk_token "<unk>" \
                    --binary ./$ARPA_FNAME.pt

python query.py --tokenizer_path $TKZ \
                    --lm_path ./$ARPA_FNAME.pt \
                    --cache_path $CACHEDIR/$ARPA_FNAME.cache \
                    --mode fast
