DEST=/mnt/task_wrapper/user_output/artifacts/

###################
# Huggingface
###################

# python arpa2binary.py --tokenizer_path $DEST/huggingface/tokenizers/wikitext-103/space/space_tokenizer.json \
#                       --arpa $DEST/kenlm/build/wt103.kenlm.arpa \
#                       --binary ./wt103.kenlm.5gram.pt

###################
# fairseq
###################

# python arpa2binary.py --tokenizer_path $DEST/data-bin/wikitext-103/dict.txt \
#                       --arpa $DEST/kenlm/build/law.train.wt103vocab.arpa \
#                       --eos_token "</s>" \
#                       --unk_token "<unk>" \
#                       --binary ./law.train.wt103vocab.kenlm.5gram.arpa.fairseq.pt

# python arpa2binary.py --tokenizer_path $DEST/wmt19.en/dict.txt \
#                       --arpa $DEST/kenlm/build/newscrawl.wmt19.min2.repunk.arpa \
#                       --eos_token "</s>" \
#                       --unk_token "<unk>" \
#                       --binary ./newscrawl.wmt19.min2.repunk.arpa.pt

# mkdir -p ./cache_5gram_newscrawl_wmt19/
# python query.py --tokenizer_path $DEST/wmt19.en/dict.txt \
#                       --input_path ./tmp.txt \
#                       --lm_path ./newscrawl.wmt19.min2.repunk.arpa.pt \
#                       --cache_path ./cache_5gram_newscrawl_wmt19/newscrawl.wmt19.min2.repunk.arpa.cache \
#                       --mode fast

# python arpa2binary.py --tokenizer_path $DEST/wmt19.en/dict.txt \
#                       --arpa $DEST/kenlm/build/law.train.en.wmt19.arpa \
#                       --eos_token "</s>" \
#                       --unk_token "<unk>" \
#                       --binary ./law.train.en.wmt19.arpa.pt

# mkdir -p ./cache_5gram_law_wmt19/
# python query.py --tokenizer_path $DEST/wmt19.en/dict.txt \
#                       --input_path ./tmp.txt \
#                       --lm_path ./law.train.en.wmt19.arpa.pt \
#                       --cache_path ./cache_5gram_law_wmt19/law.train.en.wmt19.arpa.cache \
#                       --mode fast

# DATADIR=$DEST/data/wmt14-ende-stanford/
# FILEPREF=train.bpe32k

# DATADIR=$DEST/data/iwslt14_ende/iwslt14.tokenized.de-en
# FILEPREF=train

################
# IWSLT14
################

# for LANG in de en
# do
#     ARPA=$DATADIR/$FILEPREF.$LANG.5gram.arpa
#     CACHEDIR=./cache_5gram_iwslt14_ende
#     # TKZ=$DEST/data-bin/wmt14.tokenized.de-en/dict.$LANG.txt
#     TKZ=$DEST/data-bin/iwslt14.en-de/dict.$LANG.txt
    
#     mkdir -p $CACHEDIR
#     python arpa2binary.py --tokenizer_path $TKZ \
#                         --arpa $ARPA \
#                         --eos_token "</s>" \
#                         --unk_token "<unk>" \
#                         --binary ./$FILEPREF.$LANG.5gram.arpa.pt

    
#     python query.py --tokenizer_path $TKZ \
#                         --input_path ./tmp.txt \
#                         --lm_path ./$FILEPREF.$LANG.5gram.arpa.pt \
#                         --cache_path $CACHEDIR/$FILEPREF.$LANG.5gram.arpa.cache \
#                         --mode fast
# done

################
# SUM
################


DATADIR=$DEST/data/multi_domain_new_split_multitask_version/
LANG=de
for DOMAIN in it koran law medical subtitles
do
    ARPA_FNAME=train.$LANG.tok.bpe32k.5gram.arpa
    ARPA=$DATADIR/$DOMAIN/$ARPA_FNAME
    CACHEDIR=./cache_5gram_multitask_$DOMAIN/
    # TKZ=$DEST/data-bin/wmt14.tokenized.de-en/dict.$LANG.txt
    TKZ=$DEST/data-bin/multi-task-lm-en-bin/dict.txt
    
    mkdir -p $CACHEDIR
    python arpa2binary.py --tokenizer_path $TKZ \
                        --arpa $ARPA \
                        --eos_token "</s>" \
                        --unk_token "<unk>" \
                        --binary ./$ARPA_FNAME.pt

    python query.py --tokenizer_path $TKZ \
                        --input_path ./tmp.txt \
                        --lm_path ./$ARPA_FNAME.pt \
                        --cache_path $CACHEDIR/$ARPA_FNAME.cache \
                        --mode fast

 done
