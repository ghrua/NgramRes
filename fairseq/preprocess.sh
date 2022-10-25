#!/usr/bin/env bash

DEST=/mnt/task_wrapper/user_output/artifacts/

########################################
# Multitask LM
########################################
# DEST=/mnt/task_wrapper/user_output/artifacts/
# TEXT=$DEST/data/multi_domain_new_split_multitask_version/
# LANG=en
# fairseq-preprocess \
#     --only-source \
#     --trainpref $TEXT/train.cat.$LANG.tok.bpe32k \
#     --validpref $TEXT/dev.cat.$LANG.tok.bpe32k \
#     --testpref $TEXT/test.cat.$LANG.tok.bpe32k \
#     --destdir $DEST/data-bin/multi-task-lm-$LANG-bin \
#     --workers 20

# LANG=en
# for DOMAIN in it koran law medical subtitles
# do
#     TEXT=$DEST/data/multi_domain_new_split_multitask_version/$DOMAIN
#     fairseq-preprocess \
#         --only-source \
#         --trainpref $TEXT/train.$LANG.tok.bpe32k \
#         --validpref $TEXT/dev.$LANG.tok.bpe32k \
#         --testpref $TEXT/test.$LANG.tok.bpe32k \
#         --destdir $DEST/data-bin/multi-task-lm-$LANG-$DOMAIN-bin \
#         --srcdict $DEST/data-bin/multi-task-lm-$LANG-bin/dict.txt \
#         --workers 20
# done


########################################
# Multitask MT
########################################

DEST=/mnt/task_wrapper/user_output/artifacts/
TEXT=$DEST/data/multi_domain_new_split_multitask_version/
L1=de
L2=en

for SUB in train dev test
do
    ln -s $TEXT/$SUB.cat.$L1.tok.bpe32k $TEXT/$SUB.cat.tok.bpe32k.$L1
    ln -s $TEXT/$SUB.cat.$L2.tok.bpe32k $TEXT/$SUB.cat.tok.bpe32k.$L2
done

fairseq-preprocess --source-lang $L1 --target-lang $L2 \
                   --trainpref $TEXT/train.cat.tok.bpe32k --validpref $TEXT/dev.cat.tok.bpe32k --testpref $TEXT/test.cat.tok.bpe32k \
                   --destdir  $DEST/data-bin/multi-task-mt-${L1}${L2}-bin \
                   --workers 32

for DOMAIN in it koran law medical subtitles
do
    TEXT=$DEST/data/multi_domain_new_split_multitask_version/$DOMAIN
    for SUB in train dev test
    do
        ln -s $TEXT/$SUB.$L1.tok.bpe32k $TEXT/$SUB.tok.bpe32k.$L1
        ln -s $TEXT/$SUB.$L2.tok.bpe32k $TEXT/$SUB.tok.bpe32k.$L2
    done

    fairseq-preprocess \
        --source-lang $L1 --target-lang $L2 \
        --trainpref $TEXT/train.tok.bpe32k \
        --validpref $TEXT/dev.tok.bpe32k \
        --testpref $TEXT/test.tok.bpe32k \
        --destdir $DEST/data-bin/multi-task-mt-${L1}${L2}-$DOMAIN-bin \
        --srcdict $DEST/data-bin/multi-task-mt-${L1}${L2}-bin/dict.$L1.txt \
        --tgtdict $DEST/data-bin/multi-task-mt-${L1}${L2}-bin/dict.$L2.txt \
        --workers 32
done

########################################
# wmt14 En-De
########################################

# TEXT=/mnt/task_wrapper/user_output/artifacts/data/wmt14-ende-stanford/

# python preprocess.py --source-lang en --target-lang de --joined-dictionary \
#                      --trainpref $TEXT/train.bpe32k --validpref $TEXT/newstest2013.bpe32k --testpref $TEXT/newstest2014.bpe32k \
#                      --destdir $DEST/data-bin/wmt14.tokenized.de-en \
#                      --workers 32

########################################
# IWSLT En-Vi
########################################

# TEXT=/mnt/task_wrapper/user_output/artifacts/data/iwslt15_envi/
# L1=vi
# L2=en
# rm -rf $DEST/data-bin/iwslt15.$L1-$L2

# python preprocess.py --source-lang $L1 --target-lang $L2 \
#                      --trainpref $TEXT/train --validpref $TEXT/tst2012 --testpref $TEXT/tst2013 \
#                      --destdir $DEST/data-bin/iwslt15.$L1-$L2 \
#                      --thresholdsrc 5 --thresholdtgt 5 \
#                      --workers 32

########################################
# IWSLT En-De
########################################

# TEXT=/mnt/task_wrapper/user_output/artifacts/data/iwslt14_ende/iwslt14.tokenized.de-en
# L1=de
# L2=en

# DATADIR=$DEST/data-bin/iwslt14.$L1-$L2
# rm -rf $DATADIR

# python preprocess.py --source-lang $L1 --target-lang $L2 \
#                      --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#                      --destdir $DATADIR \
#                      --workers 32

########################################
# wikitext-103
########################################
# DEST=/mnt/task_wrapper/user_output/artifacts/
# TEXT=$DEST/wikitext-103/

# fairseq-preprocess \
#     --only-source \
#     --trainpref $TEXT/wiki.train.tokens \
#     --validpref $TEXT/wiki.valid.tokens \
#     --testpref $TEXT/wiki.test.tokens \
#     --destdir $DEST/data-bin/wikitext-103 \
#     --workers 20
    #--dataset-impl raw \


# TEXT=$DEST/data/multi_domain_new_split
# fairseq-preprocess \
#     --only-source \
#     --trainpref $TEXT/law/train.en.tk.txt \
#     --validpref $TEXT/law/dev.en.tk.txt \
#     --testpref $TEXT/law/test.en.tk.txt \
#     --destdir $DEST/data-bin/multi_domain_law/ \
#     --srcdict $DEST/data-bin/wikitext-103/dict.txt \
#     --workers 20

# TEXT=$DEST/data/newscrawl_wmt19/split_fixbug
# rm -rf $DEST/data-bin/newscrawl.wmt19.bpe.fixbug
# fairseq-preprocess \
#     --only-source \
#     --trainpref $TEXT/train.txt \
#     --validpref $TEXT/valid.txt \
#     --testpref $TEXT/test.txt \
#     --destdir $DEST/data-bin/newscrawl.wmt19.bpe.fixbug \
#     --srcdict $DEST/wmt19.en/dict.txt \
#     --workers 20

# TEXT=$DEST/data/newscrawl_wmt19/multi_domain/law
# rm -rf $DEST/data-bin/newscrawl.wmt19.bpe
# fairseq-preprocess \
#     --only-source \
#     --trainpref $TEXT/train.en.norm.print.tok.bpe \
#     --validpref $TEXT/dev.en.norm.print.tok.bpe \
#     --testpref $TEXT/test.en.norm.print.tok.bpe \
#     --destdir $DEST/data-bin/multidomain.law.wmt19.bpe \
#     --srcdict $DEST/wmt19.en/dict.txt \
#     --workers 20
