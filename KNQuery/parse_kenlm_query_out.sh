DEST=/mnt/task_wrapper/user_output/artifacts/

for DOMAIN in it koran law medical subtitles
do
    python parse_kenlm_query_out.py --query_out_path $DEST/kenlm/build/$DOMAIN.test.out
done