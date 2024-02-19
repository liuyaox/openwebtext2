WORKSPACE=/dfs/comicai/yao.liu/dataset/pretrain/Intermediate_data_merged


#python -m cleaning.generate_minhashes \
#    -p 112 \
#    -d "$WORKSPACE/fanfiction/fanfiction_pqrs/*chunk*.jsonl" \
#    -o $WORKSPACE/fanfiction/minhashes_fanfiction_pqrs.pkl \
#    -l $WORKSPACE/fanfiction/minhashes_fanfiction_pqrs.log
#
#
#python -m cleaning.generate_minhashes \
#    -p 112 \
#    -d "$WORKSPACE/fanfiction/fanfiction_tuv/*chunk*.jsonl" \
#    -o $WORKSPACE/fanfiction/minhashes_fanfiction_tuv.pkl \
#    -l $WORKSPACE/fanfiction/minhashes_fanfiction_tuv.log
#
#
#python -m cleaning.generate_minhashes \
#    -p 112 \
#    -d "$WORKSPACE/fanfiction/fanfiction_wxyz/*chunk*.jsonl" \
#    -o $WORKSPACE/fanfiction/minhashes_fanfiction_wxyz.pkl \
#    -l $WORKSPACE/fanfiction/minhashes_fanfiction_wxyz.log
#
#
#python -m cleaning.generate_minhashes \
#    -p 112 \
#    -d "$WORKSPACE/skypile_books/part1_1/*chunk*.jsonl" \
#    -o $WORKSPACE/skypile_books/minhashes_part1_1.pkl \
#    -l $WORKSPACE/skypile_books/minhashes_part1_1.log
#
#
#python -m cleaning.generate_minhashes \
#    -p 112 \
#    -d "$WORKSPACE/skypile_books/part1_2/*chunk*.jsonl" \
#    -o $WORKSPACE/skypile_books/minhashes_part1_2.pkl \
#    -l $WORKSPACE/skypile_books/minhashes_part1_2.log
#
#
#python -m cleaning.generate_minhashes \
#    -p 112 \
#    -d "$WORKSPACE/skypile_books/part1_3/*chunk*.jsonl" \
#    -o $WORKSPACE/skypile_books/minhashes_part1_3.pkl \
#    -l $WORKSPACE/skypile_books/minhashes_part1_3.log
#
#
#python -m cleaning.generate_minhashes \
#    -p 112 \
#    -d "$WORKSPACE/skypile_books/part1_4/*chunk*.jsonl" \
#    -o $WORKSPACE/skypile_books/minhashes_part1_4.pkl \
#    -l $WORKSPACE/skypile_books/minhashes_part1_4.log
#
#
#python -m cleaning.generate_minhashes \
#    -p 112 \
#    -d "$WORKSPACE/skypile_books/part2_1/*chunk*.jsonl" \
#    -o $WORKSPACE/skypile_books/minhashes_part2_1.pkl \
#    -l $WORKSPACE/skypile_books/minhashes_part2_1.log
#
#
#python -m cleaning.generate_minhashes \
#    -p 112 \
#    -d "$WORKSPACE/skypile_books/part2_2/*chunk*.jsonl" \
#    -o $WORKSPACE/skypile_books/minhashes_part2_2.pkl \
#    -l $WORKSPACE/skypile_books/minhashes_part2_2.log
#
#
#python -m cleaning.generate_minhashes \
#    -p 112 \
#    -d "$WORKSPACE/skypile_books/part2_3/*chunk*.jsonl" \
#    -o $WORKSPACE/skypile_books/minhashes_part2_3.pkl \
#    -l $WORKSPACE/skypile_books/minhashes_part2_3.log
#
#
#python -m cleaning.generate_minhashes \
#    -p 112 \
#    -d "$WORKSPACE/fanfiction/fanfiction_mno/*chunk*.jsonl" \
#    -o $WORKSPACE/fanfiction/minhashes_fanfiction_mno.pkl \
#    -l $WORKSPACE/fanfiction/minhashes_fanfiction_mno.log
#
#
#python -m cleaning.minhash_lsh_batching \
#  -m "$WORKSPACE/*/minhashes_*.pkl" \
#  -d $WORKSPACE/batch_minhashes \
#  -b 100
#
#
#python -m cleaning.minhash_lsh_dedupe \
#  -p 112 \
#  -d $WORKSPACE/batch_minhashes


python -m cleaning.dedupe_from_indexes \
  -p 112 \
  -d $WORKSPACE/batch_minhashes

