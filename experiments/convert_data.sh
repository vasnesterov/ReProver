num_negatives=64
num_in_file_negatives=32
corpus_path=./data/leandojo_benchmark_4/corpus.jsonl
premises_dir=./data/leandojo_benchmark_4/random/
save_dir=./data/leandojo_benchmark_4/random_debug/


python ./scripts/convert_data.py \
  --premises_dir $premises_dir \
  --corpus_path $corpus_path \
  --save_dir $save_dir \
  --num_negatives $num_negatives \
  --num_in_file_negatives $num_in_file_negatives \
  --limit_collection 10000 \
  --limit_triples 1000
