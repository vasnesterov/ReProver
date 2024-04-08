num_negatives=64
num_in_file_negatives=32
corpus_path=./data/leandojo_benchmark_4/corpus.jsonl
collection_path=./data/leandojo_benchmark_4/random/collection.tsv


python ./scripts/convert_data.py \
  --premises_paths_list "[./data/leandojo_benchmark_4/random/train.json, ./data/leandojo_benchmark_4/random/val.json, ./data/leandojo_benchmark_4/random/test.json]" \
  --corpus_path $corpus_path \
  --collection_save_path $collection_path \
  --queries_save_path ./data/leandojo_benchmark_4/random/queries.json \
  --triples_save_path ./data/leandojo_benchmark_4/random/triples.json \
  --num_negatives $num_negatives \
  --num_in_file_negatives $num_in_file_negatives


python ./scripts/convert_data.py \
  --premises_paths_list "[./data/leandojo_benchmark_4/random/train.json]" \
  --corpus_path $corpus_path \
  --collection_save_path $collection_path \
  --queries_save_path ./data/leandojo_benchmark_4/random/queries_train.json \
  --triples_save_path ./data/leandojo_benchmark_4/random/triples_train.json \
  --num_negatives $num_negatives \
  --num_in_file_negatives $num_in_file_negatives


python ./scripts/convert_data.py \
  --premises_paths_list "[./data/leandojo_benchmark_4/random/val.json]" \
  --corpus_path $corpus_path \
  --collection_save_path $collection_path \
  --queries_save_path ./data/leandojo_benchmark_4/random/queries_val.json \
  --triples_save_path ./data/leandojo_benchmark_4/random/triples_val.json \
  --num_negatives $num_negatives \
  --num_in_file_negatives $num_in_file_negatives


  python ./scripts/convert_data.py \
  --premises_paths_list "[./data/leandojo_benchmark_4/random/test.json]" \
  --corpus_path $corpus_path \
  --collection_save_path $collection_path \
  --queries_save_path ./data/leandojo_benchmark_4/random/queries_test.json \
  --triples_save_path ./data/leandojo_benchmark_4/random/triples_test.json \
  --num_negatives $num_negatives \
  --num_in_file_negatives $num_in_file_negatives
