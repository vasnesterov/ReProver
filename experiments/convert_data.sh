
python ./scripts/convert_data.py \
  --premises_paths_list "[./data/leandojo_benchmark_4/random/train.json, \
                          ./data/leandojo_benchmark_4/random/val.json, \
                          ./data/leandojo_benchmark_4/random/test.json]" \
  --corpus_path ./data/leandojo_benchmark_4/corpus.jsonl \
  --collection_save_path ./data/leandojo_benchmark_4/random/colbert_collection.tsv \
  --queries_save_path ./data/leandojo_benchmark_4/random/colbert_queries.json \
  --triples_save_path ./data/leandojo_benchmark_4/random/colbert_triples.json \
  --num_negatives 1 \
  --num_in_file_negatives 1



