
python ./scripts/convert_data.py \
  --premises_paths_list "[./data/leandojo_benchmark_4/novel_premises/train.json, \
                          ./data/leandojo_benchmark_4/novel_premises/val.json, \
                          ./data/leandojo_benchmark_4/novel_premises/test.json]" \
  --corpus_path ./data/leandojo_benchmark_4/corpus.jsonl \
  --collection_save_path ./data/leandojo_benchmark_4/novel_premises/colbert_collection.tsv \
  --queries_save_path ./data/leandojo_benchmark_4/novel_premises/colbert_queries.json \
  --triples_save_path ./data/leandojo_benchmark_4/novel_premises/colbert_triples.json \
  --num_negatives 10 \
  --num_in_file_negatives 1



