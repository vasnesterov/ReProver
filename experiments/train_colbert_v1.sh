python scripts/train_colbert_v1.py \
  --premises_paths_list "[./data/leandojo_benchmark_4/random/train.json, \
                          ./data/leandojo_benchmark_4/random/val.json, \
                          ./data/leandojo_benchmark_4/random/test.json]" \
  --corpus_path ./data/leandojo_benchmark_4/corpus.jsonl \
  --collection_path ./data/leandojo_benchmark_4/random/colbert_collection.tsv \
  --queries_path ./data/leandojo_benchmark_4/random/colbert_queries.json \
  --triples_path ./data/leandojo_benchmark_4/random/colbert_triples.json \
  --experiment_root ./experiments/colbert-v1/
