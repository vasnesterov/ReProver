python scripts/train_colbert_v1.py \
  --data.collection_path ./data/leandojo_benchmark_4/random/colbert_collection.tsv \
  --data.queries_path ./data/leandojo_benchmark_4/random/colbert_queries.json \
  --data.triples_path ./data/leandojo_benchmark_4/random/colbert_triples.json \
  --experiment_root ./experiments/colbert-v1/ \
  \
  --run.overwrite true \
  --run.experiment 'lean' \
  --run.nranks 1 \
  --run.gpus 1 \
  \
  --colbert.bsize 1
