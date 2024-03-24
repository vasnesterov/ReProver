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
  --pretrained_checkpoint 'FacebookAI/roberta-base' \
  --colbert.use_ib_negatives true \
  --colbert.query_maxlen 512 \
  --colbert.doc_maxlen 512 \
  --colbert.lr 1e-4 \
  --colbert.warmup 2000 \
  --colbert.bsize 32 \
  --colbert.drop_duplciate_passages true \
  --logger.save_dir ./experiments/colbert-v1 \
  --logger.project lean-colbert
