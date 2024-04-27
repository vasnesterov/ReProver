python reprover/retrieval/main.py fit \
  --config reprover/retrieval/confs/cli_lean4_random_colbert.yaml \
  --data.batch_size 4 \
  --data.num_workers 0 \
  --model.checkpoint_path_or_name "google/byt5-small" \
  --model.config.nway 4
