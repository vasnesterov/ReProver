# resulting index dir will be index_root/experiment_name/index_name
python ./scripts/colbert_index.py \
  --checkpoint_path ./checkpoints/colbertv2.0/ \
  --collection_path ./data/leandojo_benchmark_4/random/collection.tsv \
  --index_root ./experiments/ \
  --experiment_name lean \
  --index_name colbert_v1 \
  --index_bsize 64 \
  --overwrite true
