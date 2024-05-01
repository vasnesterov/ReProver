## PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION is needed to make protobuf work
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python reprover/retrieval/main.py fit \
  --config reprover/retrieval/confs/cli_lean4_random_colbert_native.yaml \
  --data.batch_size 4 \
  --data.num_workers 0 \
  --model.config.checkpoint microsoft/deberta-v3-xsmall \
  --model.config.debug false
