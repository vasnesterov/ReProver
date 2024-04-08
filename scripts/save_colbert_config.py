from colbert.infra.config import ColBERTConfig
from jsonargparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument(
    "--config", type=ColBERTConfig, default=ColBERTConfig.load_from_checkpoint("./checkpoints/colbertv2.0/")
)

args = parser.parse_args()

parser.save(args, "./reprover/retrieval/confs/colbert_config.yaml", overwrite=True)
