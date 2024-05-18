"""Script for training the premise retriever.
"""

import os
from dataclasses import fields

from colbert.infra.config import ColBERTConfig
from loguru import logger
from pytorch_lightning.cli import LightningCLI
from reprover.retrieval.colbert_retrieval.datamodule import ColBERTMSMARCODataModule
from reprover.retrieval.colbert_retrieval.model import ColBERTPremiseRetrieverLightning


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:

        cfg = ColBERTConfig()
        for f in fields(cfg):
            if f.name == "nway":
                parser.link_arguments(
                    "model.config.num_negatives", "model.config.nway", compute_fn=lambda source: source + 1
                )
                parser.link_arguments(
                    "model.config.num_negatives", "data.config.nway", compute_fn=lambda source: source + 1
                )
                parser.link_arguments(
                    "model.config.num_negatives",
                    "data.config.num_negatives",
                )
            if f.name not in ("num_negatives", "nway"):
                parser.link_arguments(
                    f"model.config.{f.name}",
                    f"data.config.{f.name}",
                )


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(ColBERTPremiseRetrieverLightning, ColBERTMSMARCODataModule, save_config_callback=None)
    logger.info("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
