"""Script for training the premise retriever.
"""

import os
from dataclasses import fields

from colbert.infra.config import ColBERTConfig
from loguru import logger
from pytorch_lightning.cli import LightningCLI
from reprover.retrieval.colbert_retrieval.datamodule import ColBERTDataModule
from reprover.retrieval.colbert_retrieval.model import \
    ColBERTPremiseRetrieverLightning


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("model.checkpoint_path_or_name", "model.config.checkpoint", apply_on="instantiate")
        parser.link_arguments("model.checkpoint_path_or_name", "data.config.checkpoint", apply_on="instantiate")

        cfg = ColBERTConfig()
        for f in fields(cfg):
            if f.name != 'checkpoint':
                parser.link_arguments(
                    f"model.config.{f.name}",
                    f"data.config.{f.name}",
                )


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(ColBERTPremiseRetrieverLightning, ColBERTDataModule, save_config_callback=None)
    logger.info("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
