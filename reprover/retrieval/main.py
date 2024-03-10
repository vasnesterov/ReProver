"""Script for training the premise retriever.
"""

import os

from loguru import logger
from pytorch_lightning.cli import LightningCLI
from reprover.retrieval.datamodule import RetrievalDataModule
from reprover.retrieval.model import PremiseRetriever


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        # parser.link_arguments("model.context_model.model_name", "data.model_name")
        parser.link_arguments("data.max_seq_len", "model.max_seq_len")


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(ColbertPremiseRetriever, RetrievalDataModule)
    logger.info("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
