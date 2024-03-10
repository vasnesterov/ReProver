"""Script for training the tactic generator."""

import os

from loguru import logger
from pytorch_lightning.cli import LightningCLI
from reprover.generator.datamodule import MultipleSegmentGeneratorDataModule
from reprover.generator.model import RMTRetrievalAugmentedGenerator


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("model.backbone_model_name", "data.model_name")
        parser.link_arguments("model.max_nonmemory_seq_len", "data.max_seq_len", apply_on="instantiate")
        parser.link_arguments("data.num_segments", "model.num_segments")


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(RMTRetrievalAugmentedGenerator, MultipleSegmentGeneratorDataModule, save_config_callback=None)
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
