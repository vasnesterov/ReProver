"""Ligihtning module for the premise retriever."""

import os
import math
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Union
from transformers import T5EncoderModel, AutoTokenizer

from common import get_optimizers

torch.set_float32_matmul_precision("medium")


class Ranker(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.oup = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, 1),
            torch.nn.Sigmoid(),
        )

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool) -> "Ranker":
        return load_checkpoint(cls, ckpt_path, device, freeze)

    @property
    def embedding_size(self) -> int:
        """Return the size of the feature vector produced by ``encoder``."""
        return self.encoder.config.hidden_size

    def rank(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        """Encode a premise or a context into a feature vector."""
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state

        # Masked average.
        lens = attention_mask.sum(dim=1)
        features = (hidden_states * attention_mask.unsqueeze(2)).sum(
            dim=1
        ) / lens.unsqueeze(1)

        # Normalize the feature vector to have unit norm.
        # features = F.normalize(features, dim=1)
        return self.oup(features)

    def forward(
        self,
        state_ids: torch.LongTensor,
        state_mask: torch.LongTensor,
        target: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute the contrastive loss for premise retrieval."""
        score = self.rank(state_ids, state_mask)
        loss = F.mse_loss(score, target)
        return loss

    ############
    # Training #
    ############

    def on_fit_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
            logger.info(f"Logging to {self.trainer.log_dir}")

    def training_step(self, batch: Dict[str, Any], _) -> torch.Tensor:
        loss = self(
            batch["state_ids"],
            batch["state_mask"],
            batch["target"],
        )
        self.log(
            "loss_train", loss, on_epoch=True, sync_dist=True, batch_size=len(batch)
        )
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )

    ##############
    # Validation #
    ##############

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Retrieve premises and calculate metrics such as Recall@K and MRR."""
        loss = self(
            batch["state_ids"],
            batch["state_mask"],
            batch["target"],
        )
        self.log(
            "loss_val", loss, on_epoch=True, sync_dist=True, batch_size=len(batch)
        )