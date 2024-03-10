"""Ligihtning module for the premise retriever."""

import math
import os
import pickle
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from lean_dojo import Pos
from loguru import logger
from reprover.common import Corpus, Premise, get_optimizers, load_checkpoint, zip_strict
from transformers import AutoTokenizer, T5EncoderModel

torch.set_float32_matmul_precision("medium")


class BasePremiseRetriever(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        max_seq_len: int,
        num_retrieved: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_retrieved = num_retrieved
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.embeddings_staled = True

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool) -> "BasePremiseRetriever":
        return load_checkpoint(cls, ckpt_path, device, freeze)

    def load_corpus(self, path_or_corpus: Union[str, Corpus]) -> None:
        """Associate the retriever with a corpus."""
        raise NotImplementedError

    @property
    def embedding_size(self) -> int:
        """Return the size of the feature vector produced by ``encoder``."""
        return self.encoder.config.hidden_size

    def _encode(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.FloatTensor:
        """Encode a premise or a context into a feature vector."""
        raise NotImplementedError

    def forward(
        self,
        context_ids: torch.LongTensor,
        context_mask: torch.LongTensor,
        pos_premise_ids: torch.LongTensor,
        pos_premise_mask: torch.LongTensor,
        neg_premises_ids: torch.LongTensor,
        neg_premises_mask: torch.LongTensor,
        label: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Compute the contrastive loss for premise retrieval."""
        raise NotImplementedError

    ############
    # Training #
    ############

    def on_fit_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
            logger.info(f"Logging to {self.trainer.log_dir}")

        self.corpus = self.trainer.datamodule.corpus
        self.corpus_embeddings = None
        self.embeddings_staled = True

    def training_step(self, batch: Dict[str, Any], _) -> torch.Tensor:
        loss = self(
            batch["context_ids"],
            batch["context_mask"],
            batch["pos_premise_ids"],
            batch["pos_premise_mask"],
            batch["neg_premises_ids"],
            batch["neg_premises_mask"],
            batch["label"],
        )
        self.log("loss_train", loss, on_epoch=True, sync_dist=True, batch_size=len(batch))
        return loss

    def on_train_batch_end(self, outputs, batch, _) -> None:
        """Mark the embeddings as staled after a training batch."""
        self.embeddings_staled = True

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(self.parameters(), self.trainer, self.lr, self.warmup_steps)

    ##############
    # Validation #
    ##############

    @torch.no_grad()
    def reindex_corpus(self, batch_size: int) -> None:
        """Re-index the retrieval corpus using the up-to-date encoder."""
        raise NotImplementedError

    def retrieve_from_preprocessed(self, batch: Dict[str, Any]) -> Tuple[List[Premise], List[float]]:
        """Retrieve premises and scores from a tokenized batch"""
        raise NotImplementedError

    def on_validation_start(self) -> None:
        self.reindex_corpus(self.trainer.datamodule.eval_batch_size)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Retrieve premises and calculate metrics such as Recall@K and MRR."""
        # Retrieval.
        retrieved_premises, _ = self.retrieve_from_preprocessed(batch)

        # Evaluation & logging.
        recall = [[] for _ in range(self.num_retrieved)]
        MRR = []
        num_with_premises = 0
        tb = self.logger.experiment

        for i, (all_pos_premises, premises) in enumerate(zip_strict(batch["all_pos_premises"], retrieved_premises)):
            # Only log the first example in the batch.
            if i == 0:
                msg_gt = "\n\n".join([p.serialize() for p in all_pos_premises])
                msg_retrieved = "\n\n".join([f"{j}. {p.serialize()}" for j, p in enumerate(premises)])
                TP = len(set(premises).intersection(all_pos_premises))
                if len(all_pos_premises) == 0:
                    r = math.nan
                else:
                    r = float(TP) / len(all_pos_premises)
                msg = f"Recall@{self.num_retrieved}: {r}\n\nGround truth:\n\n```\n{msg_gt}\n```\n\nRetrieved:\n\n```\n{msg_retrieved}\n```"
                tb.add_text(f"premises_val", msg, self.global_step)

            all_pos_premises = set(all_pos_premises)
            if len(all_pos_premises) == 0:
                continue
            else:
                num_with_premises += 1
            first_match_found = False

            for j in range(self.num_retrieved):
                TP = len(all_pos_premises.intersection(premises[: (j + 1)]))
                recall[j].append(float(TP) / len(all_pos_premises))
                if premises[j] in all_pos_premises and not first_match_found:
                    MRR.append(1.0 / (j + 1))
                    first_match_found = True
            if not first_match_found:
                MRR.append(0.0)

        recall = [100 * np.mean(_) for _ in recall]

        for j in range(self.num_retrieved):
            self.log(
                f"Recall@{j+1}_val",
                recall[j],
                on_epoch=True,
                sync_dist=True,
                batch_size=num_with_premises,
            )

        self.log(
            "MRR",
            np.mean(MRR),
            on_epoch=True,
            sync_dist=True,
            batch_size=num_with_premises,
        )

    ##############
    # Prediction #
    ##############

    def on_predict_start(self) -> None:
        self.corpus = self.trainer.datamodule.corpus
        self.corpus_embeddings = None
        self.embeddings_staled = True
        self.reindex_corpus(self.trainer.datamodule.eval_batch_size)
        self.predict_step_outputs = []

    def predict_step(self, batch: Dict[str, Any], _):
        retrieved_premises, scores = self.retrieve_from_preprocessed(batch)

        for (
            url,
            commit,
            file_path,
            full_name,
            start,
            tactic_idx,
            ctx,
            pos_premises,
            premises,
            s,
        ) in zip_strict(
            batch["url"],
            batch["commit"],
            batch["file_path"],
            batch["full_name"],
            batch["start"],
            batch["tactic_idx"],
            batch["context"],
            batch["all_pos_premises"],
            retrieved_premises,
            scores,
        ):
            self.predict_step_outputs.append(
                {
                    "url": url,
                    "commit": commit,
                    "file_path": file_path,
                    "full_name": full_name,
                    "start": start,
                    "tactic_idx": tactic_idx,
                    "context": ctx,
                    "all_pos_premises": pos_premises,
                    "retrieved_premises": premises,
                    "scores": s,
                }
            )

    def on_predict_epoch_end(self) -> None:
        if self.trainer.log_dir is not None:
            path = os.path.join(self.trainer.log_dir, "predictions.pickle")
            with open(path, "wb") as oup:
                pickle.dump(self.predict_step_outputs, oup)
            logger.info(f"Retrieval predictions saved to {path}")

        self.predict_step_outputs.clear()

    def retrieve(
        self,
        state: List[str],
        file_name: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        k: int,
    ) -> Tuple[List[Premise], List[float]]:
        """Retrieve ``k`` premises from ``corpus`` using ``state`` and ``tactic_prefix`` as context."""
        raise NotImplementedError
