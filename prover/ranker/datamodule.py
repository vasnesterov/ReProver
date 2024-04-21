"""Data module for the ranker."""

import os
import json
import pickle
import random
import torch
from tqdm import tqdm
from loguru import logger
import pytorch_lightning as pl
from typing import Optional, List, Dict, Any
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ByT5Tokenizer

from prover.search_tree import InternalNode, ProofFinishedNode
from prover.proof_search_fast import SearchResult


class RankerDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        max_seq_len: int,
        tokenizer
    ) -> None:
        super().__init__()
        self.json_path = json_path
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.data = self._load_data()
    
    def _load_data(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        logger.info(f"{len(data)} examples loaded")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        ex = self.data[idx]
        return ex

    def collate(self, examples):
        state = [ex["state"] for ex in examples]
        target = torch.tensor([ex["target"] for ex in examples])
        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )

        batch = {}
        batch["state"] = state
        batch["state_ids"] = tokenized_state.input_ids
        batch["state_mask"] = tokenized_state.attention_mask
        batch["target"] = target

        return batch


class RankerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        model_name: str,
        batch_size: int,
        eval_batch_size: int,
        max_seq_len: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = RankerDataset(
                os.path.join(self.data_path, "train.json"),
                self.max_seq_len,
                self.tokenizer,
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = RankerDataset(
                os.path.join(self.data_path, "val.json"),
                self.max_seq_len,
                self.tokenizer,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )