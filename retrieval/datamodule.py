"""Datamodule for the premise retrieval."""

import os
import json
import torch
import random
import jsonlines
import itertools
from tqdm import tqdm
from loguru import logger
from copy import deepcopy
from lean_dojo import Pos
import pytorch_lightning as pl
from lean_dojo import LeanGitRepo
from typing import Optional, List
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader


from common import Premise, Context, Corpus, Batch, Example, get_all_pos_premises, load_available_premises, zip_strict, path_to_module


class RetrievalDataset(Dataset):
    def __init__(
        self,
        data_paths: List[str],
        available_premises_path: str,
        uses_lean4: bool,
        corpus: Corpus,
        num_negatives: int,
        num_in_file_negatives: int,
        max_seq_len: int,
        context_tokenizer,
        premise_tokenizer,
        is_train: bool,
    ) -> None:
        super().__init__()
        self.corpus = corpus
        self.num_negatives = num_negatives
        self.num_in_file_negatives = num_in_file_negatives
        self.max_seq_len = max_seq_len
        self.context_tokenizer = context_tokenizer
        self.premise_tokenizer = premise_tokenizer
        self.is_train = is_train
        self.available_premises = corpus.in_file_premises
        self.data = list(
            itertools.chain.from_iterable(self._load_data(path) for path in data_paths)
        )

    def _load_data(self, data_path: str) -> List[Example]:
        data = []
        logger.info(f"Loading data from {data_path}")

        data = json.load(open(data_path))
        if self.is_train:
            data = list(filter(lambda x: len(x['premises']) > 0, data))

        for row in data:
            row['all_pos_premises'] = [Premise.from_dict(dct) for dct in row['premises']]
            del row["premises"]
        

        logger.info(f"Loaded {len(data)} examples.")
        return data

    # def _load_available_premises(self, available_premises_path: str):
    #     return json.load(open(available_premises_path))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        ex = deepcopy(self.data[idx])

        ex["context"] = Context(path_to_module(ex["file_path"]), ex["full_name"], ex["state"])

        if self.is_train:
            ex["pos_premise"] = random.choice(ex["all_pos_premises"])
            num_in_file_negatives = min(self.num_in_file_negatives, len(self.available_premises[ex["full_name"]]["inFilePremises"]))
            ex["neg_premises"] = random.sample(
                    self.available_premises[ex["full_name"]]["inFilePremises"], num_in_file_negatives
                ) + random.sample(
                    self.available_premises[ex["full_name"]]["outFilePremises"], self.num_negatives - num_in_file_negatives
                )

        return ex

    def collate(self, examples: List[Example]) -> Batch:
        batch = {}

        # Tokenize the context.
        context = [ex["context"] for ex in examples]
        tokenized_context = self.context_tokenizer(
            [c.serialize() for c in context],
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        batch["context"] = context
        batch["context_ids"] = tokenized_context.input_ids
        batch["context_mask"] = tokenized_context.attention_mask

        # Tokenize the label and premises.
        if self.is_train:
            pos_premise = [ex["pos_premise"] for ex in examples]
            tokenized_pos_premise = self.premise_tokenizer(
                [p.serialize() for p in pos_premise],
                padding="longest",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            )
            batch["pos_premise"] = pos_premise
            batch["pos_premise_ids"] = tokenized_pos_premise.input_ids
            batch["pos_premise_mask"] = tokenized_pos_premise.attention_mask

            batch_size = len(examples)
            label = torch.zeros(batch_size, batch_size * (1 + self.num_negatives))

            # Check if one's negative is another's positive
            for j in range(batch_size):
                all_pos_premises = examples[j]["all_pos_premises"]
                for k in range(batch_size * (1 + self.num_negatives)):
                    if k < batch_size:
                        pos_premise_k = examples[k]["pos_premise"]
                    else:
                        pos_premise_k = examples[k % batch_size]["neg_premises"][
                            k // batch_size - 1
                        ]
                    label[j, k] = float(pos_premise_k in all_pos_premises)

            batch["label"] = label
            batch["neg_premises"] = []
            batch["neg_premises_ids"] = []
            batch["neg_premises_mask"] = []

            for i in range(self.num_negatives):
                neg_premise = [ex["neg_premises"][i] for ex in examples]
                tokenized_neg_premise = self.premise_tokenizer(
                    [p.serialize() for p in neg_premise],
                    padding="longest",
                    max_length=self.max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                )
                batch["neg_premises"].append(neg_premise)
                batch["neg_premises_ids"].append(tokenized_neg_premise.input_ids)
                batch["neg_premises_mask"].append(tokenized_neg_premise.attention_mask)

        # Copy the rest of the fields.
        for k in examples[0].keys():
            if k not in batch:
                batch[k] = [ex[k] for ex in examples]

        return batch


class RetrievalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        corpus_path: str,
        num_negatives: int,
        num_in_file_negatives: int,
        context_tokenizer_name: str,
        premise_tokenizer_name: str,
        batch_size: int,
        eval_batch_size: int,
        max_seq_len: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.num_negatives = num_negatives
        assert 0 <= num_in_file_negatives <= num_negatives
        self.num_in_file_negatives = num_in_file_negatives
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers

        self.context_tokenizer = AutoTokenizer.from_pretrained(context_tokenizer_name)
        self.premise_tokenizer = AutoTokenizer.from_pretrained(premise_tokenizer_name)
        if self.context_tokenizer.pad_token is None:
            self.context_tokenizer.pad_token = self.context_tokenizer.unk_token
        if self.premise_tokenizer.pad_token is None:
            self.premise_tokenizer.pad_token = self.premise_tokenizer.unk_token
        self.corpus = Corpus(
            jsonl_path=os.path.join(corpus_path, "corpus.jsonl"),
            dot_imports_path=os.path.join(corpus_path, "import_graph.dot"),
            json_in_file_premises_path=os.path.join(corpus_path, "available_premises.json") ## FIXME
        )

        # metadata = json.load(open(os.path.join(data_path, "../metadata.json")))
        # repo = LeanGitRepo(**metadata["from_repo"])
        self.uses_lean4 = True

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.ds_train = RetrievalDataset(
                [os.path.join(self.data_path, "train.json")],
                os.path.join(self.data_path, "available_premises.json"),
                self.uses_lean4,
                self.corpus,
                self.num_negatives,
                self.num_in_file_negatives,
                self.max_seq_len,
                self.context_tokenizer,
                self.premise_tokenizer,
                is_train=True,
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = RetrievalDataset(
                [os.path.join(self.data_path, "val.json")],
                os.path.join(self.data_path, "available_premises.json"),
                self.uses_lean4,
                self.corpus,
                self.num_negatives,
                self.num_in_file_negatives,
                self.max_seq_len,
                self.context_tokenizer,
                self.premise_tokenizer,
                is_train=False,
            )

        if stage in (None, "fit", "predict"):
            self.ds_pred = RetrievalDataset(
                [
                    os.path.join(self.data_path, f"{split}.json")
                    for split in ("train", "val", "test")
                ],
                os.path.join(self.data_path, "available_premises.json"),
                self.uses_lean4,
                self.corpus,
                self.num_negatives,
                self.num_in_file_negatives,
                self.max_seq_len,
                self.context_tokenizer,
                self.premise_tokenizer,
                is_train=False,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_pred,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_pred.collate,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
