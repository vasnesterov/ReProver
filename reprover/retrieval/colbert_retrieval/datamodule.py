from functools import partial
from itertools import chain
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from colbert.data.collection import Collection
from colbert.data.examples import Examples
from colbert.data.queries import Queries
from colbert.infra.config import ColBERTConfig
from colbert.infra.config.config import ColBERTConfig
from colbert.modeling.tokenization import (DocTokenizer, QueryTokenizer,
                                           tensorize_triples)
from colbert.utils.utils import zipstar
from torch.utils.data import DataLoader, Dataset


class ColBERTDataset(Dataset):
    def __init__(self, config: ColBERTConfig, triples: Path | str, queries: Path | str, collection: Path | str):
        self.nway = config.nway

        self.query_tokenizer = QueryTokenizer(config)
        self.doc_tokenizer = DocTokenizer(config)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)

        self.triples = Examples.cast(str(triples), nway=self.nway).tolist()
        self.queries = Queries.cast(str(queries))
        self.collection = Collection.cast(str(collection))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, i):
        query, *pids = self.triples[i]
        pids = pids[: self.nway]

        query = self.queries[query]

        try:
            pids, scores = zipstar(pids)
        except:
            scores = []

        passages = [self.collection[pid] for pid in pids]

        return query, passages, scores

    def collate(self, inputs):
        queries, passages, scores = zip(*inputs)

        positive_passages = [[p[0]] for p in passages]
        bsize = len(queries)
        passages = list(chain(*passages))

        Q, D, s = self.tensorize_triples(queries, passages, scores, bsize, self.nway)[0]
        batch = {"queries": Q, "passages": D, "scores": s, "positive_passages": positive_passages}
        return batch


class ColBERTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: ColBERTConfig,
        data_path: str,
        batch_size: int,
        eval_batch_size: int,
        max_seq_len: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.config.configure(bsize=batch_size, query_maxlen=max_seq_len, doc_maxlen=max_seq_len)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        collection = self.data_path / "collection.tsv"
        if stage in (None, "fit", "validate"):
            self.ds_train = ColBERTDataset(
                self.config, self.data_path / "triples_train.json", self.data_path / "queries_train.json", collection
            )

        if stage in (None, "fit", "validate"):
            self.ds_val = ColBERTDataset(
                self.config, self.data_path / "triples_val.json", self.data_path / "queries_val.json", collection
            )

        if stage in (None, "fit", "predict"):
            self.ds_pred = ColBERTDataset(
                self.config, self.data_path / "triples.json", self.data_path / "queries.json", collection
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            shuffle=True,
            # pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            shuffle=False,
            # pin_memory=True,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_pred,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_pred.collate,
            shuffle=False,
            # pin_memory=True,
            drop_last=False,
        )
