from functools import partial
from itertools import chain
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
from colbert.data.collection import Collection
from colbert.data.examples import Examples
from colbert.data.queries import Queries
from colbert.infra.config import ColBERTConfig
from colbert.infra.config.config import ColBERTConfig
from colbert.modeling.tokenization import DocTokenizer, QueryTokenizer, tensorize_triples
from colbert.utils.utils import zipstar
from jsonargparse.typing import Path_fr, path_type
from reprover.common import Corpus
from reprover.retrieval.datamodule import RetrievalDataset
from torch.utils.data import DataLoader, Dataset

Path_dr = path_type("dr")


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
        batch = {"queries": Q, "passages": D, "scores": s, "positive_passages": positive_passages, "contexts": queries}
        return batch


class ColBERTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: ColBERTConfig,
        data_path: str,
        batch_size: int,
        eval_batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.config.configure(bsize=batch_size)

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


class ColBERTDatasetNative(RetrievalDataset):
    def __init__(
        self,
        config: ColBERTConfig,
        corpus: Corpus,
        data_path_list: List[Path_fr | str],
        is_train: bool,
    ):
        super(RetrievalDataset, self).__init__()
        self.nway = config.nway

        self.corpus = corpus
        self.num_negatives = config.num_negatives
        self.num_in_file_negatives = config.num_in_file_negatives
        self.max_seq_len = None
        self.context_tokenizer = QueryTokenizer(config)
        self.premise_tokenizer = DocTokenizer(config)
        self.is_train = is_train

        self.data = list(chain.from_iterable(self._load_data(path) for path in data_path_list))

        self.tensorize_triples = partial(tensorize_triples, self.context_tokenizer, self.premise_tokenizer)

    def collate(self, inputs):

        queries, passages, scores = [], [], []
        positive_passages = []
        for example in inputs:
            queries.append(example["context"].serialize())

            passages.append(example["pos_premise"].serialize())
            positive_passages.append(passages[-1][0])
            passages.extend([p.serialize() for p in example["neg_premises"]])
            s = [0.0] * (len(example["neg_premises"]) + 1)
            s[0] = 1.0
            scores.extend(s)

        bsize = len(queries)

        Q, D, s = self.tensorize_triples(queries, passages, scores, bsize, self.nway)[0]
        batch = {"queries": Q, "passages": D, "scores": s, "positive_passages": positive_passages, "contexts": queries}
        return batch


class ColBERTDataModuleNative(pl.LightningDataModule):
    def __init__(
        self,
        config: ColBERTConfig,
        corpus_path: Path_fr | str,
        data_dir: Path_dr | str,
        batch_size: int,
        eval_batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.corpus_path = corpus_path
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.config.configure(bsize=batch_size)
        self.corpus = None

    def prepare_data(self) -> None:
        pass

    def save_datasets(self):
        if self.corpus is None:
            self.corpus = Corpus(self.corpus_path)

        for split in ["train", "val"]:
            data_path_list = [self.data_dir / f"{split}.json"]
            ds = ColBERTDatasetNative(self.config, self.corpus, data_path_list, is_train=True)
            ds.save_data(self.data_dir / f"{split}.ckpt")

        [self.data_dir / f"{split}.json" for split in ["train", "val", "test"]]
        ds = ColBERTDatasetNative(self.config, self.corpus, data_path_list, is_train=True)
        ds.save_data(self.data_dir / "test.ckpt")

    def setup(self, stage: Optional[str] = None) -> None:
        self.corpus = Corpus(self.corpus_path)
        if stage in (None, "fit", "validate"):
            if (self.data_dir / "train.ckpt").exists():
                data_path_list = [self.data_dir / "train.ckpt"]
            else:
                data_path_list = [self.data_dir / "train.json"]

            self.ds_train = ColBERTDatasetNative(self.config, self.corpus, data_path_list, is_train=True)

        if stage in (None, "fit", "validate"):
            if (self.data_dir / "val.ckpt").exists():
                data_path_list = [self.data_dir / "val.ckpt"]
            else:
                data_path_list = [self.data_dir / "val.json"]
            self.ds_val = ColBERTDatasetNative(self.config, self.corpus, data_path_list, is_train=True)

        if stage in (None, "fit", "predict"):
            if (self.data_dir / "test.ckpt").exists():
                data_path_list = [self.data_dir / "test.ckpt"]
            else:
                data_path_list = [self.data_dir / f"{split}.json" for split in ["train", "val", "test"]]
            self.ds_pred = ColBERTDatasetNative(self.config, self.corpus, data_path_list, is_train=False)

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
            # drop_last=False,
            drop_last=True,  # TODO: fix it
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


if __name__ == "__main__":
    cfg = ColBERTConfig()
    cfg.checkpoint = "microsoft/deberta-v3-xsmall"  # "kaiyuy/leandojo-lean4-retriever-byt5-small"
    cfg.root = Path(cfg.root) / "../experiments"
    cfg.query_maxlen = 128
    cfg.doc_maxlen = 180
    cfg.num_negatives = 7
    cfg.num_in_file_negatives = 3

    datamodule = ColBERTDataModuleNative(
        cfg,
        corpus_path="/home/yeahrmek/github/theorem_proving/ReProver/data/leandojo_benchmark_4/corpus.jsonl",
        data_dir="/home/yeahrmek/github/theorem_proving/ReProver/data/leandojo_benchmark_4/random/",
        batch_size=24,
        eval_batch_size=128,
        num_workers=1,
    )

    # datamodule.save_datasets()

    datamodule.setup()
    loader = datamodule.train_dataloader()
    import time

    s = time.time()
    for batch in loader:
        pass

    print(time.time() - s)

    loader = datamodule.val_dataloader()
    s = time.time()
    for batch in loader:
        pass
    print(time.time() - s)
