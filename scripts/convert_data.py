import hashlib
import random
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import tqdm
from common import Context, Corpus, Premise
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fc, Path_fr
from retrieval.datamodule import RetrievalDataset


class PremiseOrQueryDataset:
    def __init__(self, name: str):
        self.id_to_txt = {}
        self.txt_to_id = {}
        self.name = name

    def add(self, example: Union[Premise, Context]):
        example = example.serialize()
        hash = hashlib.md5()
        hash.update(example.encode("utf-8"))
        pid = hash.hexdigest()[:8]
        self.id_to_txt[pid] = example
        self.txt_to_id[example] = pid

    def get_id_by_example(self, example: Union[Premise, Context]) -> Optional[str]:
        return self.txt_to_id.get(example.serialize())

    def get_example_by_id(self, id: str) -> str:
        return self.id_to_txt.get(id)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.id_to_txt.items()), columns=["id", self.name])


def get_retrieval_dataset(
    premises_paths_list: List[Union[Path, str]], corpus_path: Union[Path, str]
) -> RetrievalDataset:
    retrieval_dataset = RetrievalDataset(
        premises_paths_list,
        True,
        Corpus(corpus_path),
        num_negatives=2,
        num_in_file_negatives=1,
        max_seq_len=None,
        context_tokenizer=None,
        premise_tokenizer=None,
        is_train=True,
    )
    return retrieval_dataset


def get_premises_from_dataset(
    retrieval_dataset: RetrievalDataset, return_pandas: bool = False
) -> Union[PremiseOrQueryDataset, pd.DataFrame]:
    premises = PremiseOrQueryDataset(name="premise")
    for example in retrieval_dataset.data:
        premises.add(example["pos_premise"])
    if return_pandas:
        return premises.to_frame()
    return premises


def get_queries_from_dataset(
    retrieval_dataset: RetrievalDataset, return_pandas: bool = False
) -> Union[PremiseOrQueryDataset, pd.DataFrame]:
    queries = PremiseOrQueryDataset(name="query")
    for example in retrieval_dataset.data:
        queries.add(example["context"])
    if return_pandas:
        queries.to_frame()
    return queries


def get_triples(
    premises_dataset: PremiseOrQueryDataset,
    queries_dataset: PremiseOrQueryDataset,
    retrieval_dataset: RetrievalDataset,
    seed: Optional[int] = 1234,
) -> pd.DataFrame:
    random.seed(seed)
    triples = []
    for i in tqdm.tqdm(range(len(retrieval_dataset))):
        example = retrieval_dataset[i]
        neg_premise = random.choice(example["neg_premises"])
        triples.append(
            (
                queries_dataset.get_id_by_example(example["context"]),
                premises_dataset.get_id_by_example(example["pos_premise"]),
                premises_dataset.get_id_by_example(neg_premise),
            )
        )

    triples = pd.DataFrame(triples, columns=["qid", "pid+", "pid-"])
    return triples


def convert_to_colbert(
    premises_paths_list: List[Union[Path, str]],
    corpus_path: Union[Path, str],
    triples_tsv_path: Union[Path, str],
    queries_tsv_path: Union[Path, str],
    collection_tsv_path: Union[Path, str],
    seed: Optional[int] = 1234,
) -> None:
    retrieval_dataset = get_retrieval_dataset(premises_paths_list, corpus_path)
    premises = get_premises_from_dataset(retrieval_dataset)
    queries = get_queries_from_dataset(retrieval_dataset)
    triples = get_triples(premises, queries, retrieval_dataset, seed=seed)
    premises.to_frame().to_csv(collection_tsv_path, sep="\t", header=False, index=False)
    queries.to_frame().to_csv(queries_tsv_path, sep="\t", header=False, index=False)
    triples.to_csv(triples_tsv_path, sep="\t", header=False, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--premises_paths_list", type=List[Path_fr], help="Paths to train/val/test.json")
    parser.add_argument("--corpus_path", type=Path_fr, help="Path to corpus")
    parser.add_argument(
        "--triples_tsv_path", type=Path_fc, help="Path to save triples in tsv format with ['qid', 'pid+', 'pid-']"
    )
    parser.add_argument(
        "--queries_tsv_path", type=Path_fc, help="Path to save queries in tsv format with ['id', 'query']"
    )
    parser.add_argument(
        "--collection_tsv_path", type=Path_fc, help="Path to save premises in tsv format with ['id', 'premise']"
    )

    args = parser.parse_args()

    convert_to_colbert(
        args.premises_paths_list,
        args.corpus_path,
        args.triples_tsv_path,
        args.queries_tsv_path,
        args.collection_tsv_path,
    )
