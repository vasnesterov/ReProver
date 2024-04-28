import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import jsonlines
import numpy as np
import pandas as pd
import tqdm
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_drw, Path_fc, Path_fr
from reprover.common import Context, Corpus, Premise
from reprover.retrieval.datamodule import RetrievalDataset


class PremiseOrQueryDataset:
    def __init__(self, key_name: str, value_name: str):
        self.id_to_txt = {}
        self.txt_to_id = {}
        self.value_name = value_name
        self.key_name = key_name

    def add(self, example: Union[Premise, Context]):
        example = example.serialize()
        if example in self.txt_to_id:
            return
        pid = len(self.id_to_txt)
        self.id_to_txt[pid] = example
        self.txt_to_id[example] = pid

    def get_id_by_example(self, example: Union[Premise, Context], missing_ok: bool = False) -> Optional[int]:
        if missing_ok:
            return self.txt_to_id.get(example.serialize())
        return self.txt_to_id[example.serialize()]

    def get_example_by_id(self, id: int, missing_ok: bool = False) -> str:
        if missing_ok:
            return self.id_to_txt.get(id)
        return self.id_to_txt[id]

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(list(self.id_to_txt.items()), columns=[self.key_name, self.value_name]).sort_values(
            self.key_name
        )
        return df

    def to_tsv(self, path) -> None:
        self.to_frame().to_csv(path, sep="\t", header=False, index=False)

    def to_json(self, path) -> None:
        with jsonlines.open(path, "w") as writer:
            for key, value in sorted(self.id_to_txt.items()):
                writer.write({self.key_name: key, self.value_name: value})


def get_retrieval_dataset(
    premises_paths_list: List[Union[Path, str]],
    corpus_or_path: Union[Path, str, Corpus],
    num_negatives: int,
    num_in_file_negatives: int,
) -> RetrievalDataset:
    if isinstance(corpus_or_path, Corpus):
        corpus = corpus_or_path
    else:
        corpus = Corpus(corpus_or_path)
    retrieval_dataset = RetrievalDataset(
        premises_paths_list,
        corpus,
        num_negatives=num_negatives,
        num_in_file_negatives=num_in_file_negatives,
        max_seq_len=None,
        context_tokenizer=None,
        premise_tokenizer=None,
        is_train=True,
    )
    return retrieval_dataset


def get_premises_from_dataset(
    retrieval_dataset: RetrievalDataset, return_pandas: bool = False
) -> Union[PremiseOrQueryDataset, pd.DataFrame]:
    premises = PremiseOrQueryDataset(key_name="pid", value_name="passage")
    for pr in retrieval_dataset.corpus.all_premises:
        premises.add(pr)

    if return_pandas:
        return premises.to_frame()
    return premises


def get_queries_from_dataset(
    retrieval_dataset: RetrievalDataset, return_pandas: bool = False
) -> Union[PremiseOrQueryDataset, pd.DataFrame]:
    queries = PremiseOrQueryDataset(key_name="qid", value_name="question")
    for example in retrieval_dataset.data:
        queries.add(example["context"])
    if return_pandas:
        queries.to_frame()
    return queries


def get_triples(
    premises_dataset: PremiseOrQueryDataset,
    queries_dataset: PremiseOrQueryDataset,
    retrieval_dataset: RetrievalDataset,
    num_negatives: int = 1,
    seed: Optional[int] = 1234,
) -> List[Tuple[str, str, str]]:
    random.seed(seed)
    np.random.seed(seed)

    triples = []
    for i in tqdm.tqdm(range(len(retrieval_dataset))):
        example = retrieval_dataset[i]
        pos_premise_id = premises_dataset.get_id_by_example(example["pos_premise"])

        negatives = []

        for i in range(num_negatives):
            neg_premise_id = premises_dataset.get_id_by_example(example["neg_premises"][i], missing_ok=False)
            negatives.append([neg_premise_id, 0.0])

        triples.append(
            (
                queries_dataset.get_id_by_example(example["context"]),
                [pos_premise_id, 1.0],
                *negatives,
            )
        )

    return triples


def convert_to_colbert(
    premises_dir: List[Union[Path, str]],
    corpus_path: Union[Path, str],
    save_dir: Union[Path, str],
    num_negatives: int,
    num_in_file_negatives: int,
    seed: Optional[int] = 1234,
) -> None:

    corpus = None
    premises = None
    for split in ['train', 'val', 'test']:
        print(f"Processing {split} dataset")
        premises_path = Path(premises_dir) / f"{split}.json"
        if corpus is None:
            retrieval_dataset = get_retrieval_dataset(
                [premises_path],
                corpus_path,
                num_negatives=num_negatives,
                num_in_file_negatives=num_in_file_negatives,
            )
            corpus = retrieval_dataset.corpus
        else:
            retrieval_dataset = get_retrieval_dataset(
                [premises_path],
                corpus,
                num_negatives=num_negatives,
                num_in_file_negatives=num_in_file_negatives,
            )

        if premises is None:
            premises = get_premises_from_dataset(retrieval_dataset)

        queries = get_queries_from_dataset(retrieval_dataset)
        triples = get_triples(premises, queries, retrieval_dataset, num_negatives=num_negatives, seed=seed)
        queries.to_json(Path(save_dir) / f"queries_{split}.json")

        with jsonlines.open(Path(save_dir) / f"triples_{split}.json", "w") as writer:
            for triple in triples:
                writer.write(triple)
        print('-' * 80)

    premises.to_tsv(Path(save_dir) / "collection.tsv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--premises_dir", type=Path_drw, help="Paths to directory with train/val/test.json")
    parser.add_argument("--corpus_path", type=Path_fr, help="Path to corpus")
    parser.add_argument(
        "--save_dir",
        type=Path_drw,
        help="Path to save dir",
    )
    parser.add_argument("--num_negatives", type=int, help="Number of negatives per query to generate")
    parser.add_argument(
        "--num_in_file_negatives",
        type=int,
        help="Number of in-file-negatives per query to generate",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    args = parser.parse_args()

    convert_to_colbert(
        args.premises_dir,
        args.corpus_path,
        args.save_dir,
        num_negatives=args.num_negatives,
        num_in_file_negatives=args.num_in_file_negatives,
        seed=args.seed,
    )
