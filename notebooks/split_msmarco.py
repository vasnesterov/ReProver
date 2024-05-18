from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

DATA_ROOT = Path("data/msmarco/")
COLLECTION_PATH = DATA_ROOT / "collection.tsv"
TRIPLES_PATH = DATA_ROOT / "qidpidtriples.train.full.2.tsv.gz"
N_TRIPLES = 397_768_673


def get_collection_subset(read_path: Path) -> Tuple[Dataset, Dict[int, int]]:
    ds = load_dataset(
        "csv",
        data_files=read_path.absolute().as_posix(),
        delimiter="\t",
        column_names=["pid", "passage"],
    )
    collection = ds["train"].train_test_split(test_size=0.05)["test"]
    pid_map = {pid: i for i, pid in enumerate(collection["pid"])}

    def remap_pids(example):
        example["pid"] = pid_map[example["pid"]]
        return example

    collection = collection.map(remap_pids)
    return collection, pid_map


def convert_triples_to_hf_dataset(read_path: Path, save_path: Path) -> None:
    collection = load_dataset(
        "csv",
        data_files=read_path.as_posix(),
        delimiter="\t",
        column_names=["qid", "pos_pid", "neg_pid"],
    )
    collection["train"].save_to_disk(save_path.as_posix())


def train_val_split_triples_by_qid(triples: Dataset, val_size=0.1) -> Tuple[set, set]:

    qid = np.unique(triples["qid"])

    N = len(qid)
    p = np.random.permutation(N)

    qid_dict = {}
    qid_dict["val"] = set(qid[p[: int(N * val_size)]].tolist())
    qid_dict["train"] = set(qid[p[int(N * val_size) :]].tolist())
    return qid_dict


def get_triples(pid_map: Dict[int, int], triples_dir: Path, val_size: float = 0.1) -> DatasetDict:

    def remap_pids(example):
        example["pos_pid"] = pid_map[example["pos_pid"]]
        example["neg_pid"] = pid_map[example["neg_pid"]]
        return example

    def filter_triples(example, qid_set):
        if example["pos_pid"] in pid_map and example["neg_pid"] in pid_map and example["qid"] in qid_set:
            return True
        return False

    triples_full = Dataset.load_from_disk(triples_dir.as_posix())

    qid_dict = train_val_split_triples_by_qid(triples_full, val_size=val_size)

    triples = {}
    for split, qid_set in qid_dict.items():
        triples[split] = triples_full.filter(
            lambda x: filter_triples(x, qid_set),
        ).map(remap_pids)

    return DatasetDict(triples)


def main():
    triples_full_dir = DATA_ROOT / "triples.hf"
    triples_small_dir = DATA_ROOT / "triples_small.hf"
    print(triples_full_dir.absolute().as_posix())
    if not triples_full_dir.exists():
        print("Converting whole triples tsv to HF dataset")
        convert_triples_to_hf_dataset(TRIPLES_PATH, triples_full_dir)

    print("Getting collection subset")
    collection, pid_map = get_collection_subset(COLLECTION_PATH)
    collection.save_to_disk((DATA_ROOT / "collection_small.hf").as_posix())

    print("Getting triples subset")
    dataset_dict = get_triples(pid_map, triples_full_dir)

    print(dataset_dict)
    dataset_dict.save_to_disk((triples_small_dir).as_posix())


if __name__ == "__main__":
    main()
