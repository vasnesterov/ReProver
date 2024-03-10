from typing import List, Optional, Union

from colbert import Trainer
from colbert.infra import ColBERTConfig, Run, RunConfig
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_dc, Path_fc, Path_fr


def main(args):
    with Run().context(RunConfig(nranks=1, experiment="lean")):

        config = ColBERTConfig(
            bsize=1,
            root=args.experiment_root.abs_path,
        )
        trainer = Trainer(
            triples=args.triples_path.abs_path,
            queries=args.queries_path.abs_path,
            collection=args.collection_path.abs_path,
            config=config,
        )

        if args.pretrained_checkpoint is None:
            checkpoint_path = trainer.train()
        else:
            checkpoint_path = trainer.train(checkpoint=args.pretrained_checkpoint)

        print(f"Saved checkpoint to {checkpoint_path}...")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--premises_paths_list", type=List[Path_fr], help="Paths to train/val/test.json"
    )
    parser.add_argument("--corpus_path", type=Path_fr, help="Path to corpus")
    parser.add_argument(
        "--triples_path",
        type=Path_fc,
        help="Path to triples in jsonl format with ['qid', ['pid', score], ['pid', score], ...]",
    )
    parser.add_argument(
        "--queries_path",
        type=Path_fc,
        help="Path to queries in json format with {'qid': int, 'quesion': str}",
    )
    parser.add_argument(
        "--collection_path",
        type=Path_fc,
        help="Path to premises in tsv format with ['id', 'premise']",
    )
    parser.add_argument("--experiment_root", type=Path_dc)
    parser.add_argument("--pretrained_checkpoint", type=Optional[Union[Path_fr, str]])

    args = parser.parse_args()

    main(args)
