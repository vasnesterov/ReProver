from typing import Optional, Union

from colbert import Trainer
from colbert.infra import ColBERTConfig, Run, RunConfig
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_dc, Path_fc, Path_fr
from pytorch_lightning.loggers import WandbLogger


def setup_parser():
    parser = ArgumentParser()

    parser.add_argument(
        "--run", type=RunConfig, default=RunConfig(), help="Run configuration"
    )
    parser.add_argument(
        "--colbert",
        type=ColBERTConfig,
        default=ColBERTConfig(),
        help="Colbert configuration",
    )
    parser.add_class_arguments(WandbLogger, nested_key='logger')

    parser.add_argument(
        "--data.triples_path",
        type=Path_fc,
        help="Path to triples in jsonl format with ['qid', ['pid', score], ['pid', score], ...]",
    )
    parser.add_argument(
        "--data.queries_path",
        type=Path_fc,
        help="Path to queries in json format with {'qid': int, 'quesion': str}",
    )
    parser.add_argument(
        "--data.collection_path",
        type=Path_fc,
        help="Path to premises in tsv format with ['id', 'premise']",
    )
    parser.add_argument("--experiment_root", type=Path_dc)
    parser.add_argument("--pretrained_checkpoint", type=Optional[Union[Path_fr, str]])

    parser.link_arguments(
        "experiment_root", "colbert.root", compute_fn=lambda source: source.absolute
    )

    return parser


def main(args):
    with Run().context(args.run):
        trainer = Trainer(
            triples=args.data.triples_path.absolute,
            queries=args.data.queries_path.absolute,
            collection=args.data.collection_path.absolute,
            config=args.colbert,
            logger=args.logger,
        )

        if args.pretrained_checkpoint is None:
            checkpoint_path = trainer.train()
        else:
            checkpoint_path = trainer.train(checkpoint=args.pretrained_checkpoint)

        print(f"Saved checkpoint to {checkpoint_path}...")


if __name__ == "__main__":

    parser = setup_parser()
    args = parser.parse_args()

    args = parser.instantiate_classes(args)

    main(args)
