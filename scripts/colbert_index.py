from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_dc, Path_fr, path_type

Path_dr = path_type("dr")

from reprover.retrieval.colbert_retrieval.indexer import ColBERTIndexer


def setup_parser():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=Path_dr, help="Path to ColBERT checkpoint")
    parser.add_argument(
        "--collection_path",
        type=Path_fr,
        help="Path to the ColBERT collection of premises",
    )
    parser.add_argument(
        "--index_root",
        type=Path_dc,
        help="Path to the directory with index (can be existing path)",
    )
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment or index")
    parser.add_argument("--index_name", type=str, help="Name of the experiment or index")
    parser.add_argument(
        "--index_bsize",
        type=int,
        default=64,
        help="Batch size that will be used for indexing",
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help="If True, the existing index will be overwritten",
    )
    return parser


def index(args):
    config = ColBERTConfig.load_from_checkpoint(args.checkpoint_path.absolute)
    checkpoint = Checkpoint(args.checkpoint_path.absolute, colbert_config=config)

    indexer = ColBERTIndexer(checkpoint=checkpoint)
    indexer.configure(
        index_bsize=args.index_bsize,
        root=args.index_root.absolute,
        experiment=args.experiment_name,
    )
    indexer.index(
        args.index_name,
        collection=args.collection_path.absolute,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    parser = setup_parser()
    index(parser.parse_args())
