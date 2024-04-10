from pathlib import Path
from typing import Optional

from colbert import Checkpoint, Indexer
from colbert.data.collection import Collection
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.indexing.collection_indexer import CollectionIndexer
from colbert.indexing.index_saver import IndexSaver
from colbert.infra.config import ColBERTConfig
from colbert.infra.launcher import print_memory_stats
from colbert.infra.run import Run
from colbert.modeling.checkpoint import Checkpoint
from colbert.utils.utils import create_directory


class ColBERTIndexer(Indexer):
    def __init__(
        self,
        checkpoint: Checkpoint,
        config: Optional[ColBERTConfig] = None,
        verbose: int = 3,
    ) -> None:
        """
        Use Run().context() to choose the run's configuration. They are NOT extracted from `config`.
        """
        self.index_path = None
        self.verbose = verbose
        self.checkpoint = checkpoint

        self.checkpoint_config = self.checkpoint.colbert_config

        self.config = ColBERTConfig.from_existing(
            self.checkpoint_config, config, Run().config
        )
        self.configure(checkpoint=checkpoint.name)

    def index(self, name, collection, overwrite=False):
        assert overwrite in [True, False, "reuse", "resume", "force_silent_overwrite"]

        self.configure(
            collection=collection, index_name=name, resume=overwrite == "resume",
            triples=None, queries=None
        )
        # Note: The bsize value set here is ignored internally. Users are encouraged
        # to supply their own batch size for indexing by using the index_bsize parameter in the ColBERTConfig.
        self.configure(bsize=64, partitions=None)

        self.index_path = self.config.index_path_
        index_does_not_exist = not Path(self.config.index_path_).exists()

        assert (
            overwrite in [True, "reuse", "resume", "force_silent_overwrite"]
        ) or index_does_not_exist, self.config.index_path_
        create_directory(self.config.index_path_)

        if overwrite == "force_silent_overwrite":
            self.erase(force_silent=True)
        elif overwrite is True:
            self.erase()

        if index_does_not_exist or overwrite != "reuse":
            encoder = CollectionIndexerFromCheckpoint(
                self.checkpoint,
                config=self.config,
                collection=collection,
                verbose=self.verbose,
            )
            encoder.run(None)

        return self.index_path


class CollectionIndexerFromCheckpoint(CollectionIndexer):
    """
    Given a collection and config, encode collection into index and
    stores the index on the disk in chunks.
    """

    def __init__(
        self, checkpoint: Checkpoint, config: ColBERTConfig, collection, verbose=2
    ) -> None:
        self.verbose = verbose
        self.config = config
        self.rank, self.nranks = self.config.rank, self.config.nranks

        self.use_gpu = self.config.total_visible_gpus > 0

        if self.config.rank == 0 and self.verbose > 1:
            self.config.help()

        self.collection = Collection.cast(collection)
        self.checkpoint = checkpoint
        if self.use_gpu:
            self.checkpoint = self.checkpoint.cuda()

        self.encoder = CollectionEncoder(config, self.checkpoint)
        self.saver = IndexSaver(config)

        print_memory_stats(f"RANK:{self.rank}")
