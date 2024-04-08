import os
from typing import Union

from colbert.data import Collection
from colbert.infra.config import ColBERTConfig
from colbert.infra.run import Run
from colbert.modeling.checkpoint import Checkpoint
from colbert.search.index_storage import IndexScorer
from colbert.searcher import Searcher
from reprover.retrieval.colbert_retrieval.checkpoint import TrainingCheckpoint


class TrainingSearcher(Searcher):
    """
    Searcher with the checkpoint passed as initialized model, not just the name
    The model should be TrainingCheckpoint
    """

    def __init__(
        self,
        index,
        checkpoint: Union[TrainingCheckpoint, Checkpoint],
        collection=None,
        config=None,
        index_root=None,
        verbose: int = 3,
    ):
        self.verbose = verbose

        initial_config = ColBERTConfig.from_existing(config, Run().config)

        default_index_root = initial_config.index_root_
        index_root = index_root if index_root else default_index_root
        self.index = os.path.join(index_root, index)
        self.index_config = ColBERTConfig.load_from_index(self.index)

        self.checkpoint = checkpoint
        self.config = ColBERTConfig.from_existing(self.checkpoint.colbert_config, self.index_config, initial_config)

        self.collection = Collection.cast(collection or self.config.collection)
        self.configure(checkpoint=self.checkpoint.name, collection=self.collection)

        use_gpu = self.config.total_visible_gpus > 0
        if use_gpu:
            self.checkpoint = self.checkpoint.cuda()
        load_index_with_mmap = self.config.load_index_with_mmap
        if load_index_with_mmap and use_gpu:
            raise ValueError(f"Memory-mapped index can only be used with CPU!")
        self.ranker = IndexScorer(self.index, use_gpu, load_index_with_mmap)
