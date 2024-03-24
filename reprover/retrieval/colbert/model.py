from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from colbert.data import Collection
from colbert.infra.config import ColBERTConfig
from colbert.searcher import Searcher
from lean_dojo import Pos
from loguru import logger
from tqdm import tqdm

from reprover.common import Context, Corpus, Premise, zip_strict
from reprover.retrieval.base_model import BasePremiseRetriever, PremiseRetrieverAPI
from reprover.retrieval.colbert.indexer import ColBERTIndexer

torch.set_float32_matmul_precision("medium")


class ColBERTPremiseRetriever(PremiseRetrieverAPI, Searcher):
    def __init__(
        self,
        index_name: str,
        experiment_name: str,
        checkpoint_path_or_name: Optional[str] = None,
        collection: Optional[Union[str, Collection]] = None,
        config: Optional[Union[str, ColBERTConfig]] = None,
        index_root: Optional[str] = None,
        num_retrieved: int = 100,
        verbose: int = 3,
    ) -> None:
        Searcher.__init__(
            self,
            index=index_name,
            checkpoint=checkpoint_path_or_name,
            collection=collection,
            config=config,
            index_root=Path(index_root) / experiment_name / "indexes",
            verbose=verbose,
        )
        self.indexer = ColBERTIndexer(self.checkpoint)
        self.indexer.configure(
            root=index_root,
            experiment=experiment_name,
        )
        self.index_name = index_name
        self.num_retrieved = num_retrieved
        self.corpus = None

    @property
    def embedding_size(self) -> int:
        """Return the size of the feature vector produced by ``encoder``."""
        return self.config.dim

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool) -> BasePremiseRetriever:
        # return super().load(ckpt_path, device, freeze)
        pass

    def load_corpus(self, path_or_corpus: Union[str, Corpus]) -> None:
        """Associate the retriever with a corpus."""
        if isinstance(path_or_corpus, Corpus):
            self.corpus = path_or_corpus
        else:
            self.corpus = Corpus(path_or_corpus)

    def _encode_context(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        """Encode a context into a feature vector."""
        return self.checkpoint.query(input_ids, attention_mask)

    def _encode_premise(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        keep_dims: bool = True,
        to_cpu: bool = False,
    ) -> torch.FloatTensor:
        """Encode a context into a feature vector."""
        return self.checkpoint.doc(
            input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu
        )

    @torch.no_grad()
    def reindex_corpus(self, batch_size: int) -> None:
        """Re-index the retrieval corpus using the up-to-date encoder."""
        index_bsize = self.config.index_bsize
        try:
            self.indexer.configure(index_bsize=batch_size)
            self.indexer.index(
                self.index_name, collection=self.collection, overwrite=True
            )
        finally:
            self.configure(index_bsize=index_bsize)

    def retrieve_from_preprocessed(self, batch):
        context_emb = self._encode_context(batch["context_ids"], batch["context_mask"])
        retrieved_premises, retrieved_scores = [], []
        for query_idx in tqdm(range(len(context_emb))):
            pids, ranks, scores = self.dense_search(
                context_emb[query_idx : query_idx + 1],
                self.num_retrieved,
                filter_fn=None,
                pids=None,
            )
            retrieved_premises.append([self.corpus.all_premises[pid] for pid in pids])
            retrieved_scores.append(scores)

        return retrieved_premises, scores

    def retrieve(
        self,
        state: List[str],
        file_name: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        k: int,
        reindex_batch_size: int = 32,
        do_reindex: bool = True,
    ) -> Tuple[List[Premise], List[float]]:
        """Retrieve ``k`` premises from ``corpus`` using ``state`` and ``tactic_prefix`` as context."""
        assert self.corpus is not None
        if do_reindex:
            self.reindex_corpus(batch_size=reindex_batch_size)

        contexts = [
            Context(*args).serialize()
            for args in zip_strict(file_name, theorem_full_name, theorem_pos, state)
        ]
        ranking = self.search_all(
            {i: ctx for i, ctx in enumerate(contexts)},
            k=k,
            filter_fn=None,
            full_length_search=False,
            qid_to_pids=None,
        )
        retrieved_premises, retrieved_scores = [], []
        for i in range(len(contexts)):
            pids, ranks, scores = zip(*ranking.data[i])
            retrieved_premises.append([self.corpus.all_premises[pid] for pid in pids])
            retrieved_scores.append(scores)

        return ranking, retrieved_premises, retrieved_scores


class ColBERTPremiseRetrieverLightning(BasePremiseRetriever, ColBERTPremiseRetriever):
    def __init__(
        self,
        index_name: str,
        experiment_name: str,
        checkpoint_path_or_name: Optional[str] = None,
        collection: Optional[Union[str, Collection]] = None,
        config: Optional[Union[str, ColBERTConfig]] = None,
        index_root: Optional[str] = None,
        num_retrieved: int = 100,
        verbose: int = 3,
    ) -> None:
        BasePremiseRetriever.__init__(self)
        # self.save_hyperparameters()
        ColBERTPremiseRetriever.__init__(
            self,
            index_name=index_name,
            experiment_name=experiment_name,
            checkpoint_path_or_name=checkpoint_path_or_name,
            collection=collection,
            config=config,
            index_root=index_root,
            num_retrieved=num_retrieved,
            verbose=verbose,
        )

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool) -> "BasePremiseRetriever":
        raise NotImplementedError

    def forward(
        self,
        context_ids: torch.LongTensor,
        context_mask: torch.LongTensor,
        pos_premise_ids: torch.LongTensor,
        pos_premise_mask: torch.LongTensor,
        neg_premises_ids: torch.LongTensor,
        neg_premises_mask: torch.LongTensor,
        label: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Compute the contrastive loss for premise retrieval."""
        raise NotImplementedError
