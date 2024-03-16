from typing import List, Optional, Tuple, Union

import torch
from colbert import Indexer
from colbert.data import Collection
from colbert.infra.config import ColBERTConfig
from colbert.searcher import Searcher
from lean_dojo import Pos
from loguru import logger
from tqdm import tqdm

from reprover.common import Context, Corpus, Premise, zip_strict
from reprover.retrieval.base_model import PremiseRetrieverAPI

torch.set_float32_matmul_precision("medium")


class ColBERTPremiseRetriever(PremiseRetrieverAPI, Searcher):
    def __init__(
        self,
        index_name: str,
        checkpoint_path_or_name: Optional[str] = None,
        collection: Optional[Union[str, Collection]] = None,
        config: Optional[Union[str, ColBERTConfig]] = None,
        index_root: Optional[str] = None,
        num_retrieved: int = 100,
        verbose: int = 3,
    ) -> None:
        super(Searcher, self).__init__(
            index=index_name,
            checkpoint=checkpoint_path_or_name,
            collection=collection,
            config=config,
            index_root=index_root,
            verbose=verbose,
        )
        self.indexer = Indexer(checkpoint_path_or_name)
        self.num_retrieved = num_retrieved

    @property
    def embedding_size(self) -> int:
        """Return the size of the feature vector produced by ``encoder``."""
        return self.config.dim

    def load_corpus(self, path_or_corpus: Union[str, Corpus]) -> None:
        """Associate the retriever with a corpus."""
        pass

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
        if not self.embeddings_staled:
            return
        logger.info("Re-indexing the retrieval corpus")

        self.corpus_embeddings = torch.zeros(
            len(self.corpus.all_premises),
            self.embedding_size,
            dtype=self.encoder.dtype,
            device=self.device,
        )

        for i in tqdm(range(0, len(self.corpus), batch_size)):
            batch_premises = self.corpus.all_premises[i : i + batch_size]
            tokenized_premises = self.tokenizer(
                [p.serialize() for p in batch_premises],
                padding="longest",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            self.corpus_embeddings[i : i + batch_size] = self._encode(
                tokenized_premises.input_ids, tokenized_premises.attention_mask
            )

        self.embeddings_staled = False

    def retrieve_from_preprocessed(self, batch):
        raise NotImplementedError
        context_emb = self._encode_context(batch["context_ids"], batch["context_mask"])
        assert not self.embeddings_staled
        retrieved_premises, scores = self.corpus.get_nearest_premises(
            self.corpus_embeddings,
            batch["context"],
            context_emb,
            self.num_retrieved,
        )
        return retrieved_premises, scores

    def retrieve(
        self,
        state: List[str],
        file_name: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        k: int,
        reindex_batch_size: int = 32,
    ) -> Tuple[List[Premise], List[float]]:
        """Retrieve ``k`` premises from ``corpus`` using ``state`` and ``tactic_prefix`` as context."""
        self.reindex_corpus(batch_size=reindex_batch_size)

        contexts = [
            Context(*args).serialize()
            for args in zip_strict(file_name, theorem_full_name, theorem_pos, state)
        ]
        ranking = self.searcher.search_all(
            contexts,
            k=self.num_retrieved,
            filter_fn=None,
            full_length_search=False,
            qid_to_pids=None,
        )

        return ranking
