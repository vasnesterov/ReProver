import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from colbert.data import Collection
from colbert.infra.config import ColBERTConfig
from colbert.infra.run import Run
from colbert.searcher import Searcher

# from lean_dojo import Pos
from reprover.common import Context, Premise, zip_strict
from reprover.retrieval.base_model import BasePremiseRetriever, PremiseRetrieverAPI
from reprover.retrieval.colbert_retrieval.checkpoint import TrainingCheckpoint
from reprover.retrieval.colbert_retrieval.indexer import ColBERTIndexer
from reprover.retrieval.colbert_retrieval.searcher import TrainingSearcher
from tqdm import tqdm

torch.set_float32_matmul_precision("medium")


class ColBERTPremiseRetriever(PremiseRetrieverAPI):
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
        training: bool = False,
    ) -> None:
        super().__init__()

        if training:
            initial_config = ColBERTConfig.from_existing(config, Run().config)

            index_root = Path(index_root) / experiment_name / "indexes"

            # don't load index config if there is no index
            index_path = index_root / index_name
            if (index_path / "metadata.json").exists() or (index_path / "plan.json").exists():
                index_config = ColBERTConfig.load_from_index(index_path.as_posix())
            index_config = initial_config

            index_root = index_root.as_posix()
            checkpoint = checkpoint_path_or_name or index_config.checkpoint
            checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint)
            config = ColBERTConfig.from_existing(checkpoint_config, index_config, initial_config)

            checkpoint = TrainingCheckpoint(checkpoint, colbert_config=config, verbose=verbose)
            self.searcher = TrainingSearcher(
                index=index_name,
                checkpoint=checkpoint,
                collection=collection,
                config=config,
                index_root=index_root,
                verbose=verbose,
            )
        else:
            self.searcher = Searcher(
                index=index_name,
                checkpoint=checkpoint_path_or_name,
                collection=collection,
                config=config,
                index_root=Path(index_root) / experiment_name / "indexes",
                verbose=verbose,
            )

        self.checkpoint = self.searcher.checkpoint
        self.config = self.searcher.config

        self.indexer = ColBERTIndexer(self.checkpoint)
        self.indexer.configure(
            root=index_root,
            experiment=experiment_name,
        )
        self.index_name = index_name
        self.num_retrieved = num_retrieved

    @property
    def embedding_size(self) -> int:
        """Return the size of the feature vector produced by ``encoder``."""
        return self.config.dim

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool) -> BasePremiseRetriever:
        # return super().load(ckpt_path, device, freeze)
        pass

    def load_corpus(self, path_or_corpus: str) -> None:
        """Associate the retriever with a corpus."""
        self.corpus = path_or_corpus

    def _encode_context(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.FloatTensor:
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
        return self.checkpoint.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)

    @torch.no_grad()
    def reindex_corpus(self, batch_size: int) -> None:
        """Re-index the retrieval corpus using the up-to-date encoder."""
        index_bsize = self.config.index_bsize
        try:
            self.indexer.configure(index_bsize=batch_size)
            self.indexer.index(self.index_name, collection=self.searcher.collection, overwrite=True)
        finally:
            self.searcher.configure(index_bsize=index_bsize)

        if self.searcher.ranker is None:
            self.searcher.init_ranker()

    def retrieve_from_preprocessed(self, batch):
        # context_emb = self._encode_context(batch["context_ids"], batch["context_mask"])
        if isinstance(batch, dict):
            Q = batch["queries"]
        else:
            Q, D, s = batch
        context_emb = self._encode_context(*Q)
        retrieved_premises, retrieved_scores = [], []
        for query_idx in tqdm(range(len(context_emb))):
            pids, ranks, scores = self.searcher.dense_search(
                context_emb[query_idx : query_idx + 1],
                self.num_retrieved,
                filter_fn=None,
                pids=None,
            )
            retrieved_premises.append([self.search.collection[pid] for pid in pids])
            retrieved_scores.append(scores)

        return retrieved_premises, retrieved_scores

    def retrieve(
        self,
        state: List[str],
        file_name: List[str],
        theorem_full_name: List[str],
        theorem_pos: List["Pos"],
        k: int,
        reindex_batch_size: int = 32,
        do_reindex: bool = True,
    ) -> Tuple[List[Premise], List[float]]:
        """Retrieve ``k`` premises from ``corpus`` using ``state`` and ``tactic_prefix`` as context."""
        assert self.corpus is not None
        if do_reindex:
            self.reindex_corpus(batch_size=reindex_batch_size)

        contexts = [Context(*args).serialize() for args in zip_strict(file_name, theorem_full_name, theorem_pos, state)]
        ranking = self.searcher.search_all(
            {i: ctx for i, ctx in enumerate(contexts)},
            k=k,
            filter_fn=None,
            full_length_search=False,
            qid_to_pids=None,
        )
        retrieved_premises, retrieved_scores = [], []
        for i in range(len(contexts)):
            pids, ranks, scores = zip(*ranking.data[i])
            retrieved_premises.append([self.searcher.collection[pid] for pid in pids])
            retrieved_scores.append(scores)

        return ranking, retrieved_premises, retrieved_scores


class ColBERTPremiseRetrieverLightning(BasePremiseRetriever, ColBERTPremiseRetriever):
    def __init__(
        self,
        config: ColBERTConfig,
        index_name: str,
        experiment_name: str,
        checkpoint_path_or_name: str = None,
        collection: str = None,
        index_root: str = None,
        num_retrieved: int = 100,
        lr: float = 1e-5,
        warmup_steps: int = 20000,
        verbose: int = 3,
        debug: bool = False,
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
            training=True,
        )
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.config.configure(lr=self.lr, warmup=self.warmup_steps)
        self.checkpoint_path_or_name = checkpoint_path_or_name
        self.debug = debug

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool) -> "BasePremiseRetriever":
        raise NotImplementedError

    def forward(self, batch) -> torch.FloatTensor:
        """Compute the contrastive loss for premise retrieval."""
        if isinstance(batch, dict):
            queries = batch["queries"]
            passages = batch["passages"]
            target_scores = batch["scores"]
            encoding = [queries, passages]
        else:
            queries, passages, target_scores = batch
            encoding = [queries, passages]

        scores = self.checkpoint(*encoding)

        if self.config.use_ib_negatives:
            scores, ib_loss = scores

        scores = scores.view(-1, self.config.nway)

        if len(target_scores) and not self.config.ignore_scores:
            target_scores = torch.tensor(target_scores).view(-1, self.config.nway).to(scores.device)
            target_scores = target_scores * self.config.distillation_alpha
            target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)

            log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
            loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)(log_scores, target_scores)
        else:
            loss = torch.nn.CrossEntropyLoss()(
                scores,
                torch.zeros(scores.size(0), dtype=torch.long, device=scores.device),
            )

        positive_avg = scores[:, 0].mean().item()
        negative_avg = scores[:, 1].mean().item()

        metrics = {
            "positive_score": positive_avg,
            "negative_score": negative_avg,
            "score_diff": positive_avg - negative_avg,
        }

        if self.config.use_ib_negatives:
            loss += ib_loss

            metrics["ib_loss"] = ib_loss

        metrics["loss"] = loss

        return loss, metrics

    def training_step(self, batch) -> torch.Tensor:
        loss, metrics = self(batch)

        for key, value in metrics.items():
            self.log(
                f"train/{key}",
                value,
                on_epoch=True,
                sync_dist=True,
                batch_size=len(batch),
            )
        return loss

    def on_validation_start(self) -> None:
        if not self.debug:
            self.reindex_corpus(self.trainer.datamodule.eval_batch_size)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Retrieve premises and calculate metrics such as Recall@K and MRR."""
        # Retrieval.
        retrieved_premises, _ = self.retrieve_from_preprocessed(batch)

        # Evaluation & logging.
        recall = [[] for _ in range(self.num_retrieved)]
        MRR = []
        num_with_premises = 0
        tb = self.logger.experiment
        text_columns_to_log = ["ground truth", "retrieved", f"Recall@{self.num_retrieved}"]
        text_to_log = []

        # for i, (all_pos_premises, premises) in enumerate(zip_strict(batch["all_pos_premises"], retrieved_premises)):
        for i, (all_pos_premises, premises) in enumerate(zip_strict(batch["positive_passages"], retrieved_premises)):
            # Only log the first example in the batch.
            if i == 0:
                # msg_gt = "\n\n".join([p.serialize() for p in all_pos_premises])
                # msg_retrieved = "\n\n".join([f"{j}. {p.serialize()}" for j, p in enumerate(premises)])
                msg_gt = all_pos_premises
                msg_retrieved = "\n\n".join([f"{j}. {p.serialize()}" for j, p in enumerate(premises)])
                TP = len(set(premises).intersection(all_pos_premises))
                if len(all_pos_premises) == 0:
                    r = math.nan
                else:
                    r = float(TP) / len(all_pos_premises)
                msg = f"Recall@{self.num_retrieved}: {r}\n\nGround truth:\n\n```\n{msg_gt}\n```\n\nRetrieved:\n\n```\n{msg_retrieved}\n```"
                text_to_log.append([msg_gt, msg_retrieved, f"{r:.3f}"])
                # self.logger.log_text.add_text(f"premises_val", msg, self.global_step)

            all_pos_premises = set(all_pos_premises)
            if len(all_pos_premises) == 0:
                continue
            else:
                num_with_premises += 1
            first_match_found = False

            for j in range(self.num_retrieved):
                TP = len(all_pos_premises.intersection(premises[: (j + 1)]))
                recall[j].append(float(TP) / len(all_pos_premises))
                if premises[j] in all_pos_premises and not first_match_found:
                    MRR.append(1.0 / (j + 1))
                    first_match_found = True
            if not first_match_found:
                MRR.append(0.0)

        self.logger.log_text("val/premises", columns=text_columns_to_log, data=text_to_log)
        recall = [100 * np.mean(_) for _ in recall]

        for j in range(self.num_retrieved):
            self.log(
                f"val/Recall@{j+1}",
                recall[j],
                on_epoch=True,
                sync_dist=True,
                batch_size=num_with_premises,
            )

        self.log(
            "val/MRR",
            np.mean(MRR),
            on_epoch=True,
            sync_dist=True,
            batch_size=num_with_premises,
        )
