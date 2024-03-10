"""Lightning module for the tactic generator."""

#  import openai
import pickle
import re
import time
from abc import ABC, abstractmethod
from subprocess import CalledProcessError
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from lean_dojo import Pos
from lean_dojo.utils import execute
from loguru import logger
from reprover.common import (
    IndexedCorpus,
    _format_augmented_state,
    format_augmented_state,
    get_optimizers,
    load_checkpoint,
    remove_marks,
    zip_strict,
)
from reprover.retrieval.model import PremiseRetriever
from torchmetrics import Metric
from transformers import AutoTokenizer, T5ForConditionalGeneration

torch.set_float32_matmul_precision("medium")


class TopkAccuracy(Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, batch_preds: List[List[str]], batch_gt: List[str]):
        assert len(batch_preds) == len(batch_gt)
        for preds, gt in zip(batch_preds, batch_gt):
            # This still doesn't account for short names vs. full names.
            gt = remove_marks(gt)
            preds = [remove_marks(p) for p in preds]
            self.correct += gt in preds[: self.k]
        self.total += len(batch_gt)

    def compute(self) -> float:
        return self.correct.float() / self.total


class TacticGenerator(ABC):
    """A tactic generator takes a state and generates multiple tactic candidates."""

    @abstractmethod
    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        raise NotImplementedError


class RetrievalAugmentedGenerator(TacticGenerator, pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        num_beams: int,
        eval_num_retrieved: int,
        eval_num_cpus: int,
        eval_num_theorems: int,
        max_seq_len: int,
        length_penalty: float = 0.0,
        ret_ckpt_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.eval_num_retrieved = eval_num_retrieved
        self.eval_num_cpus = eval_num_cpus
        self.eval_num_theorems = eval_num_theorems
        self.max_seq_len = max_seq_len

        if ret_ckpt_path is None:
            logger.info("Without retrieval")
            self.retriever = None
        else:
            logger.info(f"Loading the retriever from {ret_ckpt_path}")
            self.retriever = PremiseRetriever.load(ret_ckpt_path, self.device, freeze=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = T5ForConditionalGeneration.from_pretrained(model_name)

        self.topk_accuracies = dict()
        for k in range(1, num_beams + 1):
            acc = TopkAccuracy(k)
            self.topk_accuracies[k] = acc
            self.add_module(f"top{k}_acc_val", acc)

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool) -> "RetrievalAugmentedGenerator":
        return load_checkpoint(cls, ckpt_path, device, freeze)

    def forward(
        self,
        state_ids: torch.Tensor,
        state_mask: torch.Tensor,
        tactic_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.generator(
            input_ids=state_ids,
            attention_mask=state_mask,
            labels=tactic_ids,
        ).loss

    ############
    # Training #
    ############

    def training_step(self, batch, batch_idx: int):
        loss = self(
            batch["state_ids"],
            batch["state_mask"],
            batch["tactic_ids"],
        )
        self.log(
            "loss_train",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
        )
        self._log_io_texts("train", batch["state_ids"], batch["tactic_ids"])
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(self.parameters(), self.trainer, self.lr, self.warmup_steps)

    def _log_io_texts(
        self,
        split: str,
        state_ids: torch.LongTensor,
        tactic_ids: torch.LongTensor,
    ) -> None:
        tb = self.logger.experiment
        inp = self.tokenizer.decode(state_ids[0], skip_special_tokens=True)
        oup_ids = torch.where(tactic_ids[0] == -100, self.tokenizer.pad_token_id, tactic_ids[0])
        oup = self.tokenizer.decode(oup_ids, skip_special_tokens=True)
        tb.add_text(f"{split}_state", f"```\n{inp}\n```", self.global_step)
        tb.add_text(f"{split}_tactic", f"`{oup}`", self.global_step)

    def on_fit_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
            assert self.trainer is not None
            logger.info(f"Logging to {self.trainer.log_dir}")

        if self.retriever is not None:
            self.retriever.load_corpus(self.trainer.datamodule.corpus)

    ##############
    # Validation #
    ##############

    def validation_step(self, batch: Dict[str, Any], _) -> None:
        state_ids = batch["state_ids"]
        state_mask = batch["state_mask"]
        tactic_ids = batch["tactic_ids"]

        loss = self(state_ids, state_mask, tactic_ids)

        self.log(f"loss_val", loss, on_step=False, on_epoch=True, sync_dist=True)
        self._log_io_texts("val", state_ids, tactic_ids)

        # Generate topk tactic candidates via Beam Search.
        output = self.generator.generate(
            input_ids=state_ids,
            attention_mask=state_mask,
            max_length=self.max_seq_len,
            num_beams=self.num_beams,
            do_sample=False,
            num_return_sequences=self.num_beams,
            early_stopping=False,
        )
        output_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        batch_size = state_ids.size(0)
        assert len(output_text) == batch_size * self.num_beams
        tactics_pred = [output_text[i * self.num_beams : (i + 1) * self.num_beams] for i in range(batch_size)]

        tb = self.logger.experiment
        msg = "\n".join(tactics_pred[0])
        tb.add_text(f"preds_val", f"```\n{msg}\n```", self.global_step)

        # Log the topk accuracies.
        for k in range(1, self.num_beams + 1):
            topk_acc = self.topk_accuracies[k]
            topk_acc(tactics_pred, batch["tactic"])
            self.log(f"top{k}_acc_val", topk_acc, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        ckpt_path = f"{self.trainer.log_dir}/checkpoints/last.ckpt"
        self.trainer.save_checkpoint(ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")

        # data_path = self.trainer.datamodule.data_path
        # if self.retriever is None:
        #     cmd = f"python prover/evaluate.py --data-path {data_path} --num-cpus {self.eval_num_cpus} --num-theorems {self.eval_num_theorems} --ckpt_path {ckpt_path}"
        # else:
        #     self.retriever.reindex_corpus(self.trainer.datamodule.eval_batch_size)
        #     corpus_path = f"{self.trainer.log_dir}/checkpoints/indexed_corpus.pickle"
        #     pickle.dump(
        #         IndexedCorpus(
        #             self.retriever.corpus, self.retriever.corpus_embeddings.cpu()
        #         ),
        #         open(corpus_path, "wb"),
        #     )
        #     cmd = f"python prover/evaluate.py --data-path {data_path} --num-cpus {self.eval_num_cpus} --num-theorems {self.eval_num_theorems} --ckpt_path {ckpt_path} --indexed-corpus-path {corpus_path}"

        # logger.info(cmd)

        # wait_time = 3600
        # while True:
        #     try:
        #         _, err = execute(cmd, capture_output=True)
        #         break
        #     except CalledProcessError as ex:
        #         logger.error(ex)
        #         logger.error(
        #             f"Failed to evaluate. Retrying in {wait_time / 3600} hour..."
        #         )
        #         time.sleep(wait_time)
        #         wait_time *= 2

        # m = re.search(r"Pass@1: (\S+)", err)
        # assert m is not None, err
        # acc = float(m.group(1))
        # self.log("Pass@1_val", acc, on_step=False, on_epoch=True)
        # logger.info(f"Pass@1: {acc}")

    ##############
    # Prediction #
    ##############

    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        return self.batch_generate([state], [file_path], [theorem_full_name], [theorem_pos], num_samples)[0]

    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        logger.debug(state)
        if self.retriever is not None:
            retrieved_premises, _ = self.retriever.retrieve(
                state,
                file_path,
                theorem_full_name,
                theorem_pos,
                self.eval_num_retrieved,
            )
            state = [
                format_augmented_state(s, premises, self.max_seq_len, p_drop=0.0)
                for s, premises in zip_strict(state, retrieved_premises)
            ]

        tokenized_state = self.tokenizer(
            state,
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        state_ids = tokenized_state.input_ids.to(self.device)
        state_mask = tokenized_state.attention_mask.to(self.device)

        # Generate tactic candidates using beam search.
        logger.debug(f"generation, {self.max_seq_len}")
        output = self.generator.generate(
            input_ids=state_ids,
            attention_mask=state_mask,
            max_length=self.max_seq_len,
            num_beams=num_samples,
            length_penalty=self.length_penalty,
            do_sample=False,
            num_return_sequences=num_samples,
            early_stopping=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Return the output.
        raw_output_text = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        raw_scores = output.sequences_scores.tolist()
        tactics_with_scores = []

        for i in range(len(state)):
            output_text = []
            output_score = []

            for j in range(i * num_samples, (i + 1) * num_samples):
                t = remove_marks(raw_output_text[j])
                if t not in output_text:
                    output_text.append(t)
                    output_score.append(raw_scores[j])

            tactics_with_scores.append(list(zip_strict(output_text, output_score)))

        return tactics_with_scores


class RMTRetrievalAugmentedGenerator(RetrievalAugmentedGenerator):
    def __init__(
        self,
        backbone_model_name: str,
        num_memory_tokens: int,
        num_segments: int,
        lr: float,
        warmup_steps: int,
        num_beams: int,
        eval_num_retrieved: int,
        eval_num_cpus: int,
        eval_num_theorems: int,
        max_seq_len: int,
        length_penalty: float = 0.0,
        ret_ckpt_path: Optional[str] = None,
        skip_test_proving: bool = True,
        skip_topk: bool = False,
    ) -> None:
        self.num_memory_tokens = num_memory_tokens
        self.num_segments = num_segments
        self.max_nonmemory_seq_len = max_seq_len - num_memory_tokens

        super().__init__(
            backbone_model_name,
            lr,
            warmup_steps,
            num_beams,
            eval_num_retrieved,
            eval_num_cpus,
            eval_num_theorems,
            self.max_nonmemory_seq_len,
            length_penalty,
            ret_ckpt_path,
        )

        self.memory_emb = torch.nn.Embedding(self.num_memory_tokens, self.generator.model_dim)
        # self.memory_linear = torch.nn.Linear(self.generator.model_dim, self.generator.model_dim, bias=True)

        self.skip_test_proving = skip_test_proving
        self.skip_topk = skip_topk

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool) -> "RMTRetrievalAugmentedGenerator":
        return load_checkpoint(cls, ckpt_path, device, freeze)

    def _encode_with_memory(
        self,
        state_ids: List[torch.Tensor],  # state_ids[seg][idx in batch][token] = emb vector
        state_mask: List[torch.Tensor],
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes a list of segments and compute the output of encoder with memory.
        Returns this output and attention mask of the encoder on the last segment.
        """
        return_dict = dict()

        # Expand embeddings
        state_embeds = []
        num_segments = len(state_ids)
        new_state_mask = []
        batch_size = state_ids[0].shape[0]
        for segment in range(num_segments):
            memory_state_shape = (batch_size, self.num_memory_tokens, self.generator.model_dim)
            memory_mask_shape = (batch_size, self.num_memory_tokens)
            device = state_ids[segment].device
            state_embeds.append(
                torch.cat(
                    (
                        # torch.zeros(memory_state_shape, device=device), # memory: zeros
                        self.memory_emb.weight.repeat(batch_size, 1, 1),  # memory: trainable embedding
                        self.generator.shared(state_ids[segment]),
                    ),
                    dim=-2,
                )
            )
            new_state_mask.append(
                torch.cat(
                    (
                        torch.ones(memory_mask_shape, device=device),  # memory
                        state_mask[segment],
                    ),
                    dim=-1,
                )
            )

        # Compute memory
        attentions = []
        for segment in range(num_segments):
            enc_out = self.generator.encoder(
                inputs_embeds=state_embeds[segment],
                attention_mask=new_state_mask[segment],
                output_attentions=output_attentions,
            )
            if segment < num_segments - 1:
                state_embeds[segment + 1][:, : self.num_memory_tokens, :] = enc_out["last_hidden_state"][
                    :, : self.num_memory_tokens, :
                ]
            if output_attentions:
                attentions.append(enc_out.attentions)

        return_dict["encoder_output"] = enc_out
        return_dict["encoder_mask"] = new_state_mask[-1]
        if output_attentions:
            return_dict["attentions"] = attentions

        return return_dict

    def forward(
        self,
        state_ids: List[torch.Tensor],  # state_ids[seg][idx in batch][token] = emb vector
        state_mask: List[torch.Tensor],
        tactic_ids: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        enc_dict = self._encode_with_memory(state_ids, state_mask, output_attentions=output_attentions)
        enc_out = enc_dict["encoder_output"]
        last_enc_mask = enc_dict["encoder_mask"]
        # decoder_input_ids = tactic_ids
        # decoder_input_ids[decoder_input_ids == -100] = self.tokenizer.pad_token_id
        out = self.generator(
            # input_ids=decoder_input_ids,
            labels=tactic_ids,
            encoder_outputs=enc_out,
            attention_mask=last_enc_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            out.encoder_attentions = enc_dict["attentions"]
        return out

    ############
    # Training #
    ############

    def training_step(self, batch, batch_idx: int):
        loss = self(
            batch["state_ids"],
            batch["state_mask"],
            batch["tactic_ids"],
        ).loss
        self.log(
            "loss_train",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
        )
        # self._log_io_texts("train", batch["state_ids"], batch["tactic_ids"])
        return loss

    def _log_io_texts(
        self,
        split: str,
        state_ids: List[torch.LongTensor],
        tactic_ids: torch.LongTensor,
    ) -> None:
        tb = self.logger.experiment
        inp = self.tokenizer.decode(state_ids[-1][0], skip_special_tokens=True)
        oup_ids = torch.where(tactic_ids[0] == -100, self.tokenizer.pad_token_id, tactic_ids[0])
        oup = self.tokenizer.decode(oup_ids, skip_special_tokens=True)

        tb.log({"state": inp, "tactic": oup, "split": split, "global step": self.global_step})
        # tb.add_text(f"{split}_state", f"```\n{inp}\n```", self.global_step)
        # tb.add_text(f"{split}_tactic", f"`{oup}`", self.global_step)

    ##############
    # Validation #
    ##############

    def validation_step(self, batch: Dict[str, Any], _) -> None:
        state_ids = batch["state_ids"]
        state_mask = batch["state_mask"]
        tactic_ids = batch["tactic_ids"]

        loss = self(state_ids, state_mask, tactic_ids).loss
        self.log("loss_val", loss, on_step=False, on_epoch=True, sync_dist=True)
        self._log_io_texts("val", state_ids, tactic_ids)

        # Generate topk tactic candidates via Beam Search.
        if not self.skip_topk:
            enc_dict = self._encode_with_memory(state_ids, state_mask)
            enc_out = enc_dict["encoder_output"]
            last_enc_mask = enc_dict["encoder_mask"]
            output = self.generator.generate(
                encoder_outputs=enc_out,
                attention_mask=last_enc_mask,
                max_length=self.max_seq_len,
                num_beams=self.num_beams,
                do_sample=False,
                num_return_sequences=self.num_beams,
                early_stopping=False,
            )
            output_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            batch_size = state_ids[0].size(0)
            assert len(output_text) == batch_size * self.num_beams
            tactics_pred = [output_text[i * self.num_beams : (i + 1) * self.num_beams] for i in range(batch_size)]

            # tb = self.logger.experiment
            # msg = "\n".join(tactics_pred[0])
            # tb.log({"preds_val": msg}, self.global_step)

            # Log the topk accuracies.
            for k in range(1, self.num_beams + 1):
                topk_acc = self.topk_accuracies[k]
                topk_acc(tactics_pred, batch["tactic"])
                self.log(f"top{k}_acc_val", topk_acc, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        # ckpt_path = f"{self.trainer.log_dir}/checkpoints/last.ckpt"
        # self.trainer.save_checkpoint(ckpt_path)
        # logger.info(f"Saved checkpoint to {ckpt_path}")

        if not self.skip_test_proving:
            data_path = self.trainer.datamodule.data_path
            if self.retriever is None:
                cmd = f"python prover/evaluate.py --data-path {data_path} --num-cpus {self.eval_num_cpus} --num-theorems {self.eval_num_theorems} --ckpt_path {ckpt_path}"
            else:
                self.retriever.reindex_corpus(self.trainer.datamodule.eval_batch_size)
                corpus_path = f"{self.trainer.log_dir}/checkpoints/indexed_corpus.pickle"
                pickle.dump(
                    IndexedCorpus(self.retriever.corpus, self.retriever.corpus_embeddings.cpu()),
                    open(corpus_path, "wb"),
                )
                cmd = f"python prover/evaluate.py --data-path {data_path} --num-cpus {self.eval_num_cpus} --num-theorems {self.eval_num_theorems} --ckpt_path {ckpt_path} --indexed-corpus-path {corpus_path}"

            logger.info(cmd)

            wait_time = 3600
            while True:
                try:
                    _, err = execute(cmd, capture_output=True)
                    break
                except CalledProcessError as ex:
                    logger.error(ex)
                    logger.error(f"Failed to evaluate. Retrying in {wait_time / 3600} hour...")
                    time.sleep(wait_time)
                    wait_time *= 2

            m = re.search(r"Pass@1: (\S+)", err)
            assert m is not None, err
            acc = float(m.group(1))
            self.log("Pass@1_val", acc, on_step=False, on_epoch=True)
            logger.info(f"Pass@1: {acc}")

    ##############
    # Prediction #
    ##############

    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        return self.batch_generate([state], [file_path], [theorem_full_name], [theorem_pos], num_samples)[0]

    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        logger.debug(state)
        assert self.retriever is not None, "RMT is meaningless without retrieval"
        retrieved_premises, _ = self.retriever.retrieve(
            state,
            file_path,
            theorem_full_name,
            theorem_pos,
            self.eval_num_retrieved,
        )
        segments = [[] for i in range(self.num_segments)]
        for s, premises in zip_strict(state, retrieved_premises):
            used_premises = 0
            for i in range(self.num_segments):
                new_segment, new_used_premises = _format_augmented_state(
                    s,
                    premises[:used_premises],
                    self.max_nonmemory_seq_len,
                    p_drop=0.0,
                )
                segments[i].append(new_segment)
                used_premises += new_used_premises

        segments.reverse()  # best premises go in the end

        tokenized_state = [
            self.tokenizer(
                segments[i],
                padding="longest",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            )
            for i in range(self.num_segments)
        ]

        state_ids = [t.input_ids.to(self.device) for t in tokenized_state]
        state_mask = [t.attention_mask.to(self.device) for t in tokenized_state]

        enc_dict = self._encode_with_memory(state_ids, state_mask)
        enc_out = enc_dict["encoder_output"]
        last_enc_mask = enc_dict["encoder_mask"]

        # Generate tactic candidates using beam search.
        print(f"generation, {self.max_seq_len}")
        output = self.generator.generate(
            encoder_outputs=enc_out,
            attention_mask=last_enc_mask,
            max_length=self.max_seq_len,
            num_beams=num_samples,
            length_penalty=self.length_penalty,
            do_sample=False,
            num_return_sequences=num_samples,
            early_stopping=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Return the output.
        raw_output_text = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        raw_scores = output.sequences_scores.tolist()
        tactics_with_scores = []

        for i in range(len(state)):
            output_text = []
            output_score = []

            for j in range(i * num_samples, (i + 1) * num_samples):
                t = remove_marks(raw_output_text[j])
                if t not in output_text:
                    output_text.append(t)
                    output_score.append(raw_scores[j])

            tactics_with_scores.append(list(zip_strict(output_text, output_score)))

        return tactics_with_scores


# class TwoHeadRMTRetrievalAugmentedGenerator(RMTRetrievalAugmentedGenerator):


class GPT4TacticGenerator(TacticGenerator):
    def __init__(
        self,
        organization: str,
        api_key: str,
        model: str = "gpt-4",
        max_tokens: int = 1024,
        num_retries: int = 3,
        threshold: float = 0.9,
    ):
        super().__init__()
        openai.organization = organization
        openai.api_key = api_key
        self.model = model
        self.default_prompt = "You are an expert in Lean3 theorem proofs. We are trying to solve the Lean3 theorem 'THEOREM_FULL_NAME' from the mathlib file 'FILE_PATH'. The current tactic state is: 'TACTIC_STATE'. Suggest exactly NUM_SAMPLES unique tactics to progress in solving 'THEOREM_FULL_NAME', along with their confidence levels as a float between 0 and 1. Rank them in order of effectiveness. Present the tactics and their confidence levels as comma-separated tuples in this format: #(tactic_{1}, confidence_{1})#, #(tactic_{2}, confidence_{2})#, ..., #(tactic_{NUM_SAMPLES}, confidence_{NUM_SAMPLES})#."
        self.max_tokens = max_tokens
        self.num_retries = num_retries
        self.threshold = threshold

    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        prompt = (
            self.default_prompt.replace("TACTIC_STATE", state)
            .replace("FILE_PATH", file_path)
            .replace("THEOREM_FULL_NAME", theorem_full_name)
            .replace("NUM_SAMPLES", str(int(num_samples / self.threshold)))
        )
        logger.info(prompt)

        for _ in range(self.num_retries):
            response = None
            # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    # temperature=0,
                    max_tokens=self.max_tokens,
                    # stop="E:" #
                )
            except openai.error.APIError as e:
                # Handle API error here, e.g. retry or log
                logger.info(f"OpenAI API returned an API Error: {e}")
                continue
            except openai.error.APIConnectionError as e:
                # Handle connection error here
                logger.info(f"Failed to connect to OpenAI API: {e}")
                continue
            except openai.error.RateLimitError as e:
                # Handle rate limit error (we recommend using exponential backoff)
                logger.info(f"OpenAI API request exceeded rate limit: {e}")
                continue
            except Exception as e:
                logger.info(e)
                continue

            if response is None:
                continue

            logger.info(f"GPT-4 response: {response}")
            output = response["choices"][0]["message"]["content"]
            indices = []

            for i, c in enumerate(output):
                if c == "#":
                    indices.append(i)

            tactics_with_scores = []

            for i in range(1, len(indices), 2):
                tactic_and_confidence = output[indices[i - 1] + 1 : indices[i]].strip()

                try:
                    while tactic_and_confidence[0] == "(":
                        tactic_and_confidence = tactic_and_confidence[1:]

                    if tactic_and_confidence[-1] == ")":
                        tactic_and_confidence = tactic_and_confidence[:-1]

                    split_index = tactic_and_confidence.rindex(",")
                    tactic = tactic_and_confidence[:split_index].strip()
                    confidence = float(tactic_and_confidence[split_index + 1 :].strip())
                except Exception as e:
                    logger.info(e)
                    logger.info(
                        f"{self.model} output {output[indices[i-1]+1:indices[i]]} was not formatted correctly and could not be parsed."
                    )
                    continue

                tactics_with_scores.append((tactic, confidence))

            if len(tactics_with_scores) < int(self.threshold * num_samples):
                continue

            tactics_with_scores = sorted(tactics_with_scores, key=lambda x: x[1], reverse=True)[
                : min(num_samples, len(tactics_with_scores))
            ]
            logger.debug(f"GPT-4 tactics: {tactics_with_scores}")
            logger.debug(
                f"GPT-4 tactic count requested: {num_samples} / {self.threshold} = {int(num_samples / self.threshold)}"
            )
            logger.debug(f"GPT-4 tactic count received and parsed: {len(tactics_with_scores)}")
            return tactics_with_scores

        raise ValueError("GPT-4 outputs are unparsable.")

    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        return [
            self.generate(s, f, t, p, num_samples)
            for s, f, t, p in zip_strict(state, file_path, theorem_full_name, theorem_pos)
        ]


class FixedTacticGenerator(TacticGenerator):
    def __init__(self, tactic, module) -> None:
        self.tactic = tactic
        self.module = module

    def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        return [(f"{{ {self.tactic} }}", 1.0)]

    def batch_generate(
        self,
        state: List[str],
        file_path: List[str],
        theorem_full_name: List[str],
        theorem_pos: List[Pos],
        num_samples: int,
    ) -> List[List[Tuple[str, float]]]:
        return [
            self.generate(s, f, tfn, tp, num_samples)
            for s, f, tfn, tp in zip(state, file_path, theorem_full_name, theorem_pos)
        ]