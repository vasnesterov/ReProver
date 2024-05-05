import os
import re
import sys
import json
import random
import torch
import tempfile
import networkx as nx
from loguru import logger
from lean_dojo import Pos
import pytorch_lightning as pl
from dataclasses import dataclass, field
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from transformers import get_cosine_schedule_with_warmup
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from typing import Optional, List, Dict, Any, Tuple, Generator
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy

Example = Dict[str, Any]
Batch = Dict[str, Any]

MARK_START_SYMBOL = "<a>"
MARK_END_SYMBOL = "</a>"


def remove_marks(s: str) -> str:
    """Remove all :code:`<a>` and :code:`</a>` from ``s``."""
    return s.replace(MARK_START_SYMBOL, "").replace(MARK_END_SYMBOL, "")


@dataclass(unsafe_hash=True)
class Context:
    """Contexts are "queries" in our retrieval setup."""

    module: str
    theorem_full_name: str
    state: str

    def __post_init__(self) -> None:
        assert isinstance(self.module, str)
        assert isinstance(self.theorem_full_name, str)
        assert (
            isinstance(self.state, str)
            # and "⊢" in self.state
            and MARK_START_SYMBOL not in self.state
            and MARK_END_SYMBOL not in self.state
        ), self.state

    def serialize(self) -> str:
        """Serialize the context into a string for Transformers."""
        return self.state


@dataclass(unsafe_hash=True)
class Premise:
    """Premises are "documents" in our retrieval setup."""

    module: str
    """The Lean Module this premise comes from.
    """

    full_name: str
    """Fully qualified name.
    """

    code: str = field(compare=False)
    """Raw, human-written code for defining the premise.
    """

    kind: str

    @classmethod
    def from_dict(cls, dct):
        return cls(
            module = dct['module'],
            full_name = dct['name'],
            code = dct['pp'],
            kind = dct['kind'],
        )

    def __post_init__(self) -> None:
        assert isinstance(self.module, str)
        assert isinstance(self.full_name, str)
        assert isinstance(self.kind, str)
        assert isinstance(self.code, str) and self.code != ""

    def serialize(self) -> str:
        """Serialize the premise into a string for Transformers."""
        annot_full_name = f"{MARK_START_SYMBOL}{self.full_name}{MARK_END_SYMBOL}"
        code = self.code.replace(f"_root_.{self.full_name}", annot_full_name)
        # fields = self.full_name.split(".")

        # for i in range(len(fields)):
        #     prefix = ".".join(fields[i:])
        #     new_code = re.sub(f"(?<=\s)«?{prefix}»?", annot_full_name, code)
        #     if new_code != code:
        #         code = new_code
        #         break

        return code

def load_available_premises(json_path):
    data = json.load(open(json_path))

    result = dict()
    for key in data:
        result[key] = {
            'inFilePremises': [Premise.from_dict(dct) for dct in data[key]['inFilePremises']],
            'outFilePremises': [Premise.from_dict(dct) for dct in data[key]['outFilePremises']]
        }
    return result

class PremiseSet:
    """A set of premises indexed by their modules and full names."""

    module2premises: Dict[str, Dict[str, Premise]]

    def __init__(self) -> None:
        self.module2premises = {}

    def __iter__(self) -> Generator[Premise, None, None]:
        for _, premises in self.module2premises.items():
            for p in premises.values():
                yield p

    def add(self, p: Premise) -> None:
        if p.module in self.module2premises:
            self.module2premises[p.module][p.full_name] = p
        else:
            self.module2premises[p.module] = {p.full_name: p}

    def update(self, premises: List[Premise]) -> None:
        for p in premises:
            self.add(p)

    def __contains__(self, p: Premise) -> bool:
        return (
            p.module in self.module2premises and p.full_name in self.module2premises[p.module]
        )

    def __len__(self) -> int:
        return sum(len(premises) for premises in self.module2premises.values())


@dataclass(frozen=True)
class Module:
    """A module defines 0 or multiple premises."""

    name: str
    """Name of module
    """

    premises: List[Premise]
    """A list of premises defined in this file.
    """

    @property
    def is_empty(self) -> bool:
        """Check whether the module contains no premise."""
        return self.premises == []


class Corpus:
    """Our retrieval corpus is a DAG of files. Each file consists of
    premises (theorems, definitoins, etc.) that can be retrieved.
    """

    transitive_dep_graph: nx.DiGraph
    """Transitive closure of the dependency graph among files. 
    There is an edge from file X to Y iff X import Y (directly or indirectly).
    """

    all_premises: List[Premise]
    """All premises in the entire corpus.
    """

    

    def __init__(self, jsonl_path: str, dot_imports_path: str, json_in_file_premises_path: str) -> None:
        """Construct a :class:`Corpus` object from a ``corpus.jsonl`` data file."""
        dep_graph = self._load_digraph_from_dot(dot_imports_path)
        self.in_file_premises = load_available_premises(json_in_file_premises_path)
        self.all_premises = []

        logger.info(f"Building the corpus from {jsonl_path}")

        mod2premises = dict()

        for line in open(jsonl_path):
            premise_data = json.loads(line)
            module = premise_data['module']
            premise = Premise(
                module=module,
                kind=premise_data['kind'],
                full_name=premise_data['name'],
                code=premise_data['pp']
            )
            self.all_premises.append(premise)
            if module not in mod2premises:
                mod2premises[module] = []
            mod2premises[module].append(premise)

        self.module_names = []
        for mod_name in dep_graph.nodes:
            module = Module(
                name=mod_name,
                premises=mod2premises[mod_name] if mod_name in mod2premises else []
            )
            # print(f"{mod_name}: {len(mod2premises[mod_name] if mod_name in mod2premises else [])} premises")
            dep_graph.nodes[mod_name]['module'] = module
            self.module_names.append(module)

        assert nx.is_directed_acyclic_graph(dep_graph)
        self.transitive_dep_graph = nx.transitive_closure_dag(dep_graph)

        self.imported_premises_cache = {}
        self.fill_cache()

    def _load_digraph_from_dot(self, dot_file_path: str):
        digraph = nx.DiGraph()
        
        with open(dot_file_path, 'r') as dot_file:
            for line in dot_file:
                if '->' in line:
                    nodes = line.strip().split('->')
                    source = nodes[0].strip().strip('"')
                    target = nodes[1].strip().rstrip(';').strip('"')  # Remove trailing semicolon
                    digraph.add_edge(source, target)
        
        return digraph.reverse()

    def _get_module(self, name: str) -> Module:
        return self.transitive_dep_graph.nodes[name]["module"]

    def __len__(self) -> int:
        return len(self.all_premises)

    def __contains__(self, path: str) -> bool:
        return path in self.transitive_dep_graph

    def __getitem__(self, idx: int) -> Premise:
        return self.all_premises[idx]

    @property
    def modules(self) -> List[Module]:
        return [self._get_module(p) for p in self.transitive_dep_graph.nodes]

    @property
    def num_modules(self) -> int:
        return len(self.module_names)

    def get_dependencies(self, module: str) -> List[str]:
        """Return a list of (direct and indirect) dependencies of the ``module``."""
        return list(self.transitive_dep_graph.successors(module))

    def get_premises(self, module: str) -> List[Premise]:
        """Return a list of premises defined in the``module``."""
        return self._get_module(module).premises

    def num_premises(self, module: str) -> int:
        """Return the number of premises defined in the ``module``."""
        return len(self.get_premises(path))

    # def locate_premise(self, path: str, pos: Pos) -> Optional[Premise]:
    #     """Return a premise at position ``pos`` in file ``path``.

    #     Return None if no such premise can be found.
    #     """
    #     for p in self.get_premises(path):
    #         assert p.path == path
    #         if p.start <= pos <= p.end:
    #             return p
    #     return None

    def fill_cache(self) -> None:
        for path in self.transitive_dep_graph.nodes:
            self.get_imported_premises(path)

    def get_imported_premises(self, module: str) -> List[Premise]:
        """Return a list of premises imported in ``module``. The result is cached."""
        premises = self.imported_premises_cache.get(module, None)
        if premises is not None:
            return premises

        premises = []
        for m in self.transitive_dep_graph.successors(module):
            # print(f"{module} -> {m}")
            premises.extend(self._get_module(m).premises)
        self.imported_premises_cache[module] = premises
        return premises

    # def get_accessible_premises(self, path: str, pos: Pos) -> PremiseSet:
    #     """Return the set of premises accessible at position ``pos`` in file ``path``,
    #     i.e., all premises defined in the (transitively) imported files or earlier in the same file.
    #     """
    #     premises = PremiseSet()
    #     for p in self.get_premises(path):
    #         if p.end <= pos:
    #             premises.add(p)
    #     premises.update(self.get_imported_premises(path))
    #     return premises

    def get_imported_premise_indexes(self, module: str, pos: Pos) -> List[int]:
        return [
            i
            for i, p in enumerate(self.all_premises)
            if self.transitive_dep_graph.has_edge(module, p.module)
        ]

    def get_nearest_premises(
        self,
        premise_embeddings: torch.FloatTensor,
        batch_context: List[Context],
        batch_context_emb: torch.Tensor,
        k: int,
        similarities=None,
    ) -> Tuple[List[List[Premise]], List[List[float]]]:
        """Perform a batch of nearest neighbour search."""
        if similarities is None:
            similarities = batch_context_emb @ premise_embeddings.t()
        idxs_batch = similarities.argsort(dim=1, descending=True).tolist()
        results = [[] for _ in batch_context]
        scores = [[] for _ in batch_context]

        for j, (ctx, idxs) in enumerate(zip(batch_context, idxs_batch)):
            accessible_premises = set(self.get_imported_premises(ctx.module) + self.in_file_premises[ctx.theorem_full_name]["inFilePremises"])
            for i in idxs:
                p = self.all_premises[i]
                if p in accessible_premises:
                    results[j].append(p)
                    scores[j].append(similarities[j, i].item())
                    if len(results[j]) >= k:
                        
                        break
            else:
                print(ctx.module, ctx.theorem_full_name)
                print(len(accessible_premises))
                print(len(self.get_imported_premises(ctx.module)))
                raise ValueError

        return results, scores


@dataclass(frozen=True)
class IndexedCorpus:
    """A corpus with premise embeddings."""

    corpus: Corpus
    embeddings: torch.FloatTensor

    def __post_init__(self):
        assert self.embeddings.device == torch.device("cpu")
        assert len(self.embeddings) == len(self.corpus)

def path_to_module(path: str) -> str:
    path = re.sub(r".lake/packages/\w*?/", "/", path)
    normalized_path = os.path.normpath(path)
    
    parts = []
    for part in normalized_path.split(os.path.sep):
        if part and part != "." and part != "..":
            parts.append(part)    

    assert parts[-1].endswith(".lean")
    parts[-1] = parts[-1].replace(".lean", "")
    return ".".join(parts)

def get_all_pos_premises(annot_tac, corpus: Corpus) -> List[Premise]:
    """Return a list of all premises that are used in the tactic ``annot_tac``."""
    _, provenances = annot_tac
    all_pos_premises = set()

    for prov in provenances:
        def_path = prov["def_path"]
        p = corpus.locate_premise(def_path, Pos(*prov["def_pos"]))
        if p is not None:
            all_pos_premises.add(p)
        else:
            logger.warning(f"Cannot locate premise: {prov}")

    return list(all_pos_premises)


_SPACES_REGEX = re.compile(r"\s+", re.DOTALL)


def normalize_spaces(s: str) -> str:
    """Repalce any consecutive block of whitespace characters in ``s`` with a single whitespace."""
    return _SPACES_REGEX.sub(" ", s).strip()


def format_tactic(annot_tac: str, provenances, normalize: bool) -> str:
    """Use full names for the all <a>...</a>."""
    if normalize:
        annot_tac = normalize_spaces(annot_tac)
    if len(provenances) == 0:
        return annot_tac

    tac = ""
    marks = list(re.finditer(r"<a>(?P<ident>.+?)</a>", annot_tac))

    for i, (m, prov) in enumerate(zip_strict(marks, provenances)):
        last_end = marks[i - 1].end() if i > 0 else 0
        tac += annot_tac[last_end : m.start()] + "<a>" + prov["full_name"] + "</a>"

    tac += annot_tac[marks[-1].end() :]
    return tac


def format_state(s: str) -> str:
    m = re.match(r"\d+ goals", s)
    if m is not None:
        return s[m.end() :].strip()
    else:
        return s

def _format_augmented_state(
    s: str, premises: List[Premise], max_len: int, p_drop: float
) -> Tuple[str, int]:
    """
    Format a state with retrieved premises and drop some of them with probability ``p_drop``.
    Returns the augmented state and the number of included premises
    """
    s = format_state(s)

    aug_s = ""
    length = 0
    max_premises_len = max_len - len(bytes(s.encode("utf-8")))

    cnt_premises = 0
    for p in premises:
        if random.random() < p_drop:
            continue
        cnt_premises += 1
        p_str = f"{p.serialize()}\n\n"
        l = len(bytes(p_str.encode("utf-8")))
        if length + l > max_premises_len:
            break
        length += l
        aug_s = p_str + aug_s

    aug_s += s
    return aug_s, cnt_premises

def format_augmented_state(
    s: str, premises: List[Premise], max_len: int, p_drop: float
) -> str:
    """Format a state with retrieved premises and drop some of them with probability ``p_drop``."""
    return _format_augmented_state(s, premises, max_len, p_drop)[0]


def get_optimizers(
    parameters, trainer: pl.Trainer, lr: float, warmup_steps: int
) -> Dict[str, Any]:
    """Return an AdamW optimizer with cosine warmup learning rate schedule."""
    strategy = trainer.strategy

    if isinstance(strategy, DeepSpeedStrategy):
        if "offload_optimizer" in strategy.config["zero_optimization"]:
            logger.info("Optimizing with DeepSpeedCPUAdam")
            optimizer = DeepSpeedCPUAdam(parameters, lr=lr, adamw_mode=True)
        else:
            logger.info("Optimizing with FusedAdam")
            optimizer = FusedAdam(parameters, lr=lr, adam_w_mode=True)
    else:
        logger.info("Optimizing with AdamW")
        optimizer = torch.optim.AdamW(parameters, lr=lr)

    if trainer.max_steps != -1:
        max_steps = trainer.max_steps
    else:
        assert trainer.max_epochs is not None
        max_steps = (
            trainer.max_epochs
            * len(trainer.datamodule.train_dataloader())
            // trainer.accumulate_grad_batches
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
        },
    }


def _is_deepspeed_checkpoint(path: str):
    if not os.path.exists(path):
        raise FileExistsError(f"Checkpoint {path} does not exist.")
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "zero_to_fp32.py"))


def load_checkpoint(model_cls, ckpt_path: str, device, freeze: bool):
    """Handle DeepSpeed checkpoints in model loading."""
    if not _is_deepspeed_checkpoint(ckpt_path):
        model = model_cls.load_from_checkpoint(ckpt_path, strict=False).to(device)
    else:
        with tempfile.TemporaryDirectory() as dirname:
            path = os.path.join(dirname, "lightning.cpkt")
            convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, path)
            model = model_cls.load_from_checkpoint(path, strict=False)
            model = model.to(device)
    if freeze:
        model.freeze()
    return model


def zip_strict(*args):
    assert len(args) > 1 and all(len(args[0]) == len(a) for a in args[1:])
    return zip(*args)


def set_logger(verbose: bool) -> None:
    """
    Set the logging level of loguru.
    The effect of this function is global, and it should
    be called only once in the main function
    """
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")


def cpu_checkpointing_enabled(pl_module) -> bool:
    try:
        trainer = pl_module.trainer
        return (
            trainer.strategy is not None
            and isinstance(trainer.strategy, DeepSpeedStrategy)
            and trainer.strategy.config["activation_checkpointing"]["cpu_checkpointing"]
        )
    except RuntimeError:
        return False