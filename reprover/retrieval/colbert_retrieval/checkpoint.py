import torch
from colbert.modeling.checkpoint import _stack_3D_tensors
from colbert.modeling.colbert import ColBERT
from colbert.modeling.tokenization import DocTokenizer, QueryTokenizer
from colbert.utils.amp import MixedPrecisionManager
from tqdm import tqdm


class TrainingCheckpoint(ColBERT):
    def __init__(self, name, colbert_config=None, verbose: int = 3):
        super().__init__(name, colbert_config)

        self.verbose = verbose

        self.query_tokenizer = QueryTokenizer(self.colbert_config, verbose=self.verbose)
        self.doc_tokenizer = DocTokenizer(self.colbert_config)

        assert len(self.query_tokenizer.tok) == len(self.doc_tokenizer.tok)
        assert self.query_tokenizer.tok.mask_token is not None
        assert colbert_config.query_token in self.query_tokenizer.tok.get_vocab()
        assert colbert_config.doc_token in self.doc_tokenizer.tok.get_vocab()

        doc_token_id_query = self.query_tokenizer.tok.convert_tokens_to_ids(self.colbert_config.doc_token)
        doc_token_id_doc = self.doc_tokenizer.tok.convert_tokens_to_ids(self.colbert_config.doc_token)
        assert doc_token_id_doc == doc_token_id_query

        query_token_id_query = self.query_tokenizer.tok.convert_tokens_to_ids(self.colbert_config.query_token)
        query_token_id_doc = self.doc_tokenizer.tok.convert_tokens_to_ids(self.colbert_config.query_token)
        assert query_token_id_doc == query_token_id_query

        self.bert.resize_token_embeddings(len(self.query_tokenizer.tok))
        self.amp_manager = MixedPrecisionManager(True)

    def query(self, *args, to_cpu=False, **kw_args):
        Q = super().query(*args, **kw_args)
        return Q.cpu() if to_cpu else Q

    def doc(self, *args, to_cpu=False, **kw_args):
        D = super().doc(*args, **kw_args)

        if to_cpu:
            return (D[0].cpu(), *D[1:]) if isinstance(D, tuple) else D.cpu()

        return D

    @torch.no_grad
    def queryFromText(self, queries, bsize=None, to_cpu=False, context=None, full_length_search=False):
        if bsize:
            batches = self.query_tokenizer.tensorize(
                queries, context=context, bsize=bsize, full_length_search=full_length_search
            )

            batches = [self.query(input_ids, attention_mask, to_cpu=to_cpu) for input_ids, attention_mask in batches]
            return torch.cat(batches)

        input_ids, attention_mask = self.query_tokenizer.tensorize(
            queries, context=context, full_length_search=full_length_search
        )
        return self.query(input_ids, attention_mask)

    @torch.no_grad
    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False, showprogress=False, return_tokens=False):
        assert keep_dims in [True, False, "flatten"]

        if bsize:
            text_batches, reverse_indices = self.doc_tokenizer.tensorize(docs, bsize=bsize)

            returned_text = []
            if return_tokens:
                returned_text = [text for batch in text_batches for text in batch[0]]
                returned_text = [returned_text[idx] for idx in reverse_indices.tolist()]
                returned_text = [returned_text]

            keep_dims_ = "return_mask" if keep_dims == "flatten" else keep_dims
            batches = [
                self.doc(input_ids, attention_mask, keep_dims=keep_dims_, to_cpu=to_cpu, to_half=True)
                for input_ids, attention_mask in tqdm(text_batches, disable=not showprogress)
            ]

            if keep_dims is True:
                D = _stack_3D_tensors(batches)
                return (D[reverse_indices], *returned_text)

            elif keep_dims == "flatten":
                D, mask = [], []

                for D_, mask_ in batches:
                    D.append(D_)
                    mask.append(mask_)

                D, mask = torch.cat(D)[reverse_indices], torch.cat(mask)[reverse_indices]

                doclens = mask.squeeze(-1).sum(-1).tolist()

                D = D.view(-1, self.colbert_config.dim)
                D = D[mask.bool().flatten()].cpu()

                return (D, doclens, *returned_text)

            assert keep_dims is False

            D = [d for batch in batches for d in batch]
            return ([D[idx] for idx in reverse_indices.tolist()], *returned_text)

        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)
