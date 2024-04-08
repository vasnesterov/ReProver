from reprover.retrieval.colbert_retrieval.model import ColBERTPremiseRetriever, ColBERTPremiseRetrieverLightning
from reprover.retrieval.datamodule import ColBERTRetrievalDataModule

CORPUS_PATH = "./data/leandojo_benchmark_4/corpus.jsonl"
COLLECTION_PATH = "./data/leandojo_benchmark_4/random/colbert_collection.tsv"
CHECKPOINT_PATH = "./checkpoints/colbertv2.0/"
NUM_NEGATIVES = 1
NUM_RETRIEVED = 2

DATA_PATH = "./data/leandojo_benchmark_4/random/"
INDEX_ROOT = "./experiments"


def create_colbert_retriever(
    checkpoint_path,
    collection_path,
    corpus_path,
    num_retrieved,
    index_name: str = "colbert_v1",
    experiment_name: str = "lean",
    index_root: str = INDEX_ROOT,
):
    model = ColBERTPremiseRetriever(
        index_name=index_name,
        experiment_name=experiment_name,
        checkpoint_path_or_name=checkpoint_path,
        collection=collection_path,
        # config: Optional[Union[str, ColBERTConfig]] = None,
        index_root=index_root,
        num_retrieved=num_retrieved,
    )
    model.load_corpus(corpus_path)
    return model


def test_reindex(model, batch_size: int = 64):
    model.reindex_corpus(batch_size)


def test_retrieve(model, data_path, corpus_path, num_negatives, reindex_batch_size: int = 32):
    data_module = ColBERTRetrievalDataModule(
        data_path,
        corpus_path,
        num_negatives=num_negatives,
        num_in_file_negatives=1,
        colbert_config=model.config,
        batch_size=1,
        eval_batch_size=1,
        max_seq_len=128,
        verbose=3,
        num_workers=0,
    )
    data_module.setup()

    example = data_module.ds_val.data[0]
    ctx = example["context"]

    inputs = dict(
        state=[ctx.state],
        file_name=[ctx.path],
        theorem_full_name=[ctx.theorem_full_name],
        theorem_pos=[ctx.theorem_pos],
    )

    ranking, premises, scores = model.retrieve(**inputs, k=2, reindex_batch_size=reindex_batch_size, do_reindex=False)
    return ranking, premises, scores, data_module


def test_retrieve_from_preprocessed(model, data_module):
    batch = next(iter(data_module.val_dataloader()))
    return model.retrieve_from_preprocessed(batch)


def test_lightning_like_interface(
    checkpoint_path,
    num_retrieved,
    index_name: str = "colbert_v1",
    experiment_name: str = "lean",
    index_root=INDEX_ROOT,
):
    model_pl = ColBERTPremiseRetrieverLightning(
        index_name=index_name,
        experiment_name=experiment_name,
        checkpoint_path_or_name=checkpoint_path,
        collection=COLLECTION_PATH,
        index_root=index_root,
        num_retrieved=num_retrieved,
    )
    model_pl.load_corpus(CORPUS_PATH)
    return model_pl


def main():
    # print("Creating colbert retriever...")
    # model = create_colbert_retriever(CHECKPOINT_PATH, COLLECTION_PATH, CORPUS_PATH, NUM_RETRIEVED)

    # print("Testing reindexing")
    # test_reindex(model)

    # print("Testing retrieve()")
    # ranking, premises, scores, data_module = test_retrieve(model, DATA_PATH, CORPUS_PATH, NUM_NEGATIVES)

    # print("Testing retrieve_from_preprocessed()")
    # test_retrieve_from_preprocessed(model, data_module)

    print("Testing Lightning-like model creation")
    model = test_lightning_like_interface(CHECKPOINT_PATH, NUM_RETRIEVED)


if __name__ == "__main__":
    main()
