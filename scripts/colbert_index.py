from colbert import Indexer
from colbert.infra import ColBERTConfig

if __name__ == "__main__":
    config = ColBERTConfig(nbits=2, root="../experiments/test")
    indexer = Indexer(checkpoint="../checkpoints/colbertv2.0", config=config)
    indexer.configure(avoid_fork_if_possible=True, nranks=1)
    print(indexer.config.avoid_fork_if_possible)
    indexer.index(
        name="test_index",
        collection="../data/leandojo_benchmark_4/novel_premises/colbert_collection_100.tsv",
        overwrite=True,
    )
