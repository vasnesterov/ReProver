from setuptools import find_packages, setup

setup(
    name="ReProver",
    version="0.1",
    author="",
    description="Automated Theorem Prover for lean based on ReProver",
    packages=find_packages(),
    python_requires=">=3.10",
    # install_requires=[
    #     "transformers>=4.17",
    #     "pytorch-lightning>=1.6.3",
    #     "torch>=1.9.1",
    #     "ray",
    #     "jsonargparse==4.17.0",
    #     "spacy>=3.5.0",
    #     "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz",
    #     "pylean @ git+https://github.com/yeahrmek/pylean",
    #     "networkx",
    # ],
)
