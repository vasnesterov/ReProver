from setuptools import find_packages, setup

setup(
    name="ReProver",
    version="0.1",
    author="",
    description="Automated Theorem Prover for lean based on ReProver",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # "colbert-ai @ git+https://github.com/yeahrmek/ColBERT",
        "jsonargparse>=4.27.0",
    ],
)
