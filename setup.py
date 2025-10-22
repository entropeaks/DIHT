from setuptools import setup, find_packages

setup(
    name="diht",
    version="0.1.0",
    author="Timothée Coste",
    description="Deep Image Hashing Transformer (DIHT)",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "tqdm",
        "numpy",
        "pandas",
        "scikit-learn",
        "pyyaml",
        "matplotlib",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "diht-train = scripts.train_and_eval:main",
        ],
    },
)