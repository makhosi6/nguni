from setuptools import setup, find_packages

setup(
    name="whisper-multilingual-finetuning",
    version="0.1.0",
    description="Production-grade fine-tuning pipeline for Whisper-large-v3 on multilingual South African speech data",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "jiwer>=3.0.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "wandb>=0.15.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
)

