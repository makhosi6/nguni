# Whisper Multilingual Fine-tuning Pipeline

Production-grade fine-tuning pipeline for OpenAI's Whisper-large-v3 model on multilingual South African speech data.

## Features

- **Resource-Aware Training**: Automatic hyperparameter selection based on language resource tier
- **Memory Optimization**: Gradient checkpointing, mixed precision (FP16), dynamic padding
- **Multiple Training Strategies**: Per-language, multilingual, and transfer learning
- **Comprehensive Evaluation**: WER and CER metrics with per-language tracking
- **Production Ready**: Deployment-optimized models with inference engine

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

### Per-Language Training

Train a model for a specific language:

```bash
python scripts/train_per_language.py \
    --language nr \
    --resource-tier high \
    --train-jsonl nr-ZA/transcriptions/train.jsonl \
    --val-jsonl nr-ZA/transcriptions/val.jsonl \
    --audio-base-path . \
    --checkpoint-dir ./checkpoints/nr
```

### Multilingual Training

Train a single model on multiple languages:

```bash
python scripts/train_multilingual.py \
    --languages nr tn ts xh zu af \
    --train-jsonl dataset.jsonl \
    --val-jsonl dataset.jsonl \
    --audio-base-path . \
    --checkpoint-dir ./checkpoints/multilingual
```

### Transfer Learning

Train low-resource languages from multilingual base:

```bash
python scripts/train_transfer.py \
    --language en \
    --base-model ./checkpoints/multilingual/best \
    --train-jsonl en-ZA/transcriptions/train.jsonl \
    --val-jsonl en-ZA/transcriptions/val.jsonl \
    --audio-base-path . \
    --checkpoint-dir ./checkpoints/transfer/en
```

### Full Pipeline

Run the complete training pipeline:

```bash
python scripts/train_orchestrator.py \
    --strategy full \
    --high-resource nr tn ts xh zu \
    --medium-resource af \
    --low-resource en ss \
    --checkpoint-dir ./checkpoints
```

## Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py \
    --model-path ./checkpoints/nr/best \
    --val-jsonl nr-ZA/transcriptions/val.jsonl \
    --audio-base-path . \
    --language nr
```

## Inference

Run batch inference on audio files:

```bash
python scripts/inference.py \
    --model-path ./checkpoints/nr/best \
    --audio-dir ./audio_files \
    --output transcriptions.json \
    --language nr
```

## Configuration

Configuration files are in `configs/`:
- `base_config.yaml`: Default hyperparameters
- `high_resource.yaml`, `medium_resource.yaml`, `low_resource.yaml`: Resource tier overrides
- `multilingual.yaml`: Multilingual training config
- `languages/*.yaml`: Language-specific configs

## Project Structure

```
alpha/
├── src/
│   ├── data/          # Data loading and preprocessing
│   ├── model/          # Model factory, checkpoints, evaluation
│   ├── training/      # Training strategies
│   ├── config/         # Configuration management
│   ├── tracking/       # Experiment tracking
│   └── inference/      # Inference engine
├── configs/            # YAML configuration files
├── scripts/            # Training and evaluation scripts
└── requirements.txt    # Python dependencies
```

## Resource Tiers

Based on dataset size:
- **HIGH_RESOURCE** (>30k): nr, tn, ts, xh, zu
- **MEDIUM_RESOURCE** (>2k): af
- **LOW_RESOURCE** (<1k): en, ss

## Performance Targets

- HIGH_RESOURCE languages: WER < 10%
- MEDIUM_RESOURCE languages: WER < 15%
- LOW_RESOURCE languages: WER < 25%
- Multilingual model: Average WER < 12%

## License

[Your License Here]

