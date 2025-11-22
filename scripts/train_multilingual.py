#!/usr/bin/env python3
"""CLI script for multilingual training."""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import ConfigLoader
from src.training.multilingual_trainer import MultilingualTrainer
from src.config.models import ExperimentConfig


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train multilingual Whisper model")
    parser.add_argument(
        "--languages",
        type=str,
        nargs='+',
        required=True,
        help="Language codes (e.g., 'af nr tn ts xh zu')"
    )
    parser.add_argument(
        "--train-jsonl",
        type=str,
        required=True,
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--val-jsonl",
        type=str,
        help="Path to validation JSONL file"
    )
    parser.add_argument(
        "--audio-base-path",
        type=str,
        required=True,
        help="Base directory for audio files"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/multilingual",
        help="Directory for saving checkpoints"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name"
    )
    parser.add_argument(
        "--max-examples-per-lang",
        type=int,
        default=5000,
        help="Maximum examples per language"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_multilingual_config()
    
    # Override data paths
    config.data.train_jsonl = args.train_jsonl
    config.data.val_jsonl = args.val_jsonl
    config.data.audio_base_path = args.audio_base_path
    config.data.language_codes = args.languages
    config.data.max_examples_per_lang = args.max_examples_per_lang
    
    # Set experiment name
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    else:
        config.experiment_name = f"multilingual_{'_'.join(args.languages)}"
    
    # Create trainer
    trainer = MultilingualTrainer(
        config=config,
        language_codes=args.languages,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    logger.info(f"Starting multilingual training for languages: {args.languages}")
    trainer.train()
    
    logger.info("Training completed")


if __name__ == "__main__":
    main()

