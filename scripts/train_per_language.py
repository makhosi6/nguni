#!/usr/bin/env python3
"""CLI script for per-language training."""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import ConfigLoader
from src.training.per_language_trainer import PerLanguageTrainer
from src.config.models import ExperimentConfig


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Whisper model for a specific language")
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Language code (e.g., 'af', 'nr', 'en')"
    )
    parser.add_argument(
        "--resource-tier",
        type=str,
        choices=['high', 'medium', 'low'],
        default='high',
        help="Resource tier for configuration"
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
        default="./checkpoints",
        help="Directory for saving checkpoints"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_language_config(
        language_code=args.language,
        resource_tier=args.resource_tier
    )
    
    # Override data paths
    config.data.train_jsonl = args.train_jsonl
    config.data.val_jsonl = args.val_jsonl
    config.data.audio_base_path = args.audio_base_path
    
    # Set experiment name
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    else:
        config.experiment_name = f"per_language_{args.language}_{args.resource_tier}"
    
    # Create trainer
    trainer = PerLanguageTrainer(
        config=config,
        language_code=args.language,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    logger.info(f"Starting training for language: {args.language}")
    trainer.train(resume_from_checkpoint=args.resume_from)
    
    logger.info("Training completed")


if __name__ == "__main__":
    main()

