#!/usr/bin/env python3
"""CLI script for transfer learning."""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import ConfigLoader
from src.training.transfer_trainer import TransferLearningTrainer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train low-resource language via transfer learning")
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Language code (e.g., 'en', 'ss')"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to multilingual base model checkpoint"
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
        default="./checkpoints/transfer",
        help="Directory for saving checkpoints"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_language_config(
        language_code=args.language,
        resource_tier='low'
    )
    
    # Override data paths
    config.data.train_jsonl = args.train_jsonl
    config.data.val_jsonl = args.val_jsonl
    config.data.audio_base_path = args.audio_base_path
    
    # Set experiment name
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    else:
        config.experiment_name = f"transfer_{args.language}"
    
    # Create trainer
    trainer = TransferLearningTrainer(
        config=config,
        language_code=args.language,
        base_model_path=args.base_model,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    logger.info(f"Starting transfer learning for language: {args.language}")
    logger.info(f"Base model: {args.base_model}")
    trainer.train()
    
    logger.info("Training completed")


if __name__ == "__main__":
    main()

