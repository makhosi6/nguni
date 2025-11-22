#!/usr/bin/env python3
"""CLI script for full training pipeline orchestration."""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import ConfigLoader
from src.training.orchestrator import TrainingOrchestrator


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run full training pipeline")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=['full', 'multilingual', 'per_language', 'transfer'],
        default='full',
        help="Training strategy"
    )
    parser.add_argument(
        "--high-resource",
        type=str,
        nargs='+',
        default=['nr', 'tn', 'ts', 'xh', 'zu'],
        help="High-resource language codes"
    )
    parser.add_argument(
        "--medium-resource",
        type=str,
        nargs='+',
        default=['af'],
        help="Medium-resource language codes"
    )
    parser.add_argument(
        "--low-resource",
        type=str,
        nargs='+',
        default=['en', 'ss'],
        help="Low-resource language codes"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Base directory for checkpoints"
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    config_loader = ConfigLoader()
    orchestrator = TrainingOrchestrator(config_loader=config_loader)
    
    if args.strategy == 'full':
        # Run full pipeline
        results = orchestrator.run_full_pipeline(
            high_resource_languages=args.high_resource,
            medium_resource_languages=args.medium_resource,
            low_resource_languages=args.low_resource,
            checkpoint_base_dir=args.checkpoint_dir
        )
        logger.info(f"Full pipeline completed. Results: {results}")
    
    elif args.strategy == 'multilingual':
        all_languages = args.high_resource + args.medium_resource
        checkpoint = orchestrator.run_multilingual_training(
            language_codes=all_languages,
            checkpoint_dir=f"{args.checkpoint_dir}/multilingual"
        )
        logger.info(f"Multilingual training completed. Checkpoint: {checkpoint}")
    
    elif args.strategy == 'per_language':
        all_languages = args.high_resource + args.medium_resource + args.low_resource
        checkpoints = orchestrator.run_per_language_training(
            language_codes=all_languages,
            checkpoint_base_dir=f"{args.checkpoint_dir}/per_language"
        )
        logger.info(f"Per-language training completed. Checkpoints: {checkpoints}")
    
    elif args.strategy == 'transfer':
        # Need base model for transfer learning
        parser.error("Transfer learning requires --base-model. Use train_transfer.py instead.")
    
    logger.info("Pipeline completed")


if __name__ == "__main__":
    main()

