#!/usr/bin/env python3
"""
Automated training pipeline orchestrator.

Coordinates:
- Per-language training for high/medium resource languages
- Multilingual training
- Transfer learning for low-resource languages
- Model evaluation and comparison
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset_balancer import DatasetBalancer
from logging_config import setup_logging

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Orchestrate complete training pipeline."""

    def __init__(
        self,
        dataset_root: Path,
        dataset_summary_path: Path,
        output_base_dir: Path,
        config_dir: Path = None,
    ):
        """
        Initialize orchestrator.

        Args:
            dataset_root: Root directory of dataset
            dataset_summary_path: Path to dataset_summary.json
            output_base_dir: Base directory for all model outputs
            config_dir: Directory containing training configs
        """
        self.dataset_root = Path(dataset_root)
        self.dataset_summary_path = Path(dataset_summary_path)
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        if config_dir is None:
            config_dir = self.dataset_root / "training_configs"
        self.config_dir = Path(config_dir)

        # Load dataset summary
        with open(self.dataset_summary_path, "r") as f:
            self.summary = json.load(f)

        # Categorize languages
        self._categorize_languages()

    def _categorize_languages(self):
        """Categorize languages by resource level."""
        self.high_resource = []
        self.medium_resource = []
        self.low_resource = []
        self.excluded = []

        for lang_info in self.summary:
            lang_code = lang_info["language_code"]
            total = lang_info["total_examples"]

            if total >= 30000:
                self.high_resource.append(lang_code)
            elif total >= 2000:
                self.medium_resource.append(lang_code)
            elif total >= 10:
                self.low_resource.append(lang_code)
            else:
                self.excluded.append(lang_code)

        logger.info(f"High-resource languages: {self.high_resource}")
        logger.info(f"Medium-resource languages: {self.medium_resource}")
        logger.info(f"Low-resource languages: {self.low_resource}")
        logger.info(f"Excluded languages: {self.excluded}")

    def train_multilingual(self, use_wandb: bool = False) -> Path:
        """
        Train multilingual model.

        Args:
            use_wandb: Whether to use W&B tracking

        Returns:
            Path to trained multilingual model
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Training multilingual model")
        logger.info("=" * 80)

        output_dir = self.output_base_dir / "multilingual"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(self.dataset_root / "train_multilingual.py"),
            "--dataset_root",
            str(self.dataset_root),
            "--dataset_summary",
            str(self.dataset_summary_path),
            "--config",
            str(self.config_dir / "multilingual.yaml"),
            "--output_dir",
            str(output_dir),
            "--use_wandb" if use_wandb else "",
        ]
        cmd = [c for c in cmd if c]  # Remove empty strings

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(result.stdout)

        return output_dir / "best_model"

    def train_per_language(
        self, languages: List[str], use_wandb: bool = False
    ) -> Dict[str, Path]:
        """
        Train per-language models.

        Args:
            languages: List of language codes to train
            use_wandb: Whether to use W&B tracking

        Returns:
            Dictionary mapping language codes to model paths
        """
        logger.info("=" * 80)
        logger.info("STEP 2: Training per-language models")
        logger.info("=" * 80)

        model_paths = {}

        for lang_code in languages:
            logger.info(f"\nTraining model for language: {lang_code}")

            output_dir = self.output_base_dir / "per_language" / f"{lang_code}-ZA"
            output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(self.dataset_root / "train_per_language.py"),
                "--language",
                lang_code,
                "--dataset_root",
                str(self.dataset_root),
                "--output_dir",
                str(output_dir),
                "--use_wandb" if use_wandb else "",
            ]
            cmd = [c for c in cmd if c]

            try:
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(result.stdout)
                model_paths[lang_code] = output_dir / "best_model"
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to train {lang_code}: {e.stderr}")
                continue

        return model_paths

    def train_transfer_learning(
        self, base_model_path: Path, use_wandb: bool = False
    ) -> Dict[str, Path]:
        """
        Train low-resource languages with transfer learning.

        Args:
            base_model_path: Path to base multilingual model
            use_wandb: Whether to use W&B tracking

        Returns:
            Dictionary mapping language codes to model paths
        """
        logger.info("=" * 80)
        logger.info("STEP 3: Transfer learning for low-resource languages")
        logger.info("=" * 80)

        model_paths = {}

        for lang_code in self.low_resource:
            logger.info(f"\nTransfer learning for language: {lang_code}")

            output_dir = self.output_base_dir / "per_language" / f"{lang_code}-ZA"
            output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(self.dataset_root / "train_per_language.py"),
                "--language",
                lang_code,
                "--dataset_root",
                str(self.dataset_root),
                "--output_dir",
                str(output_dir),
                "--use_wandb" if use_wandb else "",
            ]
            cmd = [c for c in cmd if c]

            try:
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(result.stdout)
                model_paths[lang_code] = output_dir / "best_model"
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed transfer learning for {lang_code}: {e.stderr}")
                continue

        return model_paths

    def run_full_pipeline(self, use_wandb: bool = False):
        """
        Run complete training pipeline.

        Args:
            use_wandb: Whether to use W&B tracking
        """
        logger.info("=" * 80)
        logger.info("Starting complete training pipeline")
        logger.info("=" * 80)

        # Step 1: Train multilingual model
        multilingual_model = self.train_multilingual(use_wandb=use_wandb)

        # Step 2: Train high and medium resource languages
        high_medium_langs = self.high_resource + self.medium_resource
        per_lang_models = self.train_per_language(high_medium_langs, use_wandb=use_wandb)

        # Step 3: Transfer learning for low-resource languages
        transfer_models = self.train_transfer_learning(
            multilingual_model, use_wandb=use_wandb
        )

        # Combine all models
        all_models = {**per_lang_models, **transfer_models}

        # Save summary
        summary = {
            "multilingual_model": str(multilingual_model),
            "per_language_models": {
                lang: str(path) for lang, path in all_models.items()
            },
            "high_resource": self.high_resource,
            "medium_resource": self.medium_resource,
            "low_resource": self.low_resource,
            "excluded": self.excluded,
        }

        summary_path = self.output_base_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training pipeline completed. Summary saved to {summary_path}")
        logger.info(f"All models saved to {self.output_base_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate complete Whisper fine-tuning pipeline"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=".",
        help="Root directory of dataset",
    )
    parser.add_argument(
        "--dataset_summary",
        type=str,
        default="dataset_summary.json",
        help="Path to dataset_summary.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base output directory for all models",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="training_configs",
        help="Directory containing training configs",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for tracking",
    )
    parser.add_argument(
        "--multilingual_only",
        action="store_true",
        help="Only train multilingual model",
    )
    parser.add_argument(
        "--per_language_only",
        action="store_true",
        help="Only train per-language models (requires existing multilingual model)",
    )

    args = parser.parse_args()

    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=output_dir, log_level="INFO")

    # Initialize orchestrator
    dataset_root = Path(args.dataset_root)
    dataset_summary_path = dataset_root / args.dataset_summary

    orchestrator = TrainingOrchestrator(
        dataset_root=dataset_root,
        dataset_summary_path=dataset_summary_path,
        output_base_dir=output_dir,
        config_dir=args.config_dir,
    )

    # Run pipeline
    if args.multilingual_only:
        orchestrator.train_multilingual(use_wandb=args.use_wandb)
    elif args.per_language_only:
        high_medium_langs = orchestrator.high_resource + orchestrator.medium_resource
        orchestrator.train_per_language(high_medium_langs, use_wandb=args.use_wandb)
    else:
        orchestrator.run_full_pipeline(use_wandb=args.use_wandb)


if __name__ == "__main__":
    main()

