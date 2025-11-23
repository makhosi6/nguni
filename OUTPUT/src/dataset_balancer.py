"""
Multilingual dataset balancing and sampling strategies.

Handles:
- Balanced sampling across languages
- Resource-aware sampling
- Stratified sampling
- Data augmentation for low-resource languages
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class DatasetBalancer:
    """Balance multilingual datasets for training."""

    def __init__(
        self,
        dataset_summary_path: Union[str, Path],
        root_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize dataset balancer.

        Args:
            dataset_summary_path: Path to dataset_summary.json
            root_dir: Root directory for dataset
        """
        self.dataset_summary_path = Path(dataset_summary_path)
        self.root_dir = Path(root_dir) if root_dir else self.dataset_summary_path.parent

        with open(self.dataset_summary_path, "r") as f:
            self.summary = json.load(f)

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

    def create_balanced_dataset(
        self,
        max_examples_per_lang: int = 5000,
        output_path: Optional[Union[str, Path]] = None,
        split: str = "train",
    ) -> List[Dict]:
        """
        Create a balanced multilingual dataset.

        Args:
            max_examples_per_lang: Maximum examples per language
            output_path: Path to save balanced dataset JSONL
            split: Dataset split ('train' or 'val')

        Returns:
            List of balanced examples
        """
        balanced_examples = []
        lang_counts = defaultdict(int)

        # Process each language
        for lang_info in self.summary:
            lang_code = lang_info["language_code"]
            lang_dir = lang_info["language_dir"]

            if lang_code in self.excluded:
                continue

            # Determine how many examples to sample
            if split == "train":
                total_examples = lang_info["train_examples"]
            else:
                total_examples = lang_info["val_examples"]

            if total_examples == 0:
                continue

            # Sample examples
            num_samples = min(total_examples, max_examples_per_lang)
            examples = self._sample_examples(lang_dir, split, num_samples)

            balanced_examples.extend(examples)
            lang_counts[lang_code] = len(examples)

            logger.info(
                f"Sampled {len(examples)} examples for {lang_code} ({split})"
            )

        # Log statistics
        total = sum(lang_counts.values())
        logger.info(f"Total balanced examples: {total}")
        for lang, count in lang_counts.items():
            logger.info(f"  {lang}: {count} ({count/total*100:.1f}%)")

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for ex in balanced_examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            logger.info(f"Saved balanced dataset to {output_path}")

        return balanced_examples

    def _sample_examples(
        self, lang_dir: str, split: str, num_samples: int
    ) -> List[Dict]:
        """
        Sample examples from a language dataset.

        Args:
            lang_dir: Language directory name
            split: Dataset split
            num_samples: Number of samples to take

        Returns:
            List of sampled examples
        """
        jsonl_path = self.root_dir / lang_dir / "transcriptions" / f"{split}.jsonl"

        if not jsonl_path.exists():
            logger.warning(f"JSONL file not found: {jsonl_path}")
            return []

        # Load all examples
        examples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        # Sample
        if len(examples) <= num_samples:
            return examples

        # Random sampling
        indices = np.random.choice(len(examples), size=num_samples, replace=False)
        return [examples[i] for i in indices]

    def get_language_weights(self, strategy: str = "inverse_frequency") -> Dict[str, float]:
        """
        Get sampling weights for each language.

        Args:
            strategy: Weighting strategy ('uniform', 'inverse_frequency', 'balanced')

        Returns:
            Dictionary mapping language codes to weights
        """
        weights = {}

        if strategy == "uniform":
            # Equal weights for all languages
            valid_langs = [
                lang["language_code"]
                for lang in self.summary
                if lang["language_code"] not in self.excluded
                and lang["total_examples"] > 0
            ]
            weight = 1.0 / len(valid_langs) if valid_langs else 0.0
            weights = {lang: weight for lang in valid_langs}

        elif strategy == "inverse_frequency":
            # Weight inversely proportional to frequency
            total_examples = sum(
                lang["total_examples"]
                for lang in self.summary
                if lang["language_code"] not in self.excluded
            )

            for lang_info in self.summary:
                lang_code = lang_info["language_code"]
                if lang_code in self.excluded or lang_info["total_examples"] == 0:
                    continue

                # Inverse frequency weighting
                freq = lang_info["total_examples"] / total_examples
                weights[lang_code] = 1.0 / freq if freq > 0 else 0.0

            # Normalize
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}

        elif strategy == "balanced":
            # Balanced weights (equal representation)
            valid_langs = [
                lang["language_code"]
                for lang in self.summary
                if lang["language_code"] not in self.excluded
                and lang["total_examples"] > 0
            ]
            weight = 1.0 / len(valid_langs) if valid_langs else 0.0
            weights = {lang: weight for lang in valid_langs}

        return weights

    def create_stratified_split(
        self,
        train_ratio: float = 0.9,
        seed: int = 42,
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Create stratified train/val split per language.

        Args:
            train_ratio: Ratio of training examples
            seed: Random seed

        Returns:
            Dictionary mapping language codes to train/val splits
        """
        np.random.seed(seed)
        splits = {}

        for lang_info in self.summary:
            lang_code = lang_info["language_code"]
            lang_dir = lang_info["language_dir"]

            if lang_code in self.excluded:
                continue

            # Load all examples
            combined_path = self.root_dir / lang_dir / "transcriptions"
            train_path = combined_path / "train.jsonl"
            val_path = combined_path / "val.jsonl"

            # If splits already exist, use them
            if train_path.exists() and val_path.exists():
                train_examples = []
                val_examples = []

                with open(train_path, "r") as f:
                    for line in f:
                        if line.strip():
                            train_examples.append(json.loads(line))

                with open(val_path, "r") as f:
                    for line in f:
                        if line.strip():
                            val_examples.append(json.loads(line))

                splits[lang_code] = {
                    "train": train_examples,
                    "val": val_examples,
                }
            else:
                logger.warning(
                    f"No existing splits found for {lang_code}, skipping"
                )

        return splits

