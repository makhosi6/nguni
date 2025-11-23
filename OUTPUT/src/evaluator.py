"""
Comprehensive metric computation for ASR evaluation.

Handles:
- Word Error Rate (WER)
- Character Error Rate (CER)
- Per-language breakdowns
- Detailed error analysis
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from jiwer import cer, wer
from transformers import WhisperProcessor

logger = logging.getLogger(__name__)


class ASREvaluator:
    """Evaluator for ASR models with WER/CER metrics."""

    def __init__(self, processor: WhisperProcessor):
        """
        Initialize evaluator.

        Args:
            processor: WhisperProcessor for tokenization
        """
        self.processor = processor

    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
        languages: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute WER and CER metrics.

        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
            languages: Optional list of language codes for per-language metrics

        Returns:
            Dictionary with metrics
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions ({len(predictions)}) and references ({len(references)}) must have same length"
            )

        # Compute overall metrics
        overall_wer = wer(references, predictions)
        overall_cer = cer(references, predictions)

        metrics = {
            "wer": overall_wer,
            "cer": overall_cer,
        }

        # Per-language metrics
        if languages:
            lang_metrics = self._compute_per_language_metrics(
                predictions, references, languages
            )
            metrics.update(lang_metrics)

        return metrics

    def _compute_per_language_metrics(
        self,
        predictions: List[str],
        references: List[str],
        languages: List[str],
    ) -> Dict[str, float]:
        """
        Compute metrics per language.

        Args:
            predictions: List of predictions
            references: List of references
            languages: List of language codes

        Returns:
            Dictionary with per-language metrics
        """
        lang_groups = defaultdict(lambda: {"preds": [], "refs": []})

        for pred, ref, lang in zip(predictions, references, languages):
            lang_groups[lang]["preds"].append(pred)
            lang_groups[lang]["refs"].append(ref)

        lang_metrics = {}
        for lang, group in lang_groups.items():
            lang_wer = wer(group["refs"], group["preds"])
            lang_cer = cer(group["refs"], group["preds"])
            lang_metrics[f"wer_{lang}"] = lang_wer
            lang_metrics[f"cer_{lang}"] = lang_cer
            lang_metrics[f"count_{lang}"] = len(group["refs"])

        return lang_metrics

    def evaluate_batch(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        max_length: int = 448,
        language: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataloader.

        Args:
            model: Model to evaluate
            dataloader: DataLoader with evaluation data
            device: Device to run evaluation on
            max_length: Maximum generation length
            language: Optional language code for forced decoding

        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()

        all_predictions = []
        all_references = []
        all_languages = []

        with torch.no_grad():
            for batch in dataloader:
                input_features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)

                # Get forced decoder ids if language specified
                forced_decoder_ids = None
                if language:
                    try:
                        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                            language=language, task="transcribe"
                        )
                    except Exception:
                        pass

                # Generate predictions
                generated_ids = model.generate(
                    input_features,
                    max_length=max_length,
                    forced_decoder_ids=forced_decoder_ids,
                )

                # Decode predictions
                predictions = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                # Decode references
                labels[labels == -100] = self.processor.tokenizer.pad_token_id
                references = self.processor.batch_decode(labels, skip_special_tokens=True)

                all_predictions.extend(predictions)
                all_references.extend(references)

                # Get languages from batch if available
                if "language" in batch:
                    all_languages.extend(batch["language"])

        # Compute metrics
        metrics = self.compute_metrics(
            all_predictions,
            all_references,
            languages=all_languages if all_languages else None,
        )

        model.train()
        return metrics

    def evaluate_file(
        self,
        model: torch.nn.Module,
        jsonl_path: Union[str, Path],
        processor: WhisperProcessor,
        device: torch.device,
        root_dir: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a JSONL file.

        Args:
            model: Model to evaluate
            jsonl_path: Path to evaluation JSONL file
            processor: WhisperProcessor
            device: Device to run evaluation on
            root_dir: Root directory for audio paths
            max_samples: Maximum number of samples to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        import json

        jsonl_path = Path(jsonl_path)
        root_dir = Path(root_dir) if root_dir else jsonl_path.parent.parent

        predictions = []
        references = []
        languages = []

        model.eval()

        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break

                if not line.strip():
                    continue

                example = json.loads(line)
                audio_path = root_dir / example["audio"]
                text = example["text"]
                lang = example.get("language", "")

                try:
                    # Load and process audio
                    import librosa

                    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
                    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
                    input_features = inputs["input_features"].to(device)

                    # Generate prediction
                    with torch.no_grad():
                        generated_ids = model.generate(
                            input_features, max_length=448
                        )
                        prediction = processor.batch_decode(
                            generated_ids, skip_special_tokens=True
                        )[0]

                    predictions.append(prediction)
                    references.append(text)
                    languages.append(lang)

                except Exception as e:
                    logger.warning(f"Failed to evaluate {audio_path}: {e}")
                    continue

        # Compute metrics
        metrics = self.compute_metrics(predictions, references, languages)

        model.train()
        return metrics

