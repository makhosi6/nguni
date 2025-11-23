"""
Efficient JSONL + audio loading for Whisper fine-tuning.

Handles:
- Streaming JSONL parsing
- Audio file loading with caching
- Dynamic batching with padding
- Multi-process data loading
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import WhisperProcessor

logger = logging.getLogger(__name__)


class WhisperDataset:
    """Dataset class for Whisper fine-tuning with efficient audio loading."""

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        processor: WhisperProcessor,
        root_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        max_audio_length: float = 30.0,
        min_audio_length: float = 0.3,
    ):
        """
        Initialize Whisper dataset.

        Args:
            jsonl_path: Path to JSONL file with audio/text/language entries
            processor: WhisperProcessor for feature extraction
            root_dir: Root directory for resolving relative audio paths
            cache_dir: Directory for caching preprocessed features
            max_audio_length: Maximum audio duration in seconds
            min_audio_length: Minimum audio duration in seconds
        """
        self.jsonl_path = Path(jsonl_path)
        self.processor = processor
        self.root_dir = Path(root_dir) if root_dir else self.jsonl_path.parent.parent
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        logger.info(f"Loading dataset from {jsonl_path}")
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from JSONL file using HuggingFace datasets."""
        try:
            # Load JSONL as dataset
            self.dataset = load_dataset(
                "json",
                data_files=str(self.jsonl_path),
                split="train",
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
            logger.info(f"Loaded {len(self.dataset)} examples from {self.jsonl_path}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.

        Args:
            idx: Example index

        Returns:
            Dictionary with input_features, labels, and metadata
        """
        example = self.dataset[idx]

        # Resolve audio path
        audio_path = example["audio"]
        if isinstance(audio_path, dict):
            audio_path = audio_path.get("path", "")
        
        full_audio_path = self.root_dir / audio_path if not Path(audio_path).is_absolute() else Path(audio_path)

        if not full_audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {full_audio_path}")

        # Load audio
        try:
            audio = self._load_audio(full_audio_path)
        except Exception as e:
            logger.warning(f"Failed to load audio {full_audio_path}: {e}")
            # Return a dummy example that will be filtered
            return self._create_dummy_example()

        # Validate audio length
        duration = len(audio) / 16000.0  # Assuming 16kHz
        if duration < self.min_audio_length or duration > self.max_audio_length:
            logger.debug(f"Audio {full_audio_path} duration {duration:.2f}s out of range")
            return self._create_dummy_example()

        # Process with Whisper processor
        text = example.get("text", "")
        language = example.get("language", "")

        # Prepare forced decoder ids for language-specific decoding
        forced_decoder_ids = None
        if language:
            try:
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=language, task="transcribe"
                )
            except Exception as e:
                logger.warning(f"Failed to get decoder prompt ids for {language}: {e}")

        # Extract features
        inputs = self.processor(
            audio=audio,
            text=text,
            sampling_rate=16000,
            return_tensors="pt",
        )

        # Flatten tensors
        result = {
            "input_features": inputs["input_features"].squeeze(0),
            "labels": inputs["labels"].squeeze(0),
        }

        if forced_decoder_ids:
            result["forced_decoder_ids"] = torch.tensor(forced_decoder_ids, dtype=torch.long)

        return result

    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """Load audio file and return as numpy array."""
        import librosa

        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        return audio

    def _create_dummy_example(self) -> Dict[str, torch.Tensor]:
        """Create a dummy example for filtering."""
        return {
            "input_features": torch.zeros((80, 1)),
            "labels": torch.tensor([-100]),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for dynamic padding.

    Args:
        batch: List of examples

    Returns:
        Batched tensors with padding
    """
    # Filter out dummy examples
    batch = [ex for ex in batch if ex["labels"].numel() > 1]

    if len(batch) == 0:
        # Return minimal batch
        return {
            "input_features": torch.zeros((1, 80, 1)),
            "labels": torch.tensor([[-100]]),
        }

    # Get max lengths
    max_input_length = max(ex["input_features"].shape[1] for ex in batch)
    max_label_length = max(ex["labels"].shape[0] for ex in batch)

    # Pad sequences
    input_features = []
    labels = []
    attention_mask = []

    for ex in batch:
        input_feat = ex["input_features"]
        label = ex["labels"]

        # Pad input features
        pad_length = max_input_length - input_feat.shape[1]
        if pad_length > 0:
            input_feat = torch.nn.functional.pad(
                input_feat, (0, pad_length), mode="constant", value=0.0
            )
        input_features.append(input_feat)

        # Pad labels
        pad_length = max_label_length - label.shape[0]
        if pad_length > 0:
            label = torch.nn.functional.pad(
                label, (0, pad_length), mode="constant", value=-100
            )
        labels.append(label)

        # Create attention mask
        mask = torch.ones(input_feat.shape[1], dtype=torch.float32)
        attention_mask.append(mask)

    # Stack tensors
    input_features = torch.stack(input_features)
    labels = torch.stack(labels)
    attention_mask = torch.stack(attention_mask)

    result = {
        "input_features": input_features,
        "labels": labels,
        "attention_mask": attention_mask,
    }

    # Add forced_decoder_ids if present
    if "forced_decoder_ids" in batch[0]:
        forced_decoder_ids = [ex.get("forced_decoder_ids") for ex in batch]
        result["forced_decoder_ids"] = forced_decoder_ids

    return result


def create_dataloader(
    jsonl_path: Union[str, Path],
    processor: WhisperProcessor,
    root_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create a DataLoader for Whisper training.

    Args:
        jsonl_path: Path to JSONL file
        processor: WhisperProcessor instance
        root_dir: Root directory for audio paths
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle: Whether to shuffle data
        **dataset_kwargs: Additional arguments for WhisperDataset

    Returns:
        Configured DataLoader
    """
    dataset = WhisperDataset(jsonl_path, processor, root_dir=root_dir, **dataset_kwargs)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,  # Drop last incomplete batch
    )

    return dataloader

