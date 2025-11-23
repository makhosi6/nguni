"""
Whisper-compatible feature extraction pipeline.

Handles:
- 80-bin Mel spectrogram extraction
- Mean normalization
- SpecAugment augmentation
- Feature caching
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

logger = logging.getLogger(__name__)


class WhisperFeatureExtractorWrapper:
    """Wrapper for Whisper feature extraction with augmentation support."""

    def __init__(
        self,
        feature_extractor: Optional[WhisperFeatureExtractor] = None,
        apply_specaugment: bool = False,
        time_mask_param: int = 10,
        freq_mask_param: int = 10,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
    ):
        """
        Initialize feature extractor.

        Args:
            feature_extractor: WhisperFeatureExtractor instance (creates new if None)
            apply_specaugment: Whether to apply SpecAugment during training
            time_mask_param: Maximum time mask width
            freq_mask_param: Maximum frequency mask width
            num_time_masks: Number of time masks to apply
            num_freq_masks: Number of frequency masks to apply
        """
        if feature_extractor is None:
            from transformers import WhisperProcessor

            processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            self.feature_extractor = processor.feature_extractor
        else:
            self.feature_extractor = feature_extractor

        self.apply_specaugment = apply_specaugment
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

    def extract_features(
        self, audio: np.ndarray, sampling_rate: int = 16000, training: bool = False
    ) -> np.ndarray:
        """
        Extract Mel spectrogram features.

        Args:
            audio: Audio array (1D)
            sampling_rate: Audio sample rate
            training: Whether in training mode (for augmentation)

        Returns:
            Feature array of shape (n_mels, n_frames)
        """
        # Extract features using Whisper feature extractor
        features = self.feature_extractor(
            audio, sampling_rate=sampling_rate, return_tensors="np"
        )

        # Get the feature array
        if isinstance(features, dict):
            feature_array = features["input_features"][0]  # Remove batch dimension
        else:
            feature_array = features

        # Apply SpecAugment if training
        if training and self.apply_specaugment:
            feature_array = self._apply_specaugment(feature_array)

        return feature_array

    def _apply_specaugment(self, features: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment to features.

        Args:
            features: Feature array of shape (n_mels, n_frames)

        Returns:
            Augmented feature array
        """
        features = features.copy()
        n_mels, n_frames = features.shape

        # Time masking
        for _ in range(self.num_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, max(1, n_frames - t))
            features[:, t0 : t0 + t] = 0.0

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, max(1, n_mels - f))
            features[f0 : f0 + f, :] = 0.0

        return features

    def extract_batch(
        self,
        audio_batch: list,
        sampling_rate: int = 16000,
        training: bool = False,
    ) -> np.ndarray:
        """
        Extract features for a batch of audio.

        Args:
            audio_batch: List of audio arrays
            sampling_rate: Audio sample rate
            training: Whether in training mode

        Returns:
            Batch of feature arrays
        """
        features_batch = []
        for audio in audio_batch:
            features = self.extract_features(audio, sampling_rate, training)
            features_batch.append(features)

        return np.array(features_batch)


class FeatureCache:
    """Cache for precomputed features."""

    def __init__(self, cache_dir: Union[str, Path], max_size: int = 10000):
        """
        Initialize feature cache.

        Args:
            cache_dir: Directory to store cached features
            max_size: Maximum number of cached features
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.cache = {}

    def get_cache_path(self, audio_path: Union[str, Path]) -> Path:
        """
        Get cache path for an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Path to cached feature file
        """
        audio_path = Path(audio_path)
        # Create hash-based cache key
        import hashlib

        cache_key = hashlib.md5(str(audio_path).encode()).hexdigest()
        return self.cache_dir / f"{cache_key}.npy"

    def get(self, audio_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Get cached features.

        Args:
            audio_path: Path to audio file

        Returns:
            Cached features or None if not found
        """
        cache_path = self.get_cache_path(audio_path)
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache for {audio_path}: {e}")
        return None

    def set(self, audio_path: Union[str, Path], features: np.ndarray):
        """
        Cache features.

        Args:
            audio_path: Path to audio file
            features: Feature array to cache
        """
        cache_path = self.get_cache_path(audio_path)
        try:
            np.save(cache_path, features)
        except Exception as e:
            logger.warning(f"Failed to save cache for {audio_path}: {e}")

    def clear(self):
        """Clear all cached features."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

