"""Whisper feature extraction for audio preprocessing."""
import numpy as np
import torch
from transformers import WhisperFeatureExtractor as HFWhisperFeatureExtractor
from typing import Union
from .audio_preprocessor import load_audio, normalize_audio


class WhisperFeatureExtractor:
    """
    Wraps HuggingFace WhisperFeatureExtractor with additional preprocessing.
    
    Converts raw audio to Whisper-compatible 80-channel log-Mel spectrograms.
    """
    
    def __init__(self, model_name: str = "openai/whisper-large-v3", normalize: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            model_name: HuggingFace model identifier
            normalize: Whether to apply mean normalization to spectrograms
        """
        self.feature_extractor = HFWhisperFeatureExtractor.from_pretrained(model_name)
        self.sampling_rate = self.feature_extractor.sampling_rate  # 16000 Hz
        self.normalize = normalize
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Load, resample, and normalize audio.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Audio signal as numpy array (16kHz, mono, normalized)
        """
        # Load and resample to 16kHz
        audio = load_audio(audio_path, sr=self.sampling_rate, mono=True)
        
        # Normalize amplitude
        audio = normalize_audio(audio)
        
        return audio
    
    def extract_features(self, audio: Union[np.ndarray, str]) -> torch.Tensor:
        """
        Generate 80-bin log-Mel spectrogram from audio.
        
        Args:
            audio: Audio signal as numpy array or path to audio file
        
        Returns:
            Spectrogram tensor of shape (n_frames, 80)
        """
        # Load audio if path provided
        if isinstance(audio, str):
            audio = self.preprocess_audio(audio)
        
        # Extract features using HuggingFace extractor
        # This generates 80-bin log-Mel spectrograms with 25ms window, 10ms stride
        features = self.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        )["input_features"]
        
        # Remove batch dimension: (1, n_frames, 80) -> (n_frames, 80)
        features = features.squeeze(0)
        
        # Apply mean normalization if enabled
        if self.normalize:
            features = features - features.mean(dim=0, keepdim=True)
        
        return features

