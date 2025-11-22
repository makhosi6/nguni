"""Whisper dataset for training."""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from torch.utils.data import Dataset
from transformers import WhisperProcessor
from .feature_extractor import WhisperFeatureExtractor
from .validator import DataValidator


class WhisperDataset(Dataset):
    """
    PyTorch Dataset for Whisper training.
    
    Loads JSONL files with audio paths and transcriptions, extracts features,
    and tokenizes text for training.
    """
    
    def __init__(
        self,
        jsonl_path: str,
        audio_base_path: str,
        language_code: Optional[str] = None,
        max_audio_length: float = 30.0,
        cache_dir: Optional[str] = None,
        processor: Optional[WhisperProcessor] = None,
        feature_extractor: Optional[WhisperFeatureExtractor] = None,
        validate: bool = True
    ):
        """
        Initialize Whisper dataset.
        
        Args:
            jsonl_path: Path to JSONL file with audio paths and transcriptions
            audio_base_path: Base directory for audio files
            language_code: Language code for forced decoder IDs (e.g., 'af', 'nr')
            max_audio_length: Maximum audio duration in seconds
            cache_dir: Optional directory for feature caching
            processor: WhisperProcessor instance (created if None)
            feature_extractor: WhisperFeatureExtractor instance (created if None)
            validate: Whether to validate data on initialization
        """
        self.jsonl_path = Path(jsonl_path)
        self.audio_base_path = Path(audio_base_path)
        self.language_code = language_code
        self.max_audio_length = max_audio_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize processor and feature extractor
        if processor is None:
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        else:
            self.processor = processor
        
        if feature_extractor is None:
            self.feature_extractor = WhisperFeatureExtractor()
        else:
            self.feature_extractor = feature_extractor
        
        # Load and validate examples
        if validate:
            validator = DataValidator(
                audio_base_path=str(self.audio_base_path),
                min_duration=0.5,
                max_duration=max_audio_length
            )
            self.examples, self.stats = validator.validate_jsonl(str(self.jsonl_path))
            print(f"Loaded {len(self.examples)} valid examples from {jsonl_path}")
            if self.stats['skipped_examples'] > 0:
                print(f"Warning: Skipped {self.stats['skipped_examples']} invalid examples")
        else:
            # Load without validation (faster)
            self.examples = []
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line.strip())
                    if 'audio' in example and 'text' in example:
                        self.examples.append(example)
            self.stats = {'valid_examples': len(self.examples)}
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            idx: Example index
        
        Returns:
            Dictionary with 'input_features', 'labels', and optionally 'language'
        """
        example = self.examples[idx]
        
        # Load audio and extract features
        audio_path = self.audio_base_path / example['audio']
        
        # Check cache
        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / f"{hash(str(audio_path))}.pt"
            if cache_path.exists():
                input_features = torch.load(cache_path)
            else:
                input_features = self.feature_extractor.extract_features(str(audio_path))
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(input_features, cache_path)
        else:
            input_features = self.feature_extractor.extract_features(str(audio_path))
        
        # Tokenize text
        text = example['text']
        labels = self.processor.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=448  # Whisper max length
        )["input_ids"].squeeze(0)
        
        result = {
            'input_features': input_features,
            'labels': labels
        }
        
        # Add language code if available
        if self.language_code:
            result['language'] = self.language_code
        
        return result


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for dynamic padding.
    
    Args:
        batch: List of examples from dataset
    
    Returns:
        Batched tensors with dynamic padding
    """
    input_features = [item['input_features'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Pad input features (spectrograms)
    max_feature_len = max(f.shape[0] for f in input_features)
    padded_features = []
    for f in input_features:
        pad_len = max_feature_len - f.shape[0]
        if pad_len > 0:
            f = torch.nn.functional.pad(f, (0, 0, 0, pad_len))
        padded_features.append(f)
    input_features = torch.stack(padded_features)
    
    # Pad labels (token sequences)
    max_label_len = max(l.shape[0] for l in labels)
    padded_labels = []
    attention_mask = []
    for l in labels:
        pad_len = max_label_len - l.shape[0]
        if pad_len > 0:
            l = torch.nn.functional.pad(l, (0, pad_len), value=-100)  # -100 is ignore index
            mask = torch.ones(len(labels[0]) + pad_len, dtype=torch.bool)
            mask[len(labels[0]):] = False
        else:
            mask = torch.ones(len(l), dtype=torch.bool)
        padded_labels.append(l)
        attention_mask.append(mask)
    
    labels = torch.stack(padded_labels)
    attention_mask = torch.stack(attention_mask)
    
    result = {
        'input_features': input_features,
        'labels': labels,
        'attention_mask': attention_mask
    }
    
    # Add language if present
    if 'language' in batch[0]:
        languages = [item['language'] for item in batch]
        result['languages'] = languages
    
    return result

