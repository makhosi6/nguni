"""Evaluation and metrics computation for Whisper models."""
import torch
import numpy as np
from typing import List, Dict, Optional
from torch.utils.data import DataLoader
from jiwer import wer, cer
import re


class WhisperEvaluator:
    """
    Evaluates Whisper models on validation data.
    
    Computes WER (Word Error Rate) and CER (Character Error Rate)
    with proper text normalization.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        processor,
        device: str = "cuda"
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Whisper model
            processor: WhisperProcessor instance
            device: Evaluation device
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.model.eval()
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for metric computation.
        
        Args:
            text: Input text
        
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation (optional, can be configured)
        # text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    def compute_wer(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Compute Word Error Rate.
        
        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
        
        Returns:
            WER as float (0.0 to 1.0+)
        """
        # Normalize texts
        preds_norm = [self.normalize_text(p) for p in predictions]
        refs_norm = [self.normalize_text(r) for r in references]
        
        # Compute WER using jiwer
        error = wer(refs_norm, preds_norm)
        return error
    
    def compute_cer(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Compute Character Error Rate.
        
        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
        
        Returns:
            CER as float (0.0 to 1.0+)
        """
        # Normalize texts
        preds_norm = [self.normalize_text(p) for p in predictions]
        refs_norm = [self.normalize_text(r) for r in references]
        
        # Compute CER using jiwer
        error = cer(refs_norm, preds_norm)
        return error
    
    def evaluate(
        self,
        dataloader: DataLoader,
        language: Optional[str] = None,
        max_eval_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute WER and CER on validation set.
        
        Args:
            dataloader: DataLoader with validation data
            language: Language code for forced decoder IDs (optional)
            max_eval_samples: Maximum number of samples to evaluate (optional)
        
        Returns:
            Dictionary with "wer", "cer", "num_examples"
        """
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_eval_samples and batch_idx * dataloader.batch_size >= max_eval_samples:
                    break
                
                input_features = batch['input_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Generate predictions
                if language:
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=language,
                        task="transcribe"
                    )
                else:
                    forced_decoder_ids = None
                
                generated_ids = self.model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_length=448
                )
                
                # Decode predictions
                pred_texts = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                
                # Decode references
                ref_texts = self.processor.batch_decode(
                    labels,
                    skip_special_tokens=True
                )
                
                predictions.extend(pred_texts)
                references.extend(ref_texts)
        
        # Compute metrics
        wer_score = self.compute_wer(predictions, references)
        cer_score = self.compute_cer(predictions, references)
        
        return {
            'wer': wer_score,
            'cer': cer_score,
            'num_examples': len(predictions)
        }
    
    def evaluate_multilingual(
        self,
        dataloaders: Dict[str, DataLoader],
        max_eval_samples: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Evaluate multilingual model on multiple languages.
        
        Args:
            dataloaders: Dictionary mapping language codes to DataLoaders
            max_eval_samples: Maximum samples per language (optional)
        
        Returns:
            Dictionary with per-language and average metrics
        """
        per_language_metrics = {}
        
        for lang_code, dataloader in dataloaders.items():
            metrics = self.evaluate(dataloader, language=lang_code, max_eval_samples=max_eval_samples)
            per_language_metrics[lang_code] = metrics
        
        # Compute average metrics
        avg_wer = np.mean([m['wer'] for m in per_language_metrics.values()])
        avg_cer = np.mean([m['cer'] for m in per_language_metrics.values()])
        total_examples = sum([m['num_examples'] for m in per_language_metrics.values()])
        
        return {
            'average_wer': avg_wer,
            'average_cer': avg_cer,
            'total_examples': total_examples,
            'per_language': per_language_metrics
        }

