"""Per-language trainer for language-specific model training."""
import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional
import logging

from .base_trainer import BaseTrainer
from ..model.evaluator import WhisperEvaluator
from ..model.factory import WhisperModelFactory
from ..config.models import ExperimentConfig
from ..data.dataset import WhisperDataset, collate_fn


logger = logging.getLogger(__name__)


class PerLanguageTrainer(BaseTrainer):
    """
    Trainer for per-language model training.
    
    Automatically selects hyperparameters based on resource tier
    and sets language-specific forced decoder IDs.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        language_code: str,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        checkpoint_dir: str = "./checkpoints"
    ):
        """
        Initialize per-language trainer.
        
        Args:
            config: Experiment configuration
            language_code: Language code (e.g., 'af', 'nr')
            train_dataloader: Training data loader (created if None)
            val_dataloader: Validation data loader (created if None)
            checkpoint_dir: Directory for saving checkpoints
        """
        self.language_code = language_code
        
        # Create dataloaders if not provided
        if train_dataloader is None:
            train_dataset = WhisperDataset(
                jsonl_path=config.data.train_jsonl,
                audio_base_path=config.data.audio_base_path,
                language_code=language_code,
                max_audio_length=config.data.max_audio_length,
                cache_dir=config.data.cache_dir
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config.training.batch_size,
                shuffle=True,
                num_workers=config.data.num_workers,
                pin_memory=config.data.pin_memory,
                collate_fn=collate_fn
            )
        
        if val_dataloader is None and config.data.val_jsonl:
            val_dataset = WhisperDataset(
                jsonl_path=config.data.val_jsonl,
                audio_base_path=config.data.audio_base_path,
                language_code=language_code,
                max_audio_length=config.data.max_audio_length,
                cache_dir=config.data.cache_dir,
                validate=False  # Faster validation loading
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                pin_memory=config.data.pin_memory,
                collate_fn=collate_fn
            )
        
        super().__init__(
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            checkpoint_dir=checkpoint_dir
        )
        
        # Set language-specific forced decoder IDs
        self.forced_decoder_ids = WhisperModelFactory.get_forced_decoder_ids(
            self.processor,
            language=language_code,
            task="transcribe"
        )
        
        logger.info(f"Initialized PerLanguageTrainer for language: {language_code}")
    
    def validation_step(self) -> Dict[str, float]:
        """Run validation and return metrics."""
        evaluator = WhisperEvaluator(
            model=self.model,
            processor=self.processor,
            device=self.device
        )
        
        metrics = evaluator.evaluate(
            dataloader=self.val_dataloader,
            language=self.language_code
        )
        
        return metrics

