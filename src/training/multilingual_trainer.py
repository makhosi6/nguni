"""Multilingual trainer for training single model on multiple languages."""
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import logging

from .base_trainer import BaseTrainer
from ..model.evaluator import WhisperEvaluator
from ..data.dataset import WhisperDataset, collate_fn
from ..data.sampler import BalancedMultilingualSampler
from ..config.models import ExperimentConfig


logger = logging.getLogger(__name__)


class MultilingualTrainer(BaseTrainer):
    """
    Trainer for multilingual model training.
    
    Uses balanced sampling to ensure fair representation across languages
    and evaluates on all languages.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        language_codes: List[str],
        train_dataloaders: Optional[Dict[str, DataLoader]] = None,
        val_dataloaders: Optional[Dict[str, DataLoader]] = None,
        checkpoint_dir: str = "./checkpoints"
    ):
        """
        Initialize multilingual trainer.
        
        Args:
            config: Experiment configuration
            language_codes: List of language codes to train on
            train_dataloaders: Dictionary of training dataloaders per language (created if None)
            val_dataloaders: Dictionary of validation dataloaders per language (created if None)
            checkpoint_dir: Directory for saving checkpoints
        """
        self.language_codes = language_codes
        
        # Create dataloaders if not provided
        if train_dataloaders is None:
            train_dataloaders = self._create_dataloaders(config, language_codes, is_train=True)
        
        if val_dataloaders is None:
            val_dataloaders = self._create_dataloaders(config, language_codes, is_train=False)
        
        # Create balanced multilingual dataloader
        train_dataloader = self._create_balanced_dataloader(
            config,
            train_dataloaders,
            language_codes
        )
        
        super().__init__(
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=None,  # We handle validation separately
            checkpoint_dir=checkpoint_dir
        )
        
        self.val_dataloaders = val_dataloaders
        
        logger.info(f"Initialized MultilingualTrainer for languages: {language_codes}")
    
    def _create_dataloaders(
        self,
        config: ExperimentConfig,
        language_codes: List[str],
        is_train: bool = True
    ) -> Dict[str, DataLoader]:
        """Create dataloaders for each language."""
        dataloaders = {}
        jsonl_path = config.data.train_jsonl if is_train else config.data.val_jsonl
        
        if not jsonl_path:
            return {}
        
        for lang_code in language_codes:
            dataset = WhisperDataset(
                jsonl_path=jsonl_path,
                audio_base_path=config.data.audio_base_path,
                language_code=lang_code,
                max_audio_length=config.data.max_audio_length,
                cache_dir=config.data.cache_dir,
                validate=is_train  # Only validate training data
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=config.training.batch_size,
                shuffle=is_train,
                num_workers=config.data.num_workers,
                pin_memory=config.data.pin_memory,
                collate_fn=collate_fn
            )
            
            dataloaders[lang_code] = dataloader
        
        return dataloaders
    
    def _create_balanced_dataloader(
        self,
        config: ExperimentConfig,
        dataloaders: Dict[str, DataLoader],
        language_codes: List[str]
    ) -> DataLoader:
        """Create balanced multilingual dataloader with single-language batches."""
        # Get datasets from dataloaders
        datasets = {lang: dataloader.dataset for lang, dataloader in dataloaders.items()}
        
        # Create balanced sampler
        sampler = BalancedMultilingualSampler(
            datasets=datasets,
            max_examples_per_lang=config.data.max_examples_per_lang or 5000,
            batch_size=config.training.batch_size,
            shuffle=True
        )
        
        # Create combined dataset (for indexing)
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset(list(datasets.values()))
        
        # Create dataloader with sampler
        dataloader = DataLoader(
            combined_dataset,
            batch_sampler=sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            collate_fn=collate_fn
        )
        
        return dataloader
    
    def validation_step(self) -> Dict[str, float]:
        """Run validation on all languages and return metrics."""
        evaluator = WhisperEvaluator(
            model=self.model,
            processor=self.processor,
            device=self.device
        )
        
        metrics = evaluator.evaluate_multilingual(
            dataloaders=self.val_dataloaders
        )
        
        # Use average WER as primary metric
        metrics[self.config.evaluation.metric] = metrics['average_wer']
        
        return metrics

