"""Training orchestrator for coordinating multi-stage training."""
import logging
from typing import List, Dict, Optional
from pathlib import Path

from .per_language_trainer import PerLanguageTrainer
from .multilingual_trainer import MultilingualTrainer
from .transfer_trainer import TransferLearningTrainer
from ..config.loader import ConfigLoader
from ..config.models import ExperimentConfig


logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """
    Orchestrates complete training pipeline.
    
    Supports:
    - Sequential training (multilingual → transfer learning)
    - Parallel per-language training
    - Configuration management
    """
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        """
        Initialize training orchestrator.
        
        Args:
            config_loader: ConfigLoader instance (created if None)
        """
        if config_loader is None:
            config_loader = ConfigLoader()
        self.config_loader = config_loader
    
    def run_multilingual_training(
        self,
        language_codes: List[str],
        config_overrides: Optional[Dict] = None,
        checkpoint_dir: str = "./checkpoints/multilingual"
    ):
        """
        Train single multilingual model.
        
        Args:
            language_codes: List of language codes to include
            config_overrides: Additional configuration overrides
            checkpoint_dir: Directory for saving checkpoints
        """
        logger.info(f"Starting multilingual training for languages: {language_codes}")
        
        # Load multilingual configuration
        config = self.config_loader.load_multilingual_config()
        
        # Apply overrides
        if config_overrides:
            config_dict = config.dict()
            config_dict.update(config_overrides)
            config = ExperimentConfig(**config_dict)
        
        # Set language codes
        config.data.language_codes = language_codes
        
        # Create trainer
        trainer = MultilingualTrainer(
            config=config,
            language_codes=language_codes,
            checkpoint_dir=checkpoint_dir
        )
        
        # Train
        trainer.train()
        
        logger.info("Multilingual training completed")
        return trainer.checkpoint_manager.get_best_checkpoint()
    
    def run_per_language_training(
        self,
        language_codes: List[str],
        resource_tiers: Optional[Dict[str, str]] = None,
        checkpoint_base_dir: str = "./checkpoints"
    ) -> Dict[str, str]:
        """
        Train separate model for each language.
        
        Args:
            language_codes: List of language codes to train
            resource_tiers: Dictionary mapping language codes to resource tiers
            checkpoint_base_dir: Base directory for checkpoints
        
        Returns:
            Dictionary mapping language codes to checkpoint paths
        """
        logger.info(f"Starting per-language training for languages: {language_codes}")
        
        if resource_tiers is None:
            # Default resource tiers based on dataset summary
            resource_tiers = {
                'nr': 'high', 'tn': 'high', 'ts': 'high', 'xh': 'high', 'zu': 'high',
                'af': 'medium',
                'en': 'low', 'ss': 'low'
            }
        
        checkpoint_paths = {}
        
        for lang_code in language_codes:
            resource_tier = resource_tiers.get(lang_code, 'high')
            logger.info(f"Training {lang_code} ({resource_tier} resource)")
            
            # Load language-specific configuration
            config = self.config_loader.load_language_config(
                language_code=lang_code,
                resource_tier=resource_tier
            )
            
            # Set data paths (assuming standard structure)
            # These should be set in config or passed as overrides
            checkpoint_dir = f"{checkpoint_base_dir}/{lang_code}"
            
            # Create trainer
            trainer = PerLanguageTrainer(
                config=config,
                language_code=lang_code,
                checkpoint_dir=checkpoint_dir
            )
            
            # Train
            trainer.train()
            
            # Get best checkpoint
            best_checkpoint = trainer.checkpoint_manager.get_best_checkpoint()
            checkpoint_paths[lang_code] = best_checkpoint
        
        logger.info("Per-language training completed")
        return checkpoint_paths
    
    def run_transfer_learning(
        self,
        language_codes: List[str],
        base_model_path: str,
        checkpoint_base_dir: str = "./checkpoints/transfer"
    ) -> Dict[str, str]:
        """
        Train low-resource languages via transfer learning.
        
        Args:
            language_codes: List of low-resource language codes
            base_model_path: Path to multilingual base model checkpoint
            checkpoint_base_dir: Base directory for checkpoints
        
        Returns:
            Dictionary mapping language codes to checkpoint paths
        """
        logger.info(f"Starting transfer learning for languages: {language_codes}")
        logger.info(f"Base model: {base_model_path}")
        
        checkpoint_paths = {}
        
        for lang_code in language_codes:
            logger.info(f"Training {lang_code} via transfer learning")
            
            # Load low-resource configuration
            config = self.config_loader.load_language_config(
                language_code=lang_code,
                resource_tier='low'
            )
            
            checkpoint_dir = f"{checkpoint_base_dir}/{lang_code}"
            
            # Create trainer
            trainer = TransferLearningTrainer(
                config=config,
                language_code=lang_code,
                base_model_path=base_model_path,
                checkpoint_dir=checkpoint_dir
            )
            
            # Train
            trainer.train()
            
            # Get best checkpoint
            best_checkpoint = trainer.checkpoint_manager.get_best_checkpoint()
            checkpoint_paths[lang_code] = best_checkpoint
        
        logger.info("Transfer learning completed")
        return checkpoint_paths
    
    def run_full_pipeline(
        self,
        high_resource_languages: List[str],
        medium_resource_languages: List[str],
        low_resource_languages: List[str],
        checkpoint_base_dir: str = "./checkpoints"
    ) -> Dict[str, any]:
        """
        Execute complete training pipeline:
        1. Train multilingual model on high/medium resource languages
        2. Train low-resource languages via transfer learning
        3. Train per-language models for comparison
        
        Args:
            high_resource_languages: List of high-resource language codes
            medium_resource_languages: List of medium-resource language codes
            low_resource_languages: List of low-resource language codes
            checkpoint_base_dir: Base directory for checkpoints
        
        Returns:
            Dictionary with checkpoint paths for all models
        """
        logger.info("Starting full training pipeline")
        
        results = {}
        
        # Step 1: Train multilingual model
        all_languages = high_resource_languages + medium_resource_languages
        if all_languages:
            logger.info("Step 1: Training multilingual model")
            multilingual_checkpoint = self.run_multilingual_training(
                language_codes=all_languages,
                checkpoint_dir=f"{checkpoint_base_dir}/multilingual"
            )
            results['multilingual'] = multilingual_checkpoint
        
        # Step 2: Transfer learning for low-resource languages
        if low_resource_languages and multilingual_checkpoint:
            logger.info("Step 2: Transfer learning for low-resource languages")
            transfer_checkpoints = self.run_transfer_learning(
                language_codes=low_resource_languages,
                base_model_path=multilingual_checkpoint,
                checkpoint_base_dir=f"{checkpoint_base_dir}/transfer"
            )
            results['transfer'] = transfer_checkpoints
        
        # Step 3: Per-language models (optional, for comparison)
        logger.info("Step 3: Training per-language models")
        all_languages_for_per_lang = (
            high_resource_languages +
            medium_resource_languages +
            low_resource_languages
        )
        per_lang_checkpoints = self.run_per_language_training(
            language_codes=all_languages_for_per_lang,
            checkpoint_base_dir=f"{checkpoint_base_dir}/per_language"
        )
        results['per_language'] = per_lang_checkpoints
        
        logger.info("Full training pipeline completed")
        return results

