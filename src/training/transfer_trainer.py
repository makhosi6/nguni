"""Transfer learning trainer for low-resource languages."""
import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional
import logging

from .base_trainer import BaseTrainer
from .per_language_trainer import PerLanguageTrainer
from ..model.evaluator import WhisperEvaluator
from ..model.factory import WhisperModelFactory
from ..config.models import ExperimentConfig


logger = logging.getLogger(__name__)


class TransferLearningTrainer(PerLanguageTrainer):
    """
    Trainer for transfer learning from multilingual base model.
    
    Initializes from multilingual checkpoint and uses higher learning rate
    for rapid adaptation to low-resource languages.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        language_code: str,
        base_model_path: str,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        checkpoint_dir: str = "./checkpoints"
    ):
        """
        Initialize transfer learning trainer.
        
        Args:
            config: Experiment configuration
            language_code: Language code for low-resource language
            base_model_path: Path to multilingual checkpoint
            train_dataloader: Training data loader (created if None)
            val_dataloader: Validation data loader (created if None)
            checkpoint_dir: Directory for saving checkpoints
        """
        self.base_model_path = base_model_path
        
        # Initialize parent without calling super().__init__ yet
        # We need to load the base model first
        self.language_code = language_code
        
        # Create dataloaders
        if train_dataloader is None:
            from ..data.dataset import WhisperDataset, collate_fn
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
            from ..data.dataset import WhisperDataset, collate_fn
            val_dataset = WhisperDataset(
                jsonl_path=config.data.val_jsonl,
                audio_base_path=config.data.audio_base_path,
                language_code=language_code,
                max_audio_length=config.data.max_audio_length,
                cache_dir=config.data.cache_dir,
                validate=False
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                pin_memory=config.data.pin_memory,
                collate_fn=collate_fn
            )
        
        # Initialize base trainer components
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Load model from base checkpoint
        logger.info(f"Loading base model from: {base_model_path}")
        self.model = WhisperModelFactory.create_model(
            model_name=config.model.name,
            gradient_checkpointing=config.model.gradient_checkpointing,
            device=str(self.device),
            from_checkpoint=base_model_path
        )
        
        # Initialize processor
        self.processor = WhisperModelFactory.create_processor(config.model.name)
        
        # Setup optimizer and scheduler (with higher LR for transfer learning)
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision training
        from torch.cuda.amp import GradScaler
        self.scaler = GradScaler() if config.optimization.mixed_precision else None
        
        # Checkpoint manager
        from ..model.checkpoint import CheckpointManager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            metric_name=config.evaluation.metric,
            mode="min"
        )
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.no_improvement_count = 0
        
        # Early stopping
        self.early_stopping_patience = config.evaluation.early_stopping_patience
        self.eval_steps = config.training.eval_steps
        
        # Set language-specific forced decoder IDs
        self.forced_decoder_ids = WhisperModelFactory.get_forced_decoder_ids(
            self.processor,
            language=language_code,
            task="transcribe"
        )
        
        logger.info(f"Initialized TransferLearningTrainer for language: {language_code}")
        logger.info(f"Base model: {base_model_path}")
        logger.info(f"Learning rate: {config.training.learning_rate} (higher for transfer learning)")
    
    def validation_step(self) -> Dict[str, float]:
        """Run validation and compare against base model if available."""
        evaluator = WhisperEvaluator(
            model=self.model,
            processor=self.processor,
            device=self.device
        )
        
        metrics = evaluator.evaluate(
            dataloader=self.val_dataloader,
            language=self.language_code
        )
        
        # TODO: Compare against base model performance if base model evaluator available
        # This would require loading the base model separately for comparison
        
        return metrics

