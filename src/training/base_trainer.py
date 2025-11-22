"""Base trainer with common training loop logic."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod
import logging

from ..model.factory import WhisperModelFactory
from ..model.checkpoint import CheckpointManager
from ..model.evaluator import WhisperEvaluator
from ..config.models import ExperimentConfig
from ..utils.reproducibility import set_seed


logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Base trainer with common training loop logic.
    
    Implements mixed precision training, gradient accumulation,
    gradient clipping, early stopping, and checkpointing.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        checkpoint_dir: str = "./checkpoints"
    ):
        """
        Initialize base trainer.
        
        Args:
            config: Experiment configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            checkpoint_dir: Directory for saving checkpoints
        """
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Set random seed for reproducibility
        set_seed(
            seed=config.reproducibility.seed,
            deterministic=config.reproducibility.deterministic
        )
        
        # Initialize model
        self.model = WhisperModelFactory.create_model(
            model_name=config.model.name,
            gradient_checkpointing=config.model.gradient_checkpointing,
            device=str(self.device)
        )
        
        # Initialize processor
        self.processor = WhisperModelFactory.create_processor(config.model.name)
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.optimization.mixed_precision else None
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            metric_name=config.evaluation.metric,
            mode="min"  # Lower WER is better
        )
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.no_improvement_count = 0
        
        # Early stopping
        self.early_stopping_patience = config.evaluation.early_stopping_patience
        
        # Evaluation frequency
        self.eval_steps = config.training.eval_steps
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup AdamW optimizer with weight decay."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
    
    def _setup_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Setup linear warmup + linear decay scheduler."""
        warmup_steps = self.config.training.warmup_steps
        max_steps = self.config.training.max_steps
        
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Linear decay to zero
                return max(
                    0.0,
                    float(max_steps - current_step) / float(max(1, max_steps - warmup_steps))
                )
        
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda
        )
    
    def should_evaluate(self, step: int) -> bool:
        """
        Determine if evaluation should run at this step.
        
        Args:
            step: Current training step
        
        Returns:
            True if evaluation should run
        """
        return step % self.eval_steps == 0
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step.
        
        Args:
            batch: Training batch
        
        Returns:
            Loss value
        """
        self.model.train()
        
        input_features = batch['input_features'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass with mixed precision
        if self.scaler is not None:
            with autocast():
                outputs = self.model(
                    input_features=input_features,
                    labels=labels,
                    attention_mask=attention_mask
                )
                loss = outputs.loss / self.config.training.gradient_accumulation_steps
        else:
            outputs = self.model(
                input_features=input_features,
                labels=labels,
                attention_mask=attention_mask
            )
            loss = outputs.loss / self.config.training.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config.training.gradient_accumulation_steps
    
    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.scaler is not None:
            # Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
    
    @abstractmethod
    def validation_step(self) -> Dict[str, float]:
        """
        Run validation and return metrics.
        
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Main training loop.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from (optional)
        """
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
            state = self.checkpoint_manager.load_checkpoint(
                resume_from_checkpoint,
                self.model,
                self.optimizer,
                self.scheduler
            )
            self.global_step = state['step']
            self.current_epoch = state.get('epoch', 0)
            self.best_metric = state['metrics'].get(self.config.evaluation.metric, float('inf'))
            logger.info(f"Resumed at step {self.global_step}, epoch {self.current_epoch}")
        
        # Log initial configuration
        logger.info(f"Starting training with config: {self.config.experiment_name}")
        logger.info(f"Max steps: {self.config.training.max_steps}")
        logger.info(f"Batch size: {self.config.training.batch_size}")
        logger.info(f"Gradient accumulation: {self.config.training.gradient_accumulation_steps}")
        logger.info(f"Learning rate: {self.config.training.learning_rate}")
        
        # Training loop
        self.model.train()
        accumulated_loss = 0.0
        accumulation_steps = 0
        
        while self.global_step < self.config.training.max_steps:
            for batch in self.train_dataloader:
                if self.global_step >= self.config.training.max_steps:
                    break
                
                # Training step
                loss = self.training_step(batch)
                accumulated_loss += loss
                accumulation_steps += 1
                
                # Optimizer step after accumulation
                if accumulation_steps >= self.config.training.gradient_accumulation_steps:
                    self.optimizer_step()
                    accumulation_steps = 0
                    
                    # Logging
                    avg_loss = accumulated_loss / self.config.training.gradient_accumulation_steps
                    current_lr = self.scheduler.get_last_lr()[0]
                    
                    logger.info(
                        f"Step {self.global_step}: loss={avg_loss:.4f}, "
                        f"lr={current_lr:.2e}"
                    )
                    
                    accumulated_loss = 0.0
                
                # Evaluation
                if self.should_evaluate(self.global_step) and self.val_dataloader is not None:
                    metrics = self.validation_step()
                    metric_value = metrics.get(self.config.evaluation.metric, float('inf'))
                    
                    logger.info(
                        f"Evaluation at step {self.global_step}: "
                        f"{self.config.evaluation.metric}={metric_value:.4f}"
                    )
                    
                    # Save checkpoint if best
                    if metric_value < self.best_metric:
                        self.best_metric = metric_value
                        self.no_improvement_count = 0
                        
                        checkpoint_path = self.checkpoint_manager.save_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            step=self.global_step,
                            metrics=metrics,
                            config=self.config.dict(),
                            epoch=self.current_epoch
                        )
                        logger.info(f"Saved best checkpoint: {checkpoint_path}")
                    else:
                        self.no_improvement_count += 1
                    
                    # Early stopping
                    if (self.early_stopping_patience and
                        self.no_improvement_count >= self.early_stopping_patience):
                        logger.info(
                            f"Early stopping triggered after {self.no_improvement_count} "
                            f"evaluations without improvement"
                        )
                        return
                    
                    self.model.train()
                
                self.global_step += 1
            
            self.current_epoch += 1
        
        # Save final checkpoint
        if self.val_dataloader is not None:
            metrics = self.validation_step()
        else:
            metrics = {}
        
        final_checkpoint = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            metrics=metrics,
            config=self.config.dict(),
            epoch=self.current_epoch
        )
        logger.info(f"Training completed. Final checkpoint: {final_checkpoint}")

