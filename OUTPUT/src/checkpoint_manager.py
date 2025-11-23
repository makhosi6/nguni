"""
Training state persistence and checkpoint management.

Handles:
- Model checkpointing (best, last, intermediate)
- Optimizer state saving
- Training metadata
- Resume from checkpoint
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from transformers import TrainerState, TrainerControl

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage training checkpoints and state."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        save_strategy: str = "steps",
        save_steps: int = 500,
        save_total_limit: int = 3,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "wer",
        greater_is_better: bool = False,
    ):
        """
        Initialize checkpoint manager.

        Args:
            output_dir: Directory to save checkpoints
            save_strategy: When to save ('steps', 'epoch', 'no')
            save_steps: Save every N steps
            save_total_limit: Maximum number of checkpoints to keep
            load_best_model_at_end: Load best model after training
            metric_for_best_model: Metric to use for best model selection
            greater_is_better: Whether higher metric is better
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_strategy = save_strategy
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.load_best_model_at_end = load_best_model_at_end
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better

        self.best_metric = float("inf") if not greater_is_better else float("-inf")
        self.best_checkpoint = None
        self.checkpoint_history = []

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        step: int,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> Path:
        """
        Save training checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            step: Current training step
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model so far

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(checkpoint_dir)
        else:
            torch.save(model.state_dict(), checkpoint_dir / "pytorch_model.bin")

        # Save optimizer
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

        # Save scheduler
        if scheduler is not None:
            torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

        # Save training state
        training_state = {
            "step": step,
            "epoch": epoch,
            "best_metric": self.best_metric,
            "metrics": metrics or {},
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)

        # Update best checkpoint
        if is_best:
            if self.best_checkpoint:
                # Remove old best checkpoint
                if self.best_checkpoint.exists():
                    shutil.rmtree(self.best_checkpoint)
            self.best_checkpoint = checkpoint_dir
            best_dir = self.output_dir / "best_model"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(checkpoint_dir, best_dir)
            logger.info(f"Saved best model checkpoint at step {step}")

        # Track checkpoint history
        self.checkpoint_history.append((step, checkpoint_dir))

        # Clean up old checkpoints
        self._cleanup_checkpoints()

        logger.info(f"Saved checkpoint at step {step} to {checkpoint_dir}")
        return checkpoint_dir

    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        if len(self.checkpoint_history) <= self.save_total_limit:
            return

        # Sort by step
        self.checkpoint_history.sort(key=lambda x: x[0])

        # Remove oldest checkpoints (keep best and recent)
        num_to_remove = len(self.checkpoint_history) - self.save_total_limit
        for step, checkpoint_dir in self.checkpoint_history[:num_to_remove]:
            if checkpoint_dir != self.best_checkpoint and checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
                logger.debug(f"Removed old checkpoint: {checkpoint_dir}")

        # Update history
        self.checkpoint_history = self.checkpoint_history[num_to_remove:]

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into

        Returns:
            Dictionary with training state
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load model
        if hasattr(model, "from_pretrained"):
            model = model.from_pretrained(checkpoint_path)
        else:
            model.load_state_dict(torch.load(checkpoint_path / "pytorch_model.bin"))

        # Load optimizer
        if optimizer is not None and (checkpoint_path / "optimizer.pt").exists():
            optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt"))

        # Load scheduler
        if scheduler is not None and (checkpoint_path / "scheduler.pt").exists():
            scheduler.load_state_dict(torch.load(checkpoint_path / "scheduler.pt"))

        # Load training state
        training_state = {}
        if (checkpoint_path / "training_state.json").exists():
            with open(checkpoint_path / "training_state.json", "r") as f:
                training_state = json.load(f)

        logger.info(f"Loaded checkpoint from step {training_state.get('step', 'unknown')}")
        return training_state

    def update_best_metric(self, metric_value: float, step: int):
        """
        Update best metric value.

        Args:
            metric_value: Current metric value
            step: Current step
        """
        is_best = False
        if self.greater_is_better:
            if metric_value > self.best_metric:
                self.best_metric = metric_value
                is_best = True
        else:
            if metric_value < self.best_metric:
                self.best_metric = metric_value
                is_best = True

        if is_best:
            logger.info(
                f"New best {self.metric_for_best_model}: {metric_value:.4f} at step {step}"
            )

        return is_best

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        if self.best_checkpoint and self.best_checkpoint.exists():
            return self.best_checkpoint
        best_dir = self.output_dir / "best_model"
        if best_dir.exists():
            return best_dir
        return None

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        if not self.checkpoint_history:
            return None

        # Return checkpoint with highest step
        latest_step, latest_dir = max(self.checkpoint_history, key=lambda x: x[0])
        if latest_dir.exists():
            return latest_dir
        return None

