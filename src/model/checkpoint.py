"""Checkpoint management for saving and loading training state."""
import json
import shutil
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import random
import numpy as np


class CheckpointManager:
    """
    Manages model checkpoints with best model tracking.
    
    Saves complete training state including model, optimizer, scheduler,
    and random number generator states for reproducibility.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        keep_best_n: int = 3,
        metric_name: str = "wer",
        mode: str = "min"
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for saving checkpoints
            keep_best_n: Number of best checkpoints to retain
            metric_name: Metric name for best model selection
            mode: "min" or "max" for metric comparison
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_n = keep_best_n
        self.metric_name = metric_name
        self.mode = mode
        
        # Track best checkpoints
        self.best_checkpoints: List[Dict[str, Any]] = []
        self.best_metric_value = float('inf') if mode == "min" else float('-inf')
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        step: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        epoch: Optional[float] = None
    ) -> str:
        """
        Save complete training state.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            step: Training step
            metrics: Dictionary of evaluation metrics
            config: Training configuration
            epoch: Current epoch (optional)
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_step_{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = checkpoint_path / "model"
        model.save_pretrained(str(model_path))
        
        # Save optimizer state
        optimizer_path = checkpoint_path / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_path)
        
        # Save scheduler state if present
        if scheduler is not None:
            scheduler_path = checkpoint_path / "scheduler.pt"
            torch.save(scheduler.state_dict(), scheduler_path)
        
        # Save RNG states for reproducibility
        rng_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state['torch_cuda'] = torch.cuda.get_rng_state()
        
        rng_path = checkpoint_path / "rng_state.pt"
        torch.save(rng_state, rng_path)
        
        # Save metadata
        metadata = {
            'step': step,
            'epoch': epoch if epoch is not None else step,
            'metrics': metrics,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'checkpoint_name': checkpoint_name
        }
        
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Track best checkpoint
        metric_value = metrics.get(self.metric_name)
        if metric_value is not None:
            is_better = (
                (self.mode == "min" and metric_value < self.best_metric_value) or
                (self.mode == "max" and metric_value > self.best_metric_value)
            )
            
            if is_better:
                self.best_metric_value = metric_value
                self.best_checkpoints.append({
                    'path': str(checkpoint_path),
                    'step': step,
                    'metric': metric_value,
                    'metrics': metrics
                })
                
                # Keep only best N checkpoints
                if len(self.best_checkpoints) > self.keep_best_n:
                    # Remove oldest best checkpoint
                    oldest = self.best_checkpoints.pop(0)
                    if Path(oldest['path']).exists():
                        shutil.rmtree(oldest['path'])
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        restore_rng: bool = True
    ) -> Dict[str, Any]:
        """
        Load training state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            restore_rng: Whether to restore random number generator states
        
        Returns:
            Dictionary with loaded state information
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load model (HuggingFace format)
        model_path = checkpoint_path / "model"
        if model_path.exists():
            from transformers import WhisperForConditionalGeneration
            checkpoint_model = WhisperForConditionalGeneration.from_pretrained(
                str(model_path),
                map_location="cpu"
            )
            model.load_state_dict(checkpoint_model.state_dict(), strict=False)
        
        # Load optimizer state
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer is not None and optimizer_path.exists():
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        
        # Load scheduler state
        scheduler_path = checkpoint_path / "scheduler.pt"
        if scheduler is not None and scheduler_path.exists():
            scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
        
        # Restore RNG states
        if restore_rng:
            rng_path = checkpoint_path / "rng_state.pt"
            if rng_path.exists():
                rng_state = torch.load(rng_path, map_location="cpu")
                random.setstate(rng_state['python'])
                np.random.set_state(rng_state['numpy'])
                torch.set_rng_state(rng_state['torch_cpu'])
                if 'torch_cuda' in rng_state and torch.cuda.is_available():
                    torch.cuda.set_rng_state(rng_state['torch_cuda'])
        
        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return {
            'step': metadata.get('step', 0),
            'epoch': metadata.get('epoch', 0),
            'metrics': metadata.get('metrics', {}),
            'config': metadata.get('config', {})
        }
    
    def get_best_checkpoint(self) -> Optional[str]:
        """
        Return path to best checkpoint based on metric.
        
        Returns:
            Path to best checkpoint or None if no checkpoints saved
        """
        if not self.best_checkpoints:
            return None
        
        # Return most recent best checkpoint
        return self.best_checkpoints[-1]['path']
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint metadata dictionaries
        """
        checkpoints = []
        for checkpoint_dir in sorted(self.checkpoint_dir.iterdir()):
            if checkpoint_dir.is_dir():
                metadata_path = checkpoint_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    checkpoints.append(metadata)
        
        return checkpoints

