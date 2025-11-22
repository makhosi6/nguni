"""Experiment tracking and logging."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import wandb


logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Unified interface for experiment tracking.
    
    Supports:
    - Local file logging (JSON, CSV)
    - Weights & Biases integration
    """
    
    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        log_dir: str = "./logs",
        wandb_enabled: bool = False,
        wandb_project: Optional[str] = None
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration dictionary
            log_dir: Directory for local logs
            wandb_enabled: Enable Weights & Biases tracking
            wandb_project: W&B project name
        """
        self.experiment_name = experiment_name
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize W&B if enabled
        self.wandb_enabled = wandb_enabled
        if wandb_enabled:
            wandb.init(
                project=wandb_project or "whisper-sa-finetuning",
                name=experiment_name,
                config=config,
                dir=str(self.experiment_dir)
            )
        
        logger.info(f"Initialized experiment tracker: {experiment_name}")
        logger.info(f"Log directory: {self.experiment_dir}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log scalar metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Training step
        """
        # Log to file
        metrics_file = self.experiment_dir / "metrics.jsonl"
        with open(metrics_file, 'a') as f:
            log_entry = {'step': step, **metrics, 'timestamp': datetime.now().isoformat()}
            f.write(json.dumps(log_entry) + '\n')
        
        # Log to W&B
        if self.wandb_enabled:
            wandb.log(metrics, step=step)
        
        logger.debug(f"Logged metrics at step {step}: {metrics}")
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        if self.wandb_enabled:
            wandb.config.update(config)
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "checkpoint"):
        """
        Log file artifacts.
        
        Args:
            artifact_path: Path to artifact file
            artifact_type: Type of artifact (e.g., 'checkpoint', 'model')
        """
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            logger.warning(f"Artifact not found: {artifact_path}")
            return
        
        # Copy to experiment directory
        artifact_dest = self.experiment_dir / artifact_type / artifact_path.name
        artifact_dest.parent.mkdir(parents=True, exist_ok=True)
        
        if artifact_path.is_file():
            import shutil
            shutil.copy2(artifact_path, artifact_dest)
        elif artifact_path.is_dir():
            import shutil
            shutil.copytree(artifact_path, artifact_dest, dirs_exist_ok=True)
        
        # Log to W&B
        if self.wandb_enabled:
            wandb.log_artifact(str(artifact_path), type=artifact_type)
    
    def finish(self):
        """Finalize experiment tracking."""
        if self.wandb_enabled:
            wandb.finish()
        
        logger.info(f"Experiment tracking finished: {self.experiment_name}")

