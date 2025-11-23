"""
ML experiment tracking and metadata management.

Supports:
- Weights & Biases integration
- TensorBoard logging
- Local experiment catalog
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Track ML experiments with metadata and metrics."""

    def __init__(
        self,
        experiment_name: str,
        output_dir: Union[str, Path],
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
    ):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of experiment
            output_dir: Directory to save experiment data
            use_wandb: Whether to use Weights & Biases
            use_tensorboard: Whether to use TensorBoard
            wandb_project: W&B project name
            wandb_entity: W&B entity name
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard

        # Initialize W&B
        if use_wandb:
            try:
                import wandb

                wandb.init(
                    project=wandb_project or "whisper-finetuning",
                    entity=wandb_entity,
                    name=experiment_name,
                    dir=str(self.output_dir),
                )
                self.wandb = wandb
                logger.info("Weights & Biases initialized")
            except ImportError:
                logger.warning("wandb not installed, disabling W&B tracking")
                self.use_wandb = False
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
                self.use_wandb = False

        # Initialize TensorBoard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = self.output_dir / "tensorboard"
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
                logger.info(f"TensorBoard initialized at {tb_dir}")
            except ImportError:
                logger.warning("tensorboard not installed, disabling TensorBoard")
                self.use_tensorboard = False
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard: {e}")
                self.use_tensorboard = False

        # Experiment metadata
        self.metadata = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "output_dir": str(self.output_dir),
        }

    def log_config(self, config: Dict):
        """
        Log experiment configuration.

        Args:
            config: Configuration dictionary
        """
        self.metadata["config"] = config

        # Save to file
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Log to W&B
        if self.use_wandb:
            self.wandb.config.update(config)

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log metrics at a training step.

        Args:
            metrics: Dictionary of metric names and values
            step: Training step
        """
        # Log to W&B
        if self.use_wandb:
            self.wandb.log(metrics, step=step)

        # Log to TensorBoard
        if self.use_tensorboard:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)

    def log_model_artifact(self, model_path: Union[str, Path], artifact_name: str):
        """
        Log model as artifact.

        Args:
            model_path: Path to model
            artifact_name: Name for artifact
        """
        if self.use_wandb:
            artifact = self.wandb.Artifact(artifact_name, type="model")
            artifact.add_dir(str(model_path))
            self.wandb.log_artifact(artifact)

    def finish(self, metrics: Optional[Dict[str, float]] = None):
        """
        Finish experiment tracking.

        Args:
            metrics: Final metrics to log
        """
        self.metadata["end_time"] = datetime.now().isoformat()

        if metrics:
            self.metadata["final_metrics"] = metrics

        # Save metadata
        metadata_path = self.output_dir / "experiment_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        # Close W&B
        if self.use_wandb:
            self.wandb.finish()

        # Close TensorBoard
        if self.use_tensorboard:
            self.tb_writer.close()

        logger.info(f"Experiment tracking finished: {self.experiment_name}")

