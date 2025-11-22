"""Pydantic models for configuration validation."""
from typing import Optional, List
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Model configuration."""
    name: str = Field(default="openai/whisper-large-v3", description="HuggingFace model identifier")
    gradient_checkpointing: bool = Field(default=True, description="Enable gradient checkpointing")
    
    class Config:
        extra = "forbid"


class OptimizationConfig(BaseModel):
    """Optimization configuration."""
    mixed_precision: bool = Field(default=True, description="Enable mixed precision training (FP16)")
    optimizer: str = Field(default="adamw", description="Optimizer type")
    lr_scheduler: str = Field(default="linear", description="Learning rate scheduler type")
    
    class Config:
        extra = "forbid"


class TrainingConfig(BaseModel):
    """Training configuration."""
    batch_size: int = Field(..., gt=0, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(..., gt=0, description="Gradient accumulation steps")
    learning_rate: float = Field(..., gt=0, le=1.0, description="Learning rate")
    max_steps: int = Field(..., gt=0, description="Maximum training steps")
    warmup_steps: int = Field(..., ge=0, description="Warmup steps")
    weight_decay: float = Field(default=0.01, ge=0, le=1.0, description="Weight decay")
    max_grad_norm: float = Field(default=1.0, gt=0, description="Maximum gradient norm for clipping")
    eval_steps: int = Field(..., gt=0, description="Evaluation frequency in steps")
    save_steps: int = Field(..., gt=0, description="Checkpoint saving frequency in steps")
    
    @validator('warmup_steps')
    def warmup_less_than_max_steps(cls, v, values):
        if 'max_steps' in values and v >= values['max_steps']:
            raise ValueError('warmup_steps must be less than max_steps')
        return v
    
    class Config:
        extra = "forbid"


class DataConfig(BaseModel):
    """Data configuration."""
    train_jsonl: Optional[str] = Field(None, description="Path to training JSONL file")
    val_jsonl: Optional[str] = Field(None, description="Path to validation JSONL file")
    audio_base_path: str = Field(..., description="Base directory for audio files")
    language_code: Optional[str] = Field(None, description="Language code (e.g., 'af', 'nr')")
    language_codes: Optional[List[str]] = Field(None, description="List of language codes for multilingual")
    max_audio_length: float = Field(default=30.0, gt=0, description="Maximum audio length in seconds")
    num_workers: int = Field(default=4, ge=0, description="Number of data loading workers")
    pin_memory: bool = Field(default=True, description="Pin memory for faster GPU transfer")
    cache_dir: Optional[str] = Field(None, description="Directory for feature caching")
    max_examples_per_lang: Optional[int] = Field(None, description="Maximum examples per language for multilingual training")
    
    class Config:
        extra = "forbid"


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    metric: str = Field(default="wer", description="Primary evaluation metric")
    early_stopping_patience: Optional[int] = Field(None, ge=1, description="Early stopping patience")
    
    class Config:
        extra = "forbid"


class LoggingConfig(BaseModel):
    """Logging configuration."""
    log_level: str = Field(default="INFO", description="Logging level")
    log_dir: str = Field(default="./logs", description="Logging directory")
    wandb_project: Optional[str] = Field(None, description="Weights & Biases project name")
    wandb_enabled: bool = Field(default=False, description="Enable Weights & Biases tracking")
    
    class Config:
        extra = "forbid"


class ReproducibilityConfig(BaseModel):
    """Reproducibility configuration."""
    seed: int = Field(default=42, description="Random seed")
    deterministic: bool = Field(default=False, description="Enable deterministic operations (slower)")
    
    class Config:
        extra = "forbid"


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""
    experiment_name: str = Field(..., description="Experiment name")
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig
    data: DataConfig
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)
    device: str = Field(default="cuda", description="Device for training")
    
    class Config:
        extra = "forbid"

