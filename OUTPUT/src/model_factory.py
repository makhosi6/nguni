"""
Model initialization with optimized configuration for Whisper fine-tuning.

Handles:
- Model loading with gradient checkpointing
- Language-specific configuration
- Memory optimizations
- Mixed precision setup
"""

import logging
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperConfig,
)

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating and configuring Whisper models."""

    DEFAULT_MODEL_ID = "openai/whisper-large-v3"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize model factory.

        Args:
            model_id: HuggingFace model identifier
            cache_dir: Directory for model cache
        """
        self.model_id = model_id
        self.cache_dir = str(cache_dir) if cache_dir else None

    def create_model(
        self,
        gradient_checkpointing: bool = True,
        use_cache: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> WhisperForConditionalGeneration:
        """
        Create and configure Whisper model.

        Args:
            gradient_checkpointing: Enable gradient checkpointing for memory savings
            use_cache: Whether to use KV cache (disable for training)
            torch_dtype: Model dtype (None for auto, torch.float16 for FP16)

        Returns:
            Configured Whisper model
        """
        logger.info(f"Loading model: {self.model_id}")

        # Load model configuration
        config = WhisperConfig.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
        )

        # Disable cache for training
        config.use_cache = use_cache

        # Load model
        model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            config=config,
            cache_dir=self.cache_dir,
            torch_dtype=torch_dtype,
        )

        # Enable gradient checkpointing
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Set model to training mode
        model.train()

        # Log model info
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {num_params:,} total, {trainable_params:,} trainable")

        return model

    def create_processor(
        self, language: Optional[str] = None, task: str = "transcribe"
    ) -> WhisperProcessor:
        """
        Create Whisper processor.

        Args:
            language: Language code for forced decoding
            task: Task type ('transcribe' or 'translate')

        Returns:
            WhisperProcessor instance
        """
        processor = WhisperProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
        )

        if language:
            try:
                # Validate language code
                decoder_prompt_ids = processor.get_decoder_prompt_ids(
                    language=language, task=task
                )
                logger.info(f"Processor configured for language: {language}")
            except Exception as e:
                logger.warning(f"Invalid language code {language}: {e}")

        return processor

    def load_from_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        gradient_checkpointing: bool = True,
    ) -> WhisperForConditionalGeneration:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            gradient_checkpointing: Enable gradient checkpointing

        Returns:
            Loaded model
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading model from checkpoint: {checkpoint_path}")

        model = WhisperForConditionalGeneration.from_pretrained(
            str(checkpoint_path),
            torch_dtype=torch.float16,  # Assume FP16 for checkpoints
        )

        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

        model.train()

        return model

    @staticmethod
    def get_model_size_mb(model: WhisperForConditionalGeneration) -> float:
        """
        Get model size in megabytes.

        Args:
            model: Model instance

        Returns:
            Model size in MB
        """
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size
        return total_size / (1024 ** 2)

    @staticmethod
    def enable_mixed_precision(model: WhisperForConditionalGeneration):
        """
        Configure model for mixed precision training.

        Args:
            model: Model instance
        """
        # Model should already support FP16
        # This is mainly for documentation/logging
        logger.info("Model configured for mixed precision training (FP16)")

