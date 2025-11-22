"""Model factory for creating and configuring Whisper models."""
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from typing import Optional, Tuple


class WhisperModelFactory:
    """Factory for creating and configuring Whisper models."""
    
    @staticmethod
    def create_model(
        model_name: str = "openai/whisper-large-v3",
        language: Optional[str] = None,
        gradient_checkpointing: bool = True,
        device: str = "cuda",
        from_checkpoint: Optional[str] = None
    ) -> WhisperForConditionalGeneration:
        """
        Create Whisper model with optimizations.
        
        Args:
            model_name: HuggingFace model identifier
            language: Language code for forced decoder IDs
            gradient_checkpointing: Enable gradient checkpointing
            device: Target device
            from_checkpoint: Path to checkpoint for transfer learning
        
        Returns:
            Configured Whisper model
        """
        # Load model from checkpoint or pretrained
        if from_checkpoint:
            model = WhisperForConditionalGeneration.from_pretrained(from_checkpoint)
        else:
            model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # Move to device
        model = model.to(device)
        
        return model
    
    @staticmethod
    def get_forced_decoder_ids(
        processor: WhisperProcessor,
        language: str,
        task: str = "transcribe"
    ) -> torch.Tensor:
        """
        Get language-specific decoder prompt IDs.
        
        Args:
            processor: WhisperProcessor instance
            language: Language code (e.g., 'af', 'nr', 'en')
            task: Task type ('transcribe' or 'translate')
        
        Returns:
            Tensor of forced decoder IDs
        """
        # Get forced decoder start tokens
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language,
            task=task
        )
        
        return torch.tensor(forced_decoder_ids, dtype=torch.long)
    
    @staticmethod
    def create_processor(
        model_name: str = "openai/whisper-large-v3"
    ) -> WhisperProcessor:
        """
        Create WhisperProcessor instance.
        
        Args:
            model_name: HuggingFace model identifier
        
        Returns:
            WhisperProcessor instance
        """
        return WhisperProcessor.from_pretrained(model_name)

