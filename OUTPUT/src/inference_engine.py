"""
Production inference pipeline for Whisper models.

Handles:
- Batch inference
- Language-specific decoding
- Audio preprocessing
- Output formatting
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Production inference engine for Whisper models."""

    def __init__(
        self,
        model_path: Union[str, Path],
        processor_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to model or HuggingFace model ID
            processor_path: Optional path to processor (uses model_path if None)
            device: Device to run inference on (auto-detects if None)
            torch_dtype: Model dtype for inference
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.torch_dtype = torch_dtype

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
        ).to(device)
        self.model.eval()

        # Load processor
        processor_path = processor_path or model_path
        logger.info(f"Loading processor from {processor_path}")
        self.processor = WhisperProcessor.from_pretrained(str(processor_path))

        logger.info(f"Inference engine initialized on {device}")

    def transcribe(
        self,
        audio: Union[np.ndarray, str, Path],
        language: Optional[str] = None,
        task: str = "transcribe",
        return_timestamps: bool = False,
        max_length: int = 448,
    ) -> Union[str, Dict]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio array, file path, or Path object
            language: Optional language code for forced decoding
            task: Task type ('transcribe' or 'translate')
            return_timestamps: Whether to return timestamps
            max_length: Maximum generation length

        Returns:
            Transcription string or dictionary with timestamps
        """
        # Load audio if path provided
        if isinstance(audio, (str, Path)):
            import librosa

            audio_path = Path(audio)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

        # Process audio
        inputs = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs["input_features"].to(self.device)

        # Get forced decoder ids if language specified
        forced_decoder_ids = None
        if language:
            try:
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=language, task=task
                )
            except Exception as e:
                logger.warning(f"Failed to get decoder prompt for {language}: {e}")

        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features,
                max_length=max_length,
                forced_decoder_ids=forced_decoder_ids,
                return_timestamps=return_timestamps,
            )

        # Decode
        transcription = self.processor.batch_decode(
            generated_ids, skip_special_tokens=not return_timestamps
        )[0]

        return transcription

    def transcribe_batch(
        self,
        audio_list: List[Union[np.ndarray, str, Path]],
        language: Optional[str] = None,
        batch_size: int = 8,
    ) -> List[str]:
        """
        Transcribe a batch of audio files.

        Args:
            audio_list: List of audio arrays or file paths
            language: Optional language code
            batch_size: Batch size for processing

        Returns:
            List of transcriptions
        """
        transcriptions = []

        for i in range(0, len(audio_list), batch_size):
            batch = audio_list[i : i + batch_size]
            batch_transcriptions = []

            for audio in batch:
                try:
                    transcription = self.transcribe(audio, language=language)
                    batch_transcriptions.append(transcription)
                except Exception as e:
                    logger.warning(f"Failed to transcribe audio: {e}")
                    batch_transcriptions.append("")

            transcriptions.extend(batch_transcriptions)

        return transcriptions

    def transcribe_file(
        self,
        jsonl_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        root_dir: Optional[Union[str, Path]] = None,
        language: Optional[str] = None,
    ) -> List[Dict]:
        """
        Transcribe all audio files in a JSONL file.

        Args:
            jsonl_path: Path to input JSONL file
            output_path: Optional path to save results JSONL
            root_dir: Root directory for audio paths
            language: Optional language code (overrides per-example language)

        Returns:
            List of transcription results
        """
        import json

        jsonl_path = Path(jsonl_path)
        root_dir = Path(root_dir) if root_dir else jsonl_path.parent.parent

        results = []

        with open(jsonl_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                example = json.loads(line)
                audio_path = root_dir / example["audio"]
                lang = language or example.get("language")

                try:
                    transcription = self.transcribe(audio_path, language=lang)
                    result = {
                        "audio": example["audio"],
                        "text": transcription,
                        "language": lang,
                        "reference": example.get("text", ""),
                    }
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to transcribe {audio_path}: {e}")
                    results.append(
                        {
                            "audio": example["audio"],
                            "text": "",
                            "language": lang,
                            "error": str(e),
                        }
                    )

        # Save results if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            logger.info(f"Saved transcriptions to {output_path}")

        return results

