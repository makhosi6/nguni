"""
Audio format normalization and validation for Whisper training.

Handles:
- Format conversion (16kHz, mono, WAV)
- Amplitude normalization
- Duration validation
- Quality checks
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Audio preprocessing and validation for Whisper models."""

    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1  # Mono
    MIN_DURATION = 0.3  # seconds
    MAX_DURATION = 30.0  # seconds
    NORMALIZE_AMPLITUDE = True
    TARGET_AMP_MAX = 0.95  # Normalize to 95% of max to avoid clipping

    def __init__(
        self,
        target_sr: int = TARGET_SAMPLE_RATE,
        target_channels: int = TARGET_CHANNELS,
        normalize: bool = NORMALIZE_AMPLITUDE,
        min_duration: float = MIN_DURATION,
        max_duration: float = MAX_DURATION,
    ):
        """
        Initialize audio preprocessor.

        Args:
            target_sr: Target sample rate (default: 16000)
            target_channels: Target number of channels (default: 1 for mono)
            normalize: Whether to normalize amplitude
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
        """
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.normalize = normalize
        self.min_duration = min_duration
        self.max_duration = max_duration

    def preprocess(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        validate: bool = True,
    ) -> Tuple[np.ndarray, bool]:
        """
        Preprocess audio file.

        Args:
            input_path: Path to input audio file
            output_path: Optional path to save preprocessed audio
            validate: Whether to validate audio after preprocessing

        Returns:
            Tuple of (audio_array, is_valid)
        """
        input_path = Path(input_path)

        if not input_path.exists():
            logger.error(f"Audio file not found: {input_path}")
            return np.array([]), False

        try:
            # Load audio
            audio, sr = librosa.load(
                str(input_path),
                sr=self.target_sr,
                mono=(self.target_channels == 1),
            )

            # Ensure mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=0)

            # Normalize amplitude
            if self.normalize:
                audio = self._normalize_amplitude(audio)

            # Validate
            if validate:
                is_valid = self.validate(audio)
                if not is_valid:
                    logger.warning(f"Audio validation failed for {input_path}")
                    return audio, False

            # Save if output path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(output_path), audio, self.target_sr)

            return audio, True

        except Exception as e:
            logger.error(f"Failed to preprocess {input_path}: {e}")
            return np.array([]), False

    def _normalize_amplitude(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio amplitude.

        Args:
            audio: Audio array

        Returns:
            Normalized audio array
        """
        if len(audio) == 0:
            return audio

        max_amp = np.abs(audio).max()
        if max_amp > 0:
            # Normalize to target amplitude
            audio = audio / max_amp * self.TARGET_AMP_MAX

        return audio

    def validate(self, audio: np.ndarray) -> bool:
        """
        Validate audio quality and duration.

        Args:
            audio: Audio array

        Returns:
            True if audio is valid
        """
        if len(audio) == 0:
            return False

        duration = len(audio) / self.target_sr

        # Check duration
        if duration < self.min_duration:
            logger.debug(f"Audio too short: {duration:.2f}s < {self.min_duration}s")
            return False

        if duration > self.max_duration:
            logger.debug(f"Audio too long: {duration:.2f}s > {self.max_duration}s")
            return False

        # Check for silence
        if np.abs(audio).max() < 1e-6:
            logger.debug("Audio is silent")
            return False

        # Check for NaN or Inf
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            logger.debug("Audio contains NaN or Inf")
            return False

        return True

    def get_audio_info(self, audio_path: Union[str, Path]) -> dict:
        """
        Get audio file information.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with audio metadata
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            return {"error": "File not found"}

        try:
            audio, sr = librosa.load(str(audio_path), sr=None, mono=False)
            duration = len(audio) / sr if len(audio.shape) == 1 else len(audio[0]) / sr

            info = {
                "sample_rate": sr,
                "channels": 1 if len(audio.shape) == 1 else audio.shape[0],
                "duration": duration,
                "samples": len(audio) if len(audio.shape) == 1 else audio.shape[1],
                "dtype": str(audio.dtype),
                "max_amplitude": float(np.abs(audio).max()),
            }

            return info

        except Exception as e:
            return {"error": str(e)}


def batch_preprocess(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    pattern: str = "*.wav",
    num_workers: int = 4,
) -> dict:
    """
    Batch preprocess audio files.

    Args:
        input_dir: Directory containing input audio files
        output_dir: Directory to save preprocessed files
        pattern: File pattern to match
        num_workers: Number of parallel workers

    Returns:
        Statistics dictionary
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list(input_dir.rglob(pattern))
    preprocessor = AudioPreprocessor()

    stats = {"total": len(audio_files), "processed": 0, "failed": 0, "skipped": 0}

    def process_file(input_file: Path) -> Tuple[str, bool]:
        """Process a single file."""
        rel_path = input_file.relative_to(input_dir)
        output_file = output_dir / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        _, is_valid = preprocessor.preprocess(input_file, output_file)
        return str(input_file), is_valid

    # Process files
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_file, f): f for f in audio_files}
            for future in as_completed(futures):
                try:
                    file_path, is_valid = future.result()
                    if is_valid:
                        stats["processed"] += 1
                    else:
                        stats["failed"] += 1
                except Exception as e:
                    logger.error(f"Error processing {futures[future]}: {e}")
                    stats["failed"] += 1
    else:
        for audio_file in audio_files:
            _, is_valid = preprocessor.preprocess(
                audio_file, output_dir / audio_file.relative_to(input_dir)
            )
            if is_valid:
                stats["processed"] += 1
            else:
                stats["failed"] += 1

    return stats

