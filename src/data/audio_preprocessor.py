"""Audio preprocessing utilities for validation and normalization."""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional


def validate_audio(audio_path: str, min_duration: float = 0.5, max_duration: float = 30.0) -> Tuple[bool, str]:
    """
    Validate audio file format, duration, and quality.
    
    Args:
        audio_path: Path to audio file
        min_duration: Minimum audio duration in seconds
        max_duration: Maximum audio duration in seconds
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    audio_path = Path(audio_path)
    
    # Check file exists
    if not audio_path.exists():
        return False, f"Audio file not found: {audio_path}"
    
    # Check file extension
    valid_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    if audio_path.suffix.lower() not in valid_extensions:
        return False, f"Unsupported audio format: {audio_path.suffix}"
    
    try:
        # Try to load audio to check if it's readable
        duration = librosa.get_duration(path=str(audio_path))
        
        # Check duration
        if duration < min_duration:
            return False, f"Audio too short: {duration:.2f}s < {min_duration}s"
        
        if duration > max_duration:
            return False, f"Audio too long: {duration:.2f}s > {max_duration}s"
        
        return True, ""
    
    except Exception as e:
        return False, f"Error loading audio: {str(e)}"


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio amplitude to target dB level.
    
    Args:
        audio: Audio signal as numpy array
        target_db: Target RMS level in dB
    
    Returns:
        Normalized audio signal
    """
    # Peak normalization to [-1, 1]
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # RMS normalization to target dB
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        target_linear = 10 ** (target_db / 20.0)
        audio = audio * (target_linear / rms)
    
    # Clip to prevent clipping
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio


def detect_silence(audio: np.ndarray, threshold_db: float = -40.0, min_duration: float = 0.1) -> bool:
    """
    Detect if audio is mostly silence.
    
    Args:
        audio: Audio signal as numpy array
        threshold_db: RMS threshold in dB below which audio is considered silent
        min_duration: Minimum duration of silence to flag
    
    Returns:
        True if audio is mostly silent
    """
    if len(audio) == 0:
        return True
    
    # Compute RMS in dB
    rms = np.sqrt(np.mean(audio ** 2))
    rms_db = 20 * np.log10(rms + 1e-10)
    
    return rms_db < threshold_db


def load_audio(audio_path: str, sr: int = 16000, mono: bool = True) -> np.ndarray:
    """
    Load and resample audio file.
    
    Args:
        audio_path: Path to audio file
        sr: Target sampling rate
        mono: Convert to mono if True
    
    Returns:
        Audio signal as numpy array
    """
    audio, original_sr = librosa.load(audio_path, sr=sr, mono=mono)
    return audio


def add_noise(audio: np.ndarray, noise_type: str = 'white', snr_db: float = 20.0) -> np.ndarray:
    """
    Add noise to audio signal.
    
    Args:
        audio: Audio signal
        noise_type: Type of noise ('white', 'babble')
        snr_db: Signal-to-Noise Ratio in dB
    
    Returns:
        Noisy audio signal
    """
    # Calculate signal power
    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio
        
    # Calculate noise power based on SNR
    # SNR = 10 * log10(P_signal / P_noise)
    # P_noise = P_signal / (10 ** (SNR / 10))
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    if noise_type == 'white':
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    elif noise_type == 'babble':
        # Simulate babble by generating random frequencies
        # This is a simple approximation
        t = np.linspace(0, len(audio)/16000, len(audio))
        noise = np.zeros_like(audio)
        for _ in range(10):
            freq = np.random.uniform(100, 1000)
            phase = np.random.uniform(0, 2*np.pi)
            noise += np.sin(2 * np.pi * freq * t + phase)
        
        # Normalize to target power
        current_noise_power = np.mean(noise ** 2)
        if current_noise_power > 0:
            noise = noise * np.sqrt(noise_power / current_noise_power)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return audio + noise


def change_speed(audio: np.ndarray, speed_factor: float) -> np.ndarray:
    """
    Change audio speed (time stretching).
    
    Args:
        audio: Audio signal
        speed_factor: Speed factor (e.g., 0.9 for slower, 1.1 for faster)
    
    Returns:
        Time-stretched audio signal
    """
    if speed_factor == 1.0:
        return audio
        
    return librosa.effects.time_stretch(audio, rate=speed_factor)


