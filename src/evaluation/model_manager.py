"""Model management and inference for evaluation."""
import torch
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

from ..model.factory import WhisperModelFactory
from ..inference.engine import WhisperInferenceEngine


@dataclass
class PerformanceMetrics:
    """Performance metrics for inference."""
    throughput: float  # samples per second
    latency_p50: float  # milliseconds
    latency_p95: float  # milliseconds
    latency_p99: float  # milliseconds
    peak_memory_mb: float  # megabytes
    model_size_mb: float  # megabytes
    num_parameters: int


class ModelManager:
    """
    Manages model loading, inference, and performance profiling.
    
    Handles batch inference, feature caching, and performance measurement.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        use_fp16: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize model manager.
        
        Args:
            device: Device for inference
            use_fp16: Use FP16 for faster inference
            cache_dir: Directory for caching embeddings
        """
        self.device = device
        self.use_fp16 = use_fp16
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.models = {}
        self.cached_embeddings = {}
    
    def load_model(
        self,
        model_path: str,
        model_name: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Load model with optimization.
        
        Args:
            model_path: Path to model checkpoint
            model_name: Optional name for model (defaults to path)
        
        Returns:
            Loaded model
        """
        if model_name is None:
            model_name = Path(model_path).stem
        
        if model_name in self.models:
            return self.models[model_name]
        
        # Load model
        model = WhisperModelFactory.create_model(
            from_checkpoint=model_path,
            device=self.device
        )
        
        if self.use_fp16 and self.device == "cuda":
            model = model.half()
        
        model.eval()
        self.models[model_name] = model
        
        return model
    
    def batch_inference(
        self,
        model_name: str,
        audio_paths: List[str],
        language: Optional[str] = None,
        batch_size: int = 16,
        show_progress: bool = True
    ) -> List[str]:
        """
        Run batch inference on audio files.
        
        Args:
            model_name: Name of loaded model
            audio_paths: List of audio file paths
            language: Optional language code
            batch_size: Batch size for inference
            show_progress: Show progress bar
        
        Returns:
            List of transcriptions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        processor = WhisperModelFactory.create_processor()
        
        # Use inference engine for batch processing
        engine = WhisperInferenceEngine(
            model_path="",  # Not used, we pass model directly
            device=self.device,
            batch_size=batch_size
        )
        engine.model = model
        engine.processor = processor
        
        results = engine.transcribe(
            audio_paths=audio_paths,
            language=language
        )
        
        return [r['text'] for r in results]
    
    def profile_inference(
        self,
        model_name: str,
        audio_paths: List[str],
        language: Optional[str] = None,
        num_warmup: int = 5
    ) -> PerformanceMetrics:
        """
        Profile inference performance.
        
        Args:
            model_name: Name of loaded model
            audio_paths: List of audio files for profiling
            language: Optional language code
            num_warmup: Number of warmup iterations
        
        Returns:
            PerformanceMetrics with latency, throughput, memory
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        processor = WhisperModelFactory.create_processor()
        
        # Warmup
        for _ in range(num_warmup):
            _ = self.batch_inference(
                model_name,
                audio_paths[:1],
                language
            )
        
        # Measure latency
        latencies = []
        torch.cuda.synchronize() if self.device == "cuda" else None
        
        for audio_path in audio_paths:
            start_time = time.time()
            _ = self.batch_inference(
                model_name,
                [audio_path],
                language
            )
            torch.cuda.synchronize() if self.device == "cuda" else None
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        # Compute percentiles
        latencies = np.array(latencies)
        latency_p50 = np.percentile(latencies, 50)
        latency_p95 = np.percentile(latencies, 95)
        latency_p99 = np.percentile(latencies, 99)
        
        # Throughput
        total_time = np.sum(latencies) / 1000  # seconds
        throughput = len(audio_paths) / total_time if total_time > 0 else 0
        
        # Memory usage
        if self.device == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_memory_mb = 0.0
        
        # Model size
        model_size_mb = sum(
            p.numel() * (2 if self.use_fp16 else 4)
            for p in model.parameters()
        ) / (1024**2)
        
        num_parameters = sum(p.numel() for p in model.parameters())
        
        return PerformanceMetrics(
            throughput=throughput,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            peak_memory_mb=peak_memory_mb,
            model_size_mb=model_size_mb,
            num_parameters=num_parameters
        )
    
    def cache_embeddings(
        self,
        model_name: str,
        audio_paths: List[str],
        cache_key: Optional[str] = None
    ):
        """
        Cache encoder embeddings for reuse.
        
        Args:
            model_name: Name of loaded model
            audio_paths: List of audio file paths
            cache_key: Optional cache key (defaults to model_name)
        """
        if self.cache_dir is None:
            return
        
        if cache_key is None:
            cache_key = model_name
        
        # TODO: Implement encoder output caching
        # This would require extracting encoder features
        # and saving them for reuse across models
    
    def load_cached_embeddings(
        self,
        cache_key: str
    ) -> Optional[Dict]:
        """
        Load cached embeddings.
        
        Args:
            cache_key: Cache key
        
        Returns:
            Cached embeddings or None
        """
        if self.cache_dir is None:
            return None
        
        cache_path = self.cache_dir / f"{cache_key}_embeddings.pt"
        if cache_path.exists():
            return torch.load(cache_path)
        
        return None

