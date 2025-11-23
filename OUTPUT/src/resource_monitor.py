"""
GPU/CPU utilization and resource monitoring.
"""

import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor system resources during training."""

    def __init__(self, monitor_gpu: bool = True, monitor_cpu: bool = True):
        """
        Initialize resource monitor.

        Args:
            monitor_gpu: Whether to monitor GPU
            monitor_cpu: Whether to monitor CPU
        """
        self.monitor_gpu = monitor_gpu
        self.monitor_cpu = monitor_cpu

        # Check GPU availability
        try:
            import torch

            self.has_gpu = torch.cuda.is_available()
            if self.has_gpu:
                self.num_gpus = torch.cuda.device_count()
                logger.info(f"GPU monitoring enabled: {self.num_gpus} GPU(s) available")
            else:
                logger.info("No GPU available, GPU monitoring disabled")
        except ImportError:
            self.has_gpu = False
            logger.warning("PyTorch not available, GPU monitoring disabled")

    def get_gpu_stats(self) -> Dict[str, float]:
        """
        Get GPU utilization statistics.

        Returns:
            Dictionary with GPU stats
        """
        if not self.has_gpu or not self.monitor_gpu:
            return {}

        try:
            import torch

            stats = {}
            for i in range(self.num_gpus):
                torch.cuda.set_device(i)
                memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
                memory_total = torch.cuda.get_device_properties(i).total_memory / (
                    1024 ** 3
                )  # GB

                stats[f"gpu_{i}_memory_allocated_gb"] = memory_allocated
                stats[f"gpu_{i}_memory_reserved_gb"] = memory_reserved
                stats[f"gpu_{i}_memory_total_gb"] = memory_total
                stats[f"gpu_{i}_memory_utilization"] = (
                    memory_reserved / memory_total if memory_total > 0 else 0.0
                )

            return stats

        except Exception as e:
            logger.warning(f"Failed to get GPU stats: {e}")
            return {}

    def get_cpu_stats(self) -> Dict[str, float]:
        """
        Get CPU utilization statistics.

        Returns:
            Dictionary with CPU stats
        """
        if not self.monitor_cpu:
            return {}

        try:
            import psutil

            stats = {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024 ** 3),
                "memory_available_gb": psutil.virtual_memory().available / (1024 ** 3),
                "memory_used_gb": psutil.virtual_memory().used / (1024 ** 3),
                "memory_percent": psutil.virtual_memory().percent,
            }

            return stats

        except ImportError:
            logger.warning("psutil not installed, CPU monitoring disabled")
            return {}
        except Exception as e:
            logger.warning(f"Failed to get CPU stats: {e}")
            return {}

    def get_all_stats(self) -> Dict[str, float]:
        """
        Get all resource statistics.

        Returns:
            Dictionary with all stats
        """
        stats = {}
        stats.update(self.get_gpu_stats())
        stats.update(self.get_cpu_stats())
        return stats

    def log_stats(self, step: int, tracker: Optional[object] = None):
        """
        Log resource statistics.

        Args:
            step: Current training step
            tracker: Optional experiment tracker to log to
        """
        stats = self.get_all_stats()

        if tracker and hasattr(tracker, "log_metrics"):
            tracker.log_metrics(stats, step)

        # Also log to logger
        if stats:
            gpu_info = ", ".join(
                [
                    f"GPU{i}: {stats.get(f'gpu_{i}_memory_utilization', 0)*100:.1f}%"
                    for i in range(self.num_gpus if self.has_gpu else 0)
                    if f"gpu_{i}_memory_utilization" in stats
                ]
            )
            cpu_info = f"CPU: {stats.get('cpu_percent', 0):.1f}%"
            logger.debug(f"Step {step} - {cpu_info}, {gpu_info}")

        return stats

