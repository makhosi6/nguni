"""Reproducibility utilities for deterministic training."""
import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42, deterministic: bool = False):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Enable deterministic CUDA operations (slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Enable deterministic algorithms (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
    else:
        # Faster but less reproducible
        torch.backends.cudnn.benchmark = True

