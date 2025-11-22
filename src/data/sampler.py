"""Balanced multilingual sampler for training."""
import random
from typing import Dict, List, Iterator
from torch.utils.data import Dataset, Sampler


class BalancedMultilingualSampler(Sampler):
    """
    Sampler that ensures balanced language representation.
    
    Caps high-resource languages and ensures single-language batches
    for forced decoder ID consistency.
    """
    
    def __init__(
        self,
        datasets: Dict[str, Dataset],
        max_examples_per_lang: int = 5000,
        batch_size: int = 16,
        shuffle: bool = True
    ):
        """
        Initialize balanced multilingual sampler.
        
        Args:
            datasets: Dictionary mapping language codes to datasets
            max_examples_per_lang: Maximum samples per language
            batch_size: Batch size for single-language batches
            shuffle: Whether to shuffle within each language
        """
        self.datasets = datasets
        self.max_examples_per_lang = max_examples_per_lang
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Create indices for each language
        self.language_indices = {}
        self.language_offsets = {}
        
        offset = 0
        for lang_code, dataset in datasets.items():
            # Cap examples per language
            dataset_size = min(len(dataset), max_examples_per_lang)
            indices = list(range(dataset_size))
            
            if shuffle:
                random.shuffle(indices)
            
            self.language_indices[lang_code] = indices
            self.language_offsets[lang_code] = offset
            offset += dataset_size
        
        # Create round-robin batches
        self.batches = self._create_batches()
    
    def _create_batches(self) -> List[List[int]]:
        """Create batches with single-language examples."""
        batches = []
        language_codes = list(self.datasets.keys())
        
        # Create batches for each language
        for lang_code in language_codes:
            indices = self.language_indices[lang_code]
            offset = self.language_offsets[lang_code]
            
            # Create batches of batch_size
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                # Add offset to get global indices
                global_indices = [idx + offset for idx in batch_indices]
                batches.append(global_indices)
        
        # Shuffle batches if enabled
        if self.shuffle:
            random.shuffle(batches)
        
        return batches
    
    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches."""
        for batch in self.batches:
            yield batch
    
    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.batches)

