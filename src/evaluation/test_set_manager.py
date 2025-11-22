"""Test set management with balanced sampling and integrity verification."""
import json
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np


@dataclass
class TestSample:
    """Individual test sample."""
    audio_path: str
    reference_text: str
    audio_duration: float
    audio_checksum: str
    metadata: Optional[Dict] = None  # Demographics, quality, etc.


@dataclass
class TestSetMetadata:
    """Test set metadata."""
    creation_date: str
    source_dataset: str
    num_samples: int
    total_duration: float
    demographic_distribution: Optional[Dict] = None
    language: str
    checksum: str


@dataclass
class TestSet:
    """Test set with metadata."""
    language: str
    samples: List[TestSample]
    metadata: TestSetMetadata
    checksum: str


class TestSetManager:
    """
    Manages test set creation, loading, and validation.
    
    Ensures balanced sampling and data integrity.
    """
    
    def __init__(
        self,
        test_set_dir: str = "./test_sets",
        min_samples: int = 50,
        max_samples: int = 1000,
        seed: int = 42
    ):
        """
        Initialize test set manager.
        
        Args:
            test_set_dir: Directory for storing test sets
            min_samples: Minimum samples per language
            max_samples: Maximum samples per language
            seed: Random seed for reproducibility
        """
        self.test_set_dir = Path(test_set_dir)
        self.test_set_dir.mkdir(parents=True, exist_ok=True)
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def _compute_checksum(self, data: str) -> str:
        """Compute SHA-256 checksum."""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def _compute_file_checksum(self, file_path: Path) -> str:
        """Compute SHA-256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def create_test_set(
        self,
        language: str,
        val_data_path: str,
        audio_base_path: str,
        balance_by_metadata: Optional[str] = None
    ) -> TestSet:
        """
        Create balanced test set from validation data.
        
        Args:
            language: Language code
            val_data_path: Path to validation JSONL
            audio_base_path: Base directory for audio files
            balance_by_metadata: Metadata field to balance by (e.g., 'speaker', 'region')
        
        Returns:
            TestSet with samples and metadata
        """
        val_data_path = Path(val_data_path)
        audio_base_path = Path(audio_base_path)
        
        # Load validation data
        samples = []
        with open(val_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line.strip())
                if example.get('language') != language:
                    continue
                
                audio_path = audio_base_path / example['audio']
                if not audio_path.exists():
                    continue
                
                # Compute audio checksum
                try:
                    audio_checksum = self._compute_file_checksum(audio_path)
                except Exception:
                    continue
                
                # Get audio duration
                try:
                    import librosa
                    duration = librosa.get_duration(path=str(audio_path))
                except Exception:
                    duration = 0.0
                
                sample = TestSample(
                    audio_path=str(audio_path),
                    reference_text=example['text'],
                    audio_duration=duration,
                    audio_checksum=audio_checksum,
                    metadata=example.get('metadata')
                )
                samples.append(sample)
        
        # Balance sampling if metadata field provided
        if balance_by_metadata and samples:
            samples = self._balance_samples(samples, balance_by_metadata)
        
        # Limit to max_samples
        if len(samples) > self.max_samples:
            samples = random.sample(samples, self.max_samples)
        
        # Ensure minimum samples
        if len(samples) < self.min_samples:
            raise ValueError(
                f"Insufficient samples for {language}: "
                f"{len(samples)} < {self.min_samples}"
            )
        
        # Compute checksum
        samples_json = json.dumps([asdict(s) for s in samples], sort_keys=True)
        checksum = self._compute_checksum(samples_json)
        
        # Create metadata
        total_duration = sum(s.audio_duration for s in samples)
        demographic_dist = None
        if samples and samples[0].metadata:
            # Compute demographic distribution
            demographic_dist = {}
            for sample in samples:
                if sample.metadata:
                    for key, value in sample.metadata.items():
                        if key not in demographic_dist:
                            demographic_dist[key] = {}
                        demographic_dist[key][value] = (
                            demographic_dist[key].get(value, 0) + 1
                        )
        
        metadata = TestSetMetadata(
            creation_date=datetime.now().isoformat(),
            source_dataset=str(val_data_path),
            num_samples=len(samples),
            total_duration=total_duration,
            demographic_distribution=demographic_dist,
            language=language,
            checksum=checksum
        )
        
        test_set = TestSet(
            language=language,
            samples=samples,
            metadata=metadata,
            checksum=checksum
        )
        
        # Save test set
        self._save_test_set(test_set)
        
        return test_set
    
    def _balance_samples(
        self,
        samples: List[TestSample],
        metadata_field: str
    ) -> List[TestSample]:
        """Balance samples by metadata field."""
        # Group by metadata value
        groups = {}
        for sample in samples:
            if sample.metadata and metadata_field in sample.metadata:
                value = sample.metadata[metadata_field]
                if value not in groups:
                    groups[value] = []
                groups[value].append(sample)
        
        if not groups:
            return samples
        
        # Sample equally from each group
        min_group_size = min(len(group) for group in groups.values())
        target_per_group = min(min_group_size, self.max_samples // len(groups))
        
        balanced = []
        for group in groups.values():
            if len(group) > target_per_group:
                balanced.extend(random.sample(group, target_per_group))
            else:
                balanced.extend(group)
        
        return balanced
    
    def _save_test_set(self, test_set: TestSet):
        """Save test set to disk."""
        test_set_path = self.test_set_dir / f"{test_set.language}_test_set.jsonl"
        metadata_path = self.test_set_dir / f"{test_set.language}_metadata.json"
        
        # Save samples
        with open(test_set_path, 'w', encoding='utf-8') as f:
            for sample in test_set.samples:
                f.write(json.dumps(asdict(sample)) + '\n')
        
        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(test_set.metadata), f, indent=2)
    
    def load_test_set(self, language: str) -> TestSet:
        """
        Load test set with checksum verification.
        
        Args:
            language: Language code
        
        Returns:
            TestSet with verified integrity
        """
        test_set_path = self.test_set_dir / f"{language}_test_set.jsonl"
        metadata_path = self.test_set_dir / f"{language}_metadata.json"
        
        if not test_set_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Test set not found for language: {language}")
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
            metadata = TestSetMetadata(**metadata_dict)
        
        # Load samples
        samples = []
        with open(test_set_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample_dict = json.loads(line.strip())
                samples.append(TestSample(**sample_dict))
        
        # Verify checksum
        samples_json = json.dumps([asdict(s) for s in samples], sort_keys=True)
        computed_checksum = self._compute_checksum(samples_json)
        
        if computed_checksum != metadata.checksum:
            raise ValueError(f"Checksum mismatch for test set: {language}")
        
        test_set = TestSet(
            language=language,
            samples=samples,
            metadata=metadata,
            checksum=metadata.checksum
        )
        
        return test_set
    
    def create_cross_lingual_test_set(
        self,
        languages: List[str],
        val_data_paths: Dict[str, str],
        audio_base_path: str,
        samples_per_language: int = 200
    ) -> TestSet:
        """
        Create balanced multilingual test set.
        
        Args:
            languages: List of language codes
            val_data_paths: Dictionary mapping language to validation JSONL path
            audio_base_path: Base directory for audio files
            samples_per_language: Target samples per language
        
        Returns:
            TestSet with samples from all languages
        """
        all_samples = []
        
        for language in languages:
            if language not in val_data_paths:
                continue
            
            # Create per-language test set
            lang_test_set = self.create_test_set(
                language=language,
                val_data_path=val_data_paths[language],
                audio_base_path=audio_base_path
            )
            
            # Sample equally from each language
            if len(lang_test_set.samples) > samples_per_language:
                sampled = random.sample(
                    lang_test_set.samples,
                    samples_per_language
                )
            else:
                sampled = lang_test_set.samples
            
            all_samples.extend(sampled)
        
        # Create combined metadata
        total_duration = sum(s.audio_duration for s in all_samples)
        metadata = TestSetMetadata(
            creation_date=datetime.now().isoformat(),
            source_dataset="cross_lingual",
            num_samples=len(all_samples),
            total_duration=total_duration,
            demographic_distribution=None,
            language="multilingual",
            checksum=""  # Will be computed
        )
        
        # Compute checksum
        samples_json = json.dumps([asdict(s) for s in all_samples], sort_keys=True)
        checksum = self._compute_checksum(samples_json)
        metadata.checksum = checksum
        
        test_set = TestSet(
            language="multilingual",
            samples=all_samples,
            metadata=metadata,
            checksum=checksum
        )
        
        # Save
        self._save_test_set(test_set)
        
        return test_set
    
    def validate_test_set(self, test_set: TestSet) -> bool:
        """
        Verify test set integrity and checksums.
        
        Args:
            test_set: TestSet to validate
        
        Returns:
            True if valid
        """
        # Verify checksum
        samples_json = json.dumps([asdict(s) for s in test_set.samples], sort_keys=True)
        computed_checksum = self._compute_checksum(samples_json)
        
        if computed_checksum != test_set.checksum:
            return False
        
        # Verify audio files exist and checksums match
        for sample in test_set.samples:
            audio_path = Path(sample.audio_path)
            if not audio_path.exists():
                return False
            
            computed_audio_checksum = self._compute_file_checksum(audio_path)
            if computed_audio_checksum != sample.audio_checksum:
                return False
        
        return True

