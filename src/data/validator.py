"""Data validation and quality checks."""
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from .audio_preprocessor import validate_audio


class DataValidator:
    """Validates dataset integrity and generates statistics."""
    
    def __init__(
        self,
        audio_base_path: str,
        min_duration: float = 0.5,
        max_duration: float = 30.0
    ):
        """
        Initialize data validator.
        
        Args:
            audio_base_path: Base directory for audio files
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
        """
        self.audio_base_path = Path(audio_base_path)
        self.min_duration = min_duration
        self.max_duration = max_duration
    
    def validate_jsonl(self, jsonl_path: str) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Validate JSONL file and return valid examples with statistics.
        
        Args:
            jsonl_path: Path to JSONL file
        
        Returns:
            Tuple of (valid_examples, statistics_dict)
        """
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        
        valid_examples = []
        stats = defaultdict(int)
        stats['total_examples'] = 0
        stats['skipped_examples'] = 0
        stats['missing_audio'] = 0
        stats['invalid_duration'] = 0
        stats['invalid_transcription'] = 0
        stats['total_duration'] = 0.0
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                stats['total_examples'] += 1
                
                try:
                    example = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    stats['skipped_examples'] += 1
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
                
                # Validate required fields
                if 'audio' not in example or 'text' not in example:
                    stats['skipped_examples'] += 1
                    continue
                
                # Validate audio file
                audio_path = self.audio_base_path / example['audio']
                is_valid, error_msg = validate_audio(
                    str(audio_path),
                    min_duration=self.min_duration,
                    max_duration=self.max_duration
                )
                
                if not is_valid:
                    stats['skipped_examples'] += 1
                    if 'not found' in error_msg.lower():
                        stats['missing_audio'] += 1
                    elif 'too short' in error_msg.lower() or 'too long' in error_msg.lower():
                        stats['invalid_duration'] += 1
                    continue
                
                # Validate transcription
                text = example['text']
                if not text or not isinstance(text, str) or len(text.strip()) == 0:
                    stats['skipped_examples'] += 1
                    stats['invalid_transcription'] += 1
                    continue
                
                # Check for valid Unicode
                try:
                    text.encode('utf-8')
                except UnicodeEncodeError:
                    stats['skipped_examples'] += 1
                    stats['invalid_transcription'] += 1
                    continue
                
                # Calculate duration
                try:
                    import librosa
                    duration = librosa.get_duration(path=str(audio_path))
                    stats['total_duration'] += duration
                except Exception:
                    pass
                
                valid_examples.append(example)
        
        # Calculate average duration
        if len(valid_examples) > 0:
            stats['avg_duration'] = stats['total_duration'] / len(valid_examples)
        else:
            stats['avg_duration'] = 0.0
        
        stats['valid_examples'] = len(valid_examples)
        
        return valid_examples, dict(stats)
    
    def generate_report(self, jsonl_path: str) -> Dict[str, any]:
        """
        Generate validation report for dataset.
        
        Args:
            jsonl_path: Path to JSONL file
        
        Returns:
            Dictionary with validation statistics
        """
        valid_examples, stats = self.validate_jsonl(jsonl_path)
        
        report = {
            'jsonl_path': str(jsonl_path),
            'statistics': stats,
            'validation_passed': stats['skipped_examples'] == 0
        }
        
        return report

