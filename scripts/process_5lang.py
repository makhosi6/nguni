#!/usr/bin/env python3
"""
Script for processing and organizing the 5lang corpus.
This script handles multilingual content and code-switching cases.
"""

import os
import logging
import shutil
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict
import csv
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LanguageSegment:
    """Represents a language segment in the audio."""
    start_time: float
    end_time: float
    language: str
    text: str
    speaker_id: str

@dataclass
class AudioMetadata:
    """Metadata for an audio file."""
    file_id: str
    duration: float
    languages: Set[str]
    speakers: Set[str]
    segments: List[LanguageSegment]

class FiveLangProcessor:
    """Processor for 5lang corpus data."""
    
    def __init__(self, source_dir: Path, output_dir: Path):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.languages = {
            'eng': 'en-ZA',
            'xho': 'xh-ZA',
            'zul': 'zu-ZA',
            'ssw': 'ss-ZA',
            'nbl': 'nr-ZA'
        }
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directory structure."""
        for lang_dir in self.languages.values():
            base_dir = self.output_dir / lang_dir
            for subdir in ['audio', 'transcriptions', 'metadata', 'alignments']:
                (base_dir / subdir).mkdir(parents=True, exist_ok=True)

    def detect_language_segments(self, transcript: str) -> List[Tuple[str, str]]:
        """
        Detect language segments in transcript.
        Returns list of (language_code, text) tuples.
        """
        # Example pattern for language tags, adjust based on actual format
        pattern = r'\[([A-Za-z]+)\](.*?)(?=\[[A-Za-z]+\]|$)'
        segments = re.findall(pattern, transcript, re.DOTALL)
        return [(lang.lower(), text.strip()) for lang, text in segments]

    def process_audio_file(self, audio_path: Path) -> AudioMetadata:
        """Process a single audio file and its transcription."""
        file_id = audio_path.stem
        transcript_path = audio_path.parent / f"{file_id}.txt"
        
        if not transcript_path.exists():
            logger.warning(f"No transcript found for {audio_path}")
            return None

        # Read transcript and detect language segments
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()

        segments = []
        current_time = 0.0
        for lang_code, text in self.detect_language_segments(transcript):
            if lang_code in self.languages:
                # Estimate duration based on text length (simplified)
                duration = len(text.split()) * 0.3  # rough estimate
                segment = LanguageSegment(
                    start_time=current_time,
                    end_time=current_time + duration,
                    language=lang_code,
                    text=text,
                    speaker_id=f"SPK_{file_id}"
                )
                segments.append(segment)
                current_time += duration

        return AudioMetadata(
            file_id=file_id,
            duration=current_time,
            languages={seg.language for seg in segments},
            speakers={seg.speaker_id},
            segments=segments
        )

    def split_by_language(self, audio_path: Path, metadata: AudioMetadata):
        """Split audio and transcriptions by language."""
        for lang_code in metadata.languages:
            if lang_code not in self.languages:
                continue

            output_lang_dir = self.output_dir / self.languages[lang_code]
            
            # Copy audio segments for this language
            # Note: In practice, you'd use audio processing libraries to actually split the audio
            lang_segments = [s for s in metadata.segments if s.language == lang_code]
            
            # Create language-specific transcription
            trans_path = output_lang_dir / 'transcriptions' / f"{metadata.file_id}_{lang_code}.txt"
            with open(trans_path, 'w', encoding='utf-8') as f:
                for segment in lang_segments:
                    f.write(f"[{segment.start_time:.2f}-{segment.end_time:.2f}] {segment.text}\n")

            # Create metadata
            meta_path = output_lang_dir / 'metadata' / f"{metadata.file_id}_{lang_code}.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'file_id': metadata.file_id,
                    'language': lang_code,
                    'segments': [asdict(s) for s in lang_segments],
                    'speakers': list(metadata.speakers)
                }, f, indent=2)

    def process_corpus(self):
        """Process the entire 5lang corpus."""
        try:
            audio_files = list(self.source_dir.glob('**/*.wav'))
            logger.info(f"Found {len(audio_files)} audio files to process")

            for audio_path in audio_files:
                logger.info(f"Processing {audio_path}")
                metadata = self.process_audio_file(audio_path)
                if metadata:
                    self.split_by_language(audio_path, metadata)

            self.generate_corpus_statistics()
            logger.info("5lang corpus processing completed successfully")

        except Exception as e:
            logger.error(f"Error processing 5lang corpus: {str(e)}")
            raise

    def generate_corpus_statistics(self):
        """Generate statistics about the processed corpus."""
        stats = defaultdict(lambda: {
            'total_files': 0,
            'total_duration': 0.0,
            'unique_speakers': set(),
            'total_segments': 0
        })

        for lang_code in self.languages.values():
            meta_dir = self.output_dir / lang_code / 'metadata'
            if not meta_dir.exists():
                continue

            for meta_file in meta_dir.glob('*.json'):
                with open(meta_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    lang = data['language']
                    stats[lang]['total_files'] += 1
                    stats[lang]['unique_speakers'].update(data['speakers'])
                    stats[lang]['total_segments'] += len(data['segments'])

        # Write statistics
        stats_path = self.output_dir / 'corpus_statistics.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({
                lang: {
                    'total_files': s['total_files'],
                    'unique_speakers': len(s['unique_speakers']),
                    'total_segments': s['total_segments']
                }
                for lang, s in stats.items()
            }, f, indent=2)

def main():
    """Main execution function."""
    try:
        source_dir = Path(__file__).parent.parent / 'PROCESSED' / '5lang'
        output_dir = Path(__file__).parent.parent / 'OUTPUT' / '5lang'
        
        # Initialize completion tracker
        from dataset_completion import DatasetCompletionTracker
        tracker = DatasetCompletionTracker(Path(__file__).parent.parent)
        
        # Only process if not already completed
        if not tracker.is_completed(source_dir):
            processor = FiveLangProcessor(source_dir, output_dir)
            processor.process_corpus()
            
            # Mark as completed after successful processing
            if tracker.mark_completed(source_dir):
                logger.info("5lang corpus processing completed and marked as done")
            else:
                logger.warning("Failed to mark 5lang corpus as completed")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
