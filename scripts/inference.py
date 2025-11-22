#!/usr/bin/env python3
"""CLI script for batch inference."""
import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.engine import WhisperInferenceEngine


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run batch inference on audio files")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--audio-files",
        type=str,
        nargs='+',
        help="List of audio file paths"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file for transcriptions"
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language code (optional)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference"
    )
    
    args = parser.parse_args()
    
    # Collect audio files
    audio_files = []
    if args.audio_dir:
        audio_dir = Path(args.audio_dir)
        audio_files.extend([
            str(f) for f in audio_dir.rglob('*.wav')
        ])
        audio_files.extend([
            str(f) for f in audio_dir.rglob('*.mp3')
        ])
        audio_files.extend([
            str(f) for f in audio_dir.rglob('*.flac')
        ])
    
    if args.audio_files:
        audio_files.extend(args.audio_files)
    
    if not audio_files:
        parser.error("No audio files found. Provide --audio-dir or --audio-files")
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Initialize inference engine
    engine = WhisperInferenceEngine(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Transcribe
    logger.info("Starting transcription...")
    results = engine.transcribe(
        audio_paths=audio_files,
        language=args.language
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Transcription completed. Results saved to: {output_path}")
    logger.info(f"Transcribed {len(results)} files")


if __name__ == "__main__":
    main()

