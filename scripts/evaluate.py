#!/usr/bin/env python3
"""CLI script for model evaluation."""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.factory import WhisperModelFactory
from src.model.evaluator import WhisperEvaluator
from src.data.dataset import WhisperDataset, collate_fn
from torch.utils.data import DataLoader


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--val-jsonl",
        type=str,
        required=True,
        help="Path to validation JSONL file"
    )
    parser.add_argument(
        "--audio-base-path",
        type=str,
        required=True,
        help="Base directory for audio files"
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language code (optional, for forced decoder IDs)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for evaluation"
    )
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    model = WhisperModelFactory.create_model(
        from_checkpoint=args.model_path,
        device=args.device
    )
    processor = WhisperModelFactory.create_processor()
    
    # Create dataset
    dataset = WhisperDataset(
        jsonl_path=args.val_jsonl,
        audio_base_path=args.audio_base_path,
        language_code=args.language,
        validate=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Evaluate
    evaluator = WhisperEvaluator(model, processor, device=args.device)
    metrics = evaluator.evaluate(dataloader, language=args.language)
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"WER: {metrics['wer']:.4f}")
    print(f"CER: {metrics['cer']:.4f}")
    print(f"Number of examples: {metrics['num_examples']}")
    print("="*50)


if __name__ == "__main__":
    main()

