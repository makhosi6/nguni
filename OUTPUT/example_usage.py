#!/usr/bin/env python3
"""
Example usage of the Whisper fine-tuning pipeline.

This script demonstrates:
1. Training a single language model
2. Running inference
3. Evaluating model performance
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from evaluator import ASREvaluator
from inference_engine import InferenceEngine
from transformers import WhisperProcessor


def example_inference():
    """Example: Run inference on a trained model."""
    print("=" * 80)
    print("Example 1: Inference")
    print("=" * 80)

    # Initialize inference engine
    model_path = "models/per_language/nr-ZA/best_model"
    
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please train a model first using train_per_language.py")
        return

    engine = InferenceEngine(model_path)

    # Transcribe a single audio file
    audio_file = "nr-ZA/audio/example.wav"
    if Path(audio_file).exists():
        transcription = engine.transcribe(audio_file, language="nr")
        print(f"Transcription: {transcription}")
    else:
        print(f"Audio file not found: {audio_file}")

    # Batch transcription
    audio_files = ["file1.wav", "file2.wav"]
    transcriptions = engine.transcribe_batch(audio_files, language="nr")
    for audio, text in zip(audio_files, transcriptions):
        print(f"{audio}: {text}")


def example_evaluation():
    """Example: Evaluate a trained model."""
    print("=" * 80)
    print("Example 2: Evaluation")
    print("=" * 80)

    model_path = "models/per_language/nr-ZA/best_model"
    
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return

    # Load model and processor
    from transformers import WhisperForConditionalGeneration
    
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Initialize evaluator
    evaluator = ASREvaluator(processor)

    # Evaluate on validation set
    val_jsonl = "nr-ZA/transcriptions/val.jsonl"
    if Path(val_jsonl).exists():
        metrics = evaluator.evaluate_file(
            val_jsonl,
            processor=processor,
            device=device,
            root_dir=".",
            max_samples=100,  # Limit for quick demo
        )
        
        print(f"WER: {metrics.get('wer', 0):.4f}")
        print(f"CER: {metrics.get('cer', 0):.4f}")
        
        # Per-language metrics if available
        for key, value in metrics.items():
            if key.startswith("wer_") or key.startswith("cer_"):
                print(f"{key}: {value:.4f}")
    else:
        print(f"Validation file not found: {val_jsonl}")


def example_training_command():
    """Example: Show training command."""
    print("=" * 80)
    print("Example 3: Training Commands")
    print("=" * 80)

    print("\n1. Train a single language model:")
    print("   python train_per_language.py \\")
    print("       --language nr \\")
    print("       --dataset_root . \\")
    print("       --output_dir models/nr-ZA \\")
    print("       --use_wandb")

    print("\n2. Train multilingual model:")
    print("   python train_multilingual.py \\")
    print("       --dataset_root . \\")
    print("       --dataset_summary dataset_summary.json \\")
    print("       --output_dir models/multilingual \\")
    print("       --use_wandb")

    print("\n3. Run complete pipeline:")
    print("   python train_orchestrator.py \\")
    print("       --dataset_root . \\")
    print("       --dataset_summary dataset_summary.json \\")
    print("       --output_dir models \\")
    print("       --use_wandb")


if __name__ == "__main__":
    print("\nWhisper Fine-Tuning Pipeline - Example Usage\n")
    
    # Show training commands
    example_training_command()
    
    # Try inference if model exists
    example_inference()
    
    # Try evaluation if model exists
    example_evaluation()
    
    print("\n" + "=" * 80)
    print("For more information, see TRAINING_README.md")
    print("=" * 80)

