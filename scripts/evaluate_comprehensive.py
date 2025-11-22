#!/usr/bin/env python3
"""CLI script for running comprehensive evaluation suite."""
import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.orchestrator import EvaluationOrchestrator, EvaluationConfig


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation suite"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs='+',
        required=True,
        help="Language codes to evaluate"
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        required=True,
        help="Path to baseline Whisper-large-v3 model"
    )
    parser.add_argument(
        "--fine-tuned-models",
        type=str,
        required=True,
        help="JSON file mapping model names to paths"
    )
    parser.add_argument(
        "--multilingual-model",
        type=str,
        help="Path to multilingual model (optional)"
    )
    parser.add_argument(
        "--per-language-models",
        type=str,
        help="JSON file mapping languages to per-language model paths"
    )
    parser.add_argument(
        "--test-set-dir",
        type=str,
        default="./test_sets",
        help="Directory for test sets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory for evaluation results"
    )
    parser.add_argument(
        "--create-test-sets",
        action="store_true",
        help="Create test sets from validation data"
    )
    parser.add_argument(
        "--val-data-paths",
        type=str,
        help="JSON file mapping languages to validation JSONL paths"
    )
    parser.add_argument(
        "--audio-base-path",
        type=str,
        help="Base directory for audio files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for evaluation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    
    args = parser.parse_args()
    
    # Load fine-tuned models
    with open(args.fine_tuned_models, 'r') as f:
        fine_tuned_models = json.load(f)
    
    # Load per-language models if provided
    per_language_models = None
    if args.per_language_models:
        with open(args.per_language_models, 'r') as f:
            per_language_models = json.load(f)
    
    # Load validation data paths if provided
    val_data_paths = None
    if args.val_data_paths:
        with open(args.val_data_paths, 'r') as f:
            val_data_paths = json.load(f)
    
    # Create configuration
    config = EvaluationConfig(
        languages=args.languages,
        baseline_model_path=args.baseline_model,
        fine_tuned_models=fine_tuned_models,
        multilingual_model_path=args.multilingual_model,
        per_language_models=per_language_models,
        test_set_dir=args.test_set_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Create orchestrator
    orchestrator = EvaluationOrchestrator(config)
    
    # Run evaluation
    logger.info("Starting comprehensive evaluation")
    results = orchestrator.run_full_evaluation(
        create_test_sets=args.create_test_sets,
        val_data_paths=val_data_paths,
        audio_base_path=args.audio_base_path
    )
    
    logger.info(f"Evaluation completed. Results saved to: {results.results_path}")
    
    # Print summary
    if results.baseline_comparison:
        summary = results.baseline_comparison.summary
        logger.info(f"Baseline comparison summary:")
        logger.info(f"  Comparisons: {summary['num_comparisons']}")
        logger.info(f"  Improvements: {summary['num_improvements']}")
        logger.info(f"  Avg improvement: {summary['avg_improvement_percent']:.2f}%")
        logger.info(f"  Significant improvements: {summary['num_significant_improvements']}")


if __name__ == "__main__":
    main()

