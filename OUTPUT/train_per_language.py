#!/usr/bin/env python3
"""
Language-specific fine-tuning script for Whisper-large-v3.

Supports:
- High-resource languages (nr, tn, ts, xh, zu)
- Medium-resource languages (af)
- Low-resource languages (en, ss) with transfer learning
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from transformers import (
    Trainer,
    TrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from checkpoint_manager import CheckpointManager
from data_loader import WhisperDataset, collate_fn
from evaluator import ASREvaluator
from experiment_tracker import ExperimentTracker
from logging_config import setup_logging
from model_factory import ModelFactory
from resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load training configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_config_for_language(language_code: str, config_dir: Path) -> Path:
    """Get appropriate config file for language based on resource level."""
    # Language categorization
    high_resource = ["nr", "tn", "ts", "xh", "zu"]
    medium_resource = ["af"]
    low_resource = ["en", "ss"]

    if language_code in high_resource:
        return config_dir / "high_resource.yaml"
    elif language_code in medium_resource:
        return config_dir / "medium_resource.yaml"
    elif language_code in low_resource:
        return config_dir / "low_resource.yaml"
    else:
        raise ValueError(f"Unknown language code: {language_code}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper-large-v3 for a specific language"
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Language code (e.g., 'nr', 'tn', 'af')",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=".",
        help="Root directory of dataset",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML (auto-selected if not provided)",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="training_configs",
        help="Directory containing training configs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for tracking",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="whisper-finetuning",
        help="W&B project name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=output_dir, log_level="INFO")

    logger.info(f"Starting training for language: {args.language}")

    # Load configuration
    if args.config:
        config_path = Path(args.config)
    else:
        config_dir = Path(args.config_dir)
        config_path = get_config_for_language(args.language, config_dir)

    logger.info(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize experiment tracking
    experiment_name = f"whisper-{args.language}-{Path(args.output_dir).name}"
    tracker = ExperimentTracker(
        experiment_name=experiment_name,
        output_dir=output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )
    tracker.log_config(config)

    # Initialize resource monitor
    monitor = ResourceMonitor()

    # Initialize model factory
    model_factory = ModelFactory(
        model_id=config["model"]["model_id"],
        cache_dir=output_dir / "cache",
    )

    # Load or create model
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        model = model_factory.load_from_checkpoint(
            args.resume_from,
            gradient_checkpointing=config["model"]["gradient_checkpointing"],
        )
    else:
        # Check for base model for transfer learning
        base_model = config["model"].get("base_model")
        if base_model and base_model == "multilingual_finetuned":
            # Load from multilingual model (assumes it exists)
            multilingual_path = output_dir.parent / "multilingual" / "best_model"
            if multilingual_path.exists():
                logger.info(f"Loading base model from {multilingual_path}")
                model = model_factory.load_from_checkpoint(
                    multilingual_path,
                    gradient_checkpointing=config["model"]["gradient_checkpointing"],
                )
            else:
                logger.warning(
                    f"Multilingual model not found at {multilingual_path}, using base model"
                )
                model = model_factory.create_model(
                    gradient_checkpointing=config["model"]["gradient_checkpointing"]
                )
        else:
            model = model_factory.create_model(
                gradient_checkpointing=config["model"]["gradient_checkpointing"]
            )

    # Initialize processor
    processor = model_factory.create_processor(language=args.language)

    # Setup data paths
    dataset_root = Path(args.dataset_root)
    lang_dir = dataset_root / f"{args.language}-ZA"
    train_jsonl = lang_dir / "transcriptions" / "train.jsonl"
    val_jsonl = lang_dir / "transcriptions" / "val.jsonl"

    if not train_jsonl.exists():
        raise FileNotFoundError(f"Training data not found: {train_jsonl}")

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = WhisperDataset(
        train_jsonl,
        processor,
        root_dir=dataset_root,
        max_audio_length=config["data"]["max_audio_length"],
        min_audio_length=config["data"]["min_audio_length"],
    )

    val_dataset = None
    if val_jsonl.exists():
        val_dataset = WhisperDataset(
            val_jsonl,
            processor,
            root_dir=dataset_root,
            max_audio_length=config["data"]["max_audio_length"],
            min_audio_length=config["data"]["min_audio_length"],
        )

    # Initialize evaluator
    evaluator = ASREvaluator(processor)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        max_steps=config["training"]["max_steps"],
        warmup_steps=config["training"]["warmup_steps"],
        lr_scheduler_type=config["training"]["lr_scheduler"],
        weight_decay=config["training"]["weight_decay"],
        max_grad_norm=config["training"]["max_grad_norm"],
        fp16=config["training"]["fp16"],
        dataloader_num_workers=config["training"].get("dataloader_num_workers", 4),
        dataloader_pin_memory=config["training"].get("dataloader_pin_memory", True),
        eval_strategy=config["evaluation"]["eval_strategy"],
        eval_steps=config["evaluation"]["eval_steps"] if val_loader else None,
        save_strategy=config["checkpointing"]["save_strategy"],
        save_steps=config["checkpointing"]["save_steps"],
        save_total_limit=config["checkpointing"]["save_total_limit"],
        load_best_model_at_end=config["evaluation"]["load_best_model_at_end"],
        metric_for_best_model=config["evaluation"]["metric_for_best_model"],
        greater_is_better=config["evaluation"]["greater_is_better"],
        logging_steps=50,
        report_to="wandb" if args.use_wandb else "none",
        seed=args.seed,
    )

    # Create data collator wrapper
    from transformers import DataCollatorWithPadding
    
    class WhisperDataCollator:
        """Data collator for Whisper using custom collate function."""
        def __call__(self, features):
            return collate_fn(features)
    
    # Create trainer with custom compute_metrics
    def compute_metrics(eval_pred):
        """Compute WER/CER metrics."""
        predictions, labels = eval_pred
        # Decode predictions and labels
        pred_str = processor.batch_decode(predictions, skip_special_tokens=True)
        labels[labels == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels, skip_special_tokens=True)
        
        # Compute metrics
        metrics = evaluator.compute_metrics(pred_str, label_str)
        return metrics

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
        data_collator=WhisperDataCollator(),
        compute_metrics=compute_metrics if val_dataset else None,
    )

    # Start training
    logger.info("Starting training...")
    try:
        trainer.train(resume_from_checkpoint=args.resume_from)
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

    # Final evaluation
    if val_dataset:
        logger.info("Running final evaluation...")
        from torch.utils.data import DataLoader
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            collate_fn=collate_fn,
            num_workers=config["training"].get("dataloader_num_workers", 4),
        )
        device = next(model.parameters()).device
        final_metrics = evaluator.evaluate_batch(
            model, val_loader, device, language=args.language
        )
        logger.info(f"Final metrics: {final_metrics}")
        tracker.log_metrics(final_metrics, step=config["training"]["max_steps"])

    # Save final model
    final_model_dir = output_dir / "final_model"
    model.save_pretrained(final_model_dir)
    processor.save_pretrained(final_model_dir)
    logger.info(f"Final model saved to {final_model_dir}")

    # Finish tracking
    tracker.finish(final_metrics if val_loader else None)

    logger.info("Training pipeline completed")


if __name__ == "__main__":
    main()

