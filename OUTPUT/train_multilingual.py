#!/usr/bin/env python3
"""
Multilingual balanced training script for Whisper-large-v3.

Trains a single model on multiple languages with balanced sampling.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from transformers import Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import WhisperDataset, collate_fn
from dataset_balancer import DatasetBalancer
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


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper-large-v3 on multilingual balanced dataset"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=".",
        help="Root directory of dataset",
    )
    parser.add_argument(
        "--dataset_summary",
        type=str,
        default="dataset_summary.json",
        help="Path to dataset_summary.json",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training_configs/multilingual.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--max_examples_per_lang",
        type=int,
        default=5000,
        help="Maximum examples per language for balancing",
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

    logger.info("Starting multilingual training")

    # Load configuration
    config_path = Path(args.config)
    logger.info(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize experiment tracking
    experiment_name = f"whisper-multilingual-{Path(args.output_dir).name}"
    tracker = ExperimentTracker(
        experiment_name=experiment_name,
        output_dir=output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )
    tracker.log_config(config)

    # Initialize resource monitor
    monitor = ResourceMonitor()

    # Initialize dataset balancer
    dataset_root = Path(args.dataset_root)
    dataset_summary_path = dataset_root / args.dataset_summary
    balancer = DatasetBalancer(dataset_summary_path, root_dir=dataset_root)

    # Create balanced datasets
    logger.info("Creating balanced multilingual dataset...")
    max_examples = args.max_examples_per_lang or config["data"].get(
        "max_examples_per_lang", 5000
    )

    # Create balanced train dataset
    balanced_train_path = output_dir / "balanced_train.jsonl"
    train_examples = balancer.create_balanced_dataset(
        max_examples_per_lang=max_examples,
        output_path=balanced_train_path,
        split="train",
    )

    # Create balanced val dataset
    balanced_val_path = output_dir / "balanced_val.jsonl"
    val_examples = balancer.create_balanced_dataset(
        max_examples_per_lang=max_examples // 10,  # 10% for validation
        output_path=balanced_val_path,
        split="val",
    )

    logger.info(
        f"Created balanced dataset: {len(train_examples)} train, {len(val_examples)} val"
    )

    # Initialize model factory
    model_factory = ModelFactory(
        model_id=config["model"]["model_id"],
        cache_dir=output_dir / "cache",
    )

    # Create model
    model = model_factory.create_model(
        gradient_checkpointing=config["model"]["gradient_checkpointing"]
    )

    # Initialize processor (no specific language for multilingual)
    processor = model_factory.create_processor()

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = WhisperDataset(
        balanced_train_path,
        processor,
        root_dir=dataset_root,
        max_audio_length=config["data"]["max_audio_length"],
        min_audio_length=config["data"]["min_audio_length"],
    )

    val_dataset = WhisperDataset(
        balanced_val_path,
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
        eval_steps=config["evaluation"]["eval_steps"],
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

    # Create data collator
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
        compute_metrics=compute_metrics,
    )

    # Start training
    logger.info("Starting multilingual training...")
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

    # Final evaluation
    logger.info("Running final evaluation...")
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        collate_fn=collate_fn,
        num_workers=config["training"].get("dataloader_num_workers", 4),
    )
    device = next(model.parameters()).device
    final_metrics = evaluator.evaluate_batch(model, val_loader, device)
    logger.info(f"Final metrics: {final_metrics}")
    tracker.log_metrics(final_metrics, step=config["training"]["max_steps"])

    # Save final model
    final_model_dir = output_dir / "final_model"
    model.save_pretrained(final_model_dir)
    processor.save_pretrained(final_model_dir)
    logger.info(f"Final model saved to {final_model_dir}")

    # Finish tracking
    tracker.finish(final_metrics)

    logger.info("Multilingual training pipeline completed")


if __name__ == "__main__":
    main()

