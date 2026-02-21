#!/usr/bin/env python3
"""
Full training script for C-PFN with configuration management
"""

import argparse
from pathlib import Path
import torch

from cpfn.models import MultiverseTransformer
from cpfn.training import Trainer
from cpfn.utils import Config, get_device


def main():
    parser = argparse.ArgumentParser(description="Train Causal Prior Function Network")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config file (will be created if not exists)",
    )
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--n-features", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    parser.add_argument("--val-interval", type=int, default=100, help="Validate every N epochs")

    args = parser.parse_args()

    # Load or create config
    config_path = Path(args.config)
    if config_path.exists():
        config = Config.load(args.config)
        print(f"Loaded config from {args.config}")
        # Only apply CLI overrides when explicitly provided (not default None)
        if args.num_epochs   is not None: config.num_epochs    = args.num_epochs
        if args.n_features   is not None: config.n_features    = args.n_features
        if args.n_samples    is not None: config.n_samples     = args.n_samples
        if args.learning_rate is not None: config.learning_rate = args.learning_rate
        if args.device != "auto":
            config.device = args.device
        # Always resolve "auto" to an actual device string
        if config.device in ("auto", None):
            config.device = get_device("auto")
    else:
        config = Config(
            num_epochs=args.num_epochs,
            n_features=args.n_features,
            n_samples=args.n_samples,
            learning_rate=args.learning_rate,
            device=get_device(args.device),
        )
        config.save(args.config)
        print(f"Created config at {args.config}")

    print(f"\nConfiguration:")
    for key, val in config.to_dict().items():
        print(f"  {key}: {val}")

    # Create model
    device = config.device
    model = MultiverseTransformer(
        n_features=config.n_features,
        embed_dim=config.embed_dim,
        n_heads=config.n_heads,
        n_decoder_layers=config.n_decoder_layers,
    )

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    # Create trainer
    trainer = Trainer(
        model=model,
        n_features=config.n_features,
        n_samples=config.n_samples,
        learning_rate=config.learning_rate,
        device=device,
        weight_decay=1e-4,  # L2 regularization
        log_dir=config.log_dir,
        checkpoint_dir=config.checkpoint_dir,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")

    # Train
    print(f"\nStarting training...")
    history = trainer.train(
        num_epochs=config.num_epochs,
        log_interval=config.log_interval,
        checkpoint_interval=config.checkpoint_interval,
        val_interval=args.val_interval,
        early_stopping=not args.no_early_stopping,
    )

    print(f"\nTraining complete!")
    print(f"Logs saved to: {trainer.log_dir}")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    if hasattr(trainer, 'best_epoch'):
        print(f"Best model: epoch {trainer.best_epoch+1} with F1={trainer.best_f1:.4f}")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")

    return trainer


if __name__ == "__main__":
    main()
