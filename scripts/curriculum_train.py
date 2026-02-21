#!/usr/bin/env python3
"""
Curriculum Learning Training Script
Progressively train on 5, 8, 10+ features to find theoretical limits
"""

import argparse
from pathlib import Path
import torch
import json

from cpfn.models import MultiverseTransformer
from cpfn.training import Trainer
from cpfn.utils import Config, get_device


def train_curriculum(
    n_features_list=[5, 8, 10, 15],
    num_epochs=2000,
    learning_rate=1e-4,
    device="auto",
    val_interval=50,
):
    """
    Curriculum learning: train progressively on more features.
    
    Args:
        n_features_list: List of feature counts to train on
        num_epochs: Epochs per feature count
        learning_rate: Learning rate
        device: Device to use
        val_interval: Validation interval
    """
    device = get_device(device)
    results = {}

    print("=" * 80)
    print("CURRICULUM LEARNING: Progressive Feature Scaling")
    print("=" * 80)

    for i, n_features in enumerate(n_features_list, 1):
        print(f"\n{'='*80}")
        print(f"STAGE {i}/{len(n_features_list)}: Training on {n_features} features")
        print(f"{'='*80}\n")

        # Reduce samples for 8+ features to avoid GPU OOM
        n_samples = 30 if n_features <= 5 else 20
        if n_samples != 30:
            print(f"(Using {n_samples} samples to reduce GPU memory usage)\n")

        # Create config for this stage
        config = Config(
            n_features=n_features,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
        )

        # Create model with reduced embed_dim for 8+ features to avoid numerical issues
        embed_dim = 96 if n_features >= 8 else 128
        model = MultiverseTransformer(
            n_features=n_features,
            embed_dim=embed_dim,
            n_heads=4 if n_features >= 8 else 8,  # Fewer heads for stability
            n_decoder_layers=4,
        )

        print(f"Model: {model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Device: {device}\n")

        # Create trainer with regularization
        trainer = Trainer(
            model=model,
            n_features=n_features,
            n_samples=n_samples,  # Use reduced samples for 8+ features
            learning_rate=learning_rate,
            device=device,
            log_dir=f"logs/curriculum/{n_features}_features",
            checkpoint_dir=f"checkpoints/curriculum/{n_features}_features",
            weight_decay=1e-4,  # L2 regularization
        )

        # Train (skip validation for 8+ features due to GPU memory constraints)
        use_validation = n_features <= 5
        effective_val_interval = val_interval if use_validation else num_epochs + 1
        
        print(f"Starting training for {num_epochs} epochs...")
        if not use_validation:
            print(f"(Validation disabled for {n_features} features to avoid GPU OOM)\n")
        else:
            print()
        
        history = trainer.train(
            num_epochs=num_epochs,
            log_interval=100,
            checkpoint_interval=500,
            val_interval=effective_val_interval,
            early_stopping=use_validation,
        )

        # Store results
        results[n_features] = {
            "best_epoch": trainer.best_epoch,
            "best_f1": trainer.best_f1,
            "model_params": sum(p.numel() for p in model.parameters()),
            "history": history,
        }

        print(f"\n{'='*80}")
        print(f"STAGE {i} COMPLETE")
        print(f"{'='*80}")
        print(f"Best F1: {trainer.best_f1:.4f} at epoch {trainer.best_epoch+1}")
        print(f"Checkpoint saved: checkpoints/curriculum/{n_features}_features/best_model.pt\n")

    # Save curriculum results
    output_file = Path("curriculum_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("CURRICULUM LEARNING COMPLETE")
    print("=" * 80)
    print("\nCURRICULUM LEARNING RESULTS:")
    print("━" * 80)
    print(f"{'Features':<15}{'Best F1':<15}{'Best Epoch':<15}{'Parameters':<15}")
    print("━" * 80)

    for n_features in n_features_list:
        if n_features in results:
            r = results[n_features]
            print(
                f"{n_features:<15}{r['best_f1']:<15.4f}{r['best_epoch']+1:<15}{r['model_params']:<15,}"
            )

    print("━" * 80)
    print(f"\nResults saved to: {output_file}")

    # Analyze scaling
    f1_scores = [results[f]["best_f1"] for f in n_features_list if f in results]
    if len(f1_scores) > 1:
        print("\nFEATURE SCALING ANALYSIS:")
        print("━" * 80)

        # Find degradation point
        best_idx = f1_scores.index(max(f1_scores))
        print(f"Peak performance: {n_features_list[best_idx]} features (F1={f1_scores[best_idx]:.4f})")

        if best_idx < len(f1_scores) - 1:
            degradation = f1_scores[best_idx] - f1_scores[-1]
            print(f"Degradation to {n_features_list[-1]} features: {degradation:.4f}")
            print(f"→ Theoretical limit appears to be around {n_features_list[best_idx]} features")
        else:
            print(f"→ No degradation detected up to {n_features_list[-1]} features")
            print(f"→ Theoretical limit may be higher than tested range")

        print("━" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum learning on variable feature counts")
    parser.add_argument(
        "--features",
        type=int,
        nargs="+",
        default=[5, 8, 10, 15],
        help="Feature counts to train on (default: 5 8 10 15)",
    )
    parser.add_argument("--num-epochs", type=int, default=2000, help="Epochs per stage")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--val-interval", type=int, default=50, help="Validation interval")

    args = parser.parse_args()

    train_curriculum(
        n_features_list=args.features,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        val_interval=args.val_interval,
    )
