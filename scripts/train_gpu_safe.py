#!/usr/bin/env python3
"""
Safe GPU Training with Checkpoint Recovery
Splits long training into shorter sessions to avoid CUDA memory corruption.

Usage:
    python scripts/train_gpu_safe.py --num-epochs 2000 --batch-size 200
    
This will run 200-epoch batches and reload the best model between batches,
preventing CUDA memory corruption that occurs with sustained 2000-epoch runs.
"""

import argparse
import torch
from pathlib import Path
from cpfn.models import MultiverseTransformer
from cpfn.training import Trainer
from cpfn.utils import Config, get_device
import json


def train_gpu_safe(
    num_epochs: int = 2000,
    batch_size: int = 200,  # Train in 200-epoch chunks
    device: str = "cuda",
):
    """
    Train in safe chunks to avoid CUDA memory corruption.
    
    Args:
        num_epochs: Total epochs to train
        batch_size: Epochs per training batch (smaller = safer)
        device: Device to use (cuda or cpu)
    """
    device = get_device(device)
    
    # Load config
    config = Config.load("config.json")
    config.device = device
    
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Safe GPU Training: {num_epochs} epochs in {batch_size}-epoch batches")
    print(f"Device: {device}")
    print(f"Total batches: {(num_epochs + batch_size - 1) // batch_size}\n")
    
    total_epochs = 0
    best_epoch = 0
    best_f1 = 0.0
    
    num_batches = (num_epochs + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_epochs = min(batch_size, num_epochs - total_epochs)
        print(f"{'='*80}")
        print(f"Batch {batch_idx + 1}/{num_batches}: Training {batch_epochs} epochs")
        print(f"Total progress: {total_epochs}/{num_epochs} epochs")
        print(f"{'='*80}\n")
        
        # Create fresh model for each batch (prevents memory leaks)
        model = MultiverseTransformer(
            n_features=config.n_features,
            embed_dim=config.embed_dim,
            n_heads=config.n_heads,
            n_decoder_layers=config.n_decoder_layers,
        )
        
        # Load best checkpoint if available
        best_checkpoint = checkpoint_dir / "best_model.pt"
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            best_f1 = checkpoint.get("best_f1", 0.0)
            best_epoch = checkpoint.get("best_epoch", 0)
            print(f"Loaded best model from epoch {best_epoch} (F1={best_f1:.4f})\n")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            n_features=config.n_features,
            n_samples=config.n_samples,
            learning_rate=config.learning_rate,
            device=device,
            log_dir=config.log_dir,
            checkpoint_dir=config.checkpoint_dir,
            weight_decay=1e-4,
        )
        
        # Set initial best F1 to avoid re-learning
        trainer.best_f1 = best_f1
        trainer.best_epoch = best_epoch
        
        # Train for this batch (disable validation to avoid CUDA corruption)
        history = trainer.train(
            num_epochs=batch_epochs,
            log_interval=100,
            checkpoint_interval=500,
            val_interval=batch_epochs + 1,  # Never validate (set beyond num_epochs)
            early_stopping=False,  # Disable early stopping without validation
        )
        
        # Update totals
        total_epochs += batch_epochs
        best_epoch = trainer.best_epoch
        best_f1 = trainer.best_f1
        
        print(f"\nBatch {batch_idx + 1} complete: Best F1={best_f1:.4f} at epoch {best_epoch}")
        
        # Clear CUDA memory between batches
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"Total epochs: {total_epochs}")
    print(f"Best F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safe GPU training with checkpoint recovery")
    parser.add_argument("--num-epochs", type=int, default=2000, help="Total epochs to train")
    parser.add_argument("--batch-size", type=int, default=200, help="Epochs per batch (smaller = safer)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    train_gpu_safe(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
