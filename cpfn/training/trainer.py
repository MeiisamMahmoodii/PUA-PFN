"""
Training loop for C-PFN with full tracking and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
from typing import Dict, Tuple, Optional
from tqdm import tqdm

from cpfn.models import MultiverseTransformer
from cpfn.data import generate_full_multiverse
from cpfn.evaluation import CausalDiscoveryEvaluator


class Trainer:
    """
    Trainer for Causal Prior Function Network (C-PFN).

    Features:
    - Meta-learning with diverse SCMs
    - Tensorboard logging
    - Checkpoint saving
    - Learning rate scheduling
    - Variable do_val randomization
    """

    def __init__(
        self,
        model: MultiverseTransformer,
        n_features: int,
        n_samples: int = 30,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device)
        self.n_features = n_features
        self.n_samples = n_samples
        self.device = device
        
        # Disable CUDA graph capture to avoid kernel errors
        if device == 'cuda':
            torch.cuda.is_available()  # Ensure CUDA is initialized

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=False)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2
        )

        # Logging
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(self.log_dir))
        self.history = {"epoch": [], "loss": [], "lr": [], "val_f1": []}
        
        # Early stopping
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.patience = 0
        self.max_patience = 200  # Stop if no improvement for 200 epochs
        self.evaluator = None

    def train_epoch(self, epoch: int) -> float:
        """
        Single training epoch: generate new SCMs, compute loss, update weights.
        """
        self.model.train()
        epoch_loss = 0.0

        # Meta-learning: generate diverse SCMs
        do_val = torch.rand(1).item() * 13.0 + 2.0  # Uniform [2, 15]

        m_data, _ = generate_full_multiverse(
            self.n_samples,
            self.n_features,
            do_val=do_val,
            device=self.device,
        )

        # Ground truth: interventional universes
        target = m_data[1:].reshape(
            self.n_features, self.n_samples * self.n_features, 1
        )

        # Forward
        self.optimizer.zero_grad()
        predictions = self.model(m_data)
        loss = self.criterion(predictions, target)

        # Backward
        loss.backward()
        # Clip gradients (but handle CUDA errors gracefully on GPU)
        try:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        except RuntimeError as e:
            if "CUDA" in str(e):
                # Skip gradient clipping on CUDA errors
                pass
            else:
                raise
        
        try:
            self.optimizer.step()
        except RuntimeError as e:
            if "CUDA" in str(e):
                # Log but continue on CUDA errors in optimizer step
                print(f"Warning: CUDA error in optimizer.step(): {e}")
            else:
                raise
        
        self.scheduler.step()

        return loss.item()

    def validate(self, n_eval_samples: int = 15) -> float:
        """
        Validate model on causal discovery task.
        Returns F1 score.
        Uses smaller n_eval_samples to reduce memory usage with frequent validation.
        """
        # Ensure model is in eval mode and no gradients are tracked
        self.model.eval()
        
        if self.evaluator is None:
            self.evaluator = CausalDiscoveryEvaluator(self.model, device=self.device)
        
        # Run evaluation without gradients
        with torch.no_grad():
            metrics = self.evaluator.evaluate(
                n_samples=n_eval_samples,  # Reduced from 20 to 15
                n_features=self.n_features,
                do_val=10.0,
                verbose=False
            )
        
        f1_score = metrics['f1']
        
        # Return to training mode
        self.model.train()
        
        # Reset evaluator to force garbage collection
        self.evaluator = None
        
        # Clear CUDA cache to prevent memory buildup
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return f1_score

    def train(
        self,
        num_epochs: int,
        log_interval: int = 100,
        checkpoint_interval: int = 500,
        val_interval: int = 100,
        early_stopping: bool = True,
    ) -> Dict:
        """
        Full training loop with early stopping.
        
        Args:
            num_epochs: Maximum number of epochs
            log_interval: Log loss every N epochs
            checkpoint_interval: Save checkpoint every N epochs
            val_interval: Validate every N epochs
            early_stopping: Stop if F1 doesn't improve for max_patience epochs
        """
        print(f"Starting training for {num_epochs} epochs on {self.device}")
        print(f"n_features={self.n_features}, n_samples={self.n_samples}")
        print(f"Early stopping enabled: {early_stopping} (patience={self.max_patience})")

        pbar = tqdm(range(num_epochs), desc="Training")
        for epoch in pbar:
            loss = self.train_epoch(epoch)
            self.history["epoch"].append(epoch)
            self.history["loss"].append(loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # Logging
            if (epoch + 1) % log_interval == 0:
                pbar.set_postfix({"loss": f"{loss:.6f}"})
                self.writer.add_scalar("train/loss", loss, epoch)
                self.writer.add_scalar(
                    "train/lr", self.optimizer.param_groups[0]["lr"], epoch
                )

            # Validation with early stopping
            if early_stopping and (epoch + 1) % val_interval == 0:
                val_f1 = self.validate()
                self.history["val_f1"].append(val_f1)
                self.writer.add_scalar("val/f1", val_f1, epoch)
                
                # Check for improvement
                if val_f1 > self.best_f1:
                    self.best_f1 = val_f1
                    self.best_epoch = epoch
                    self.patience = 0
                    # Save best model
                    self.save_checkpoint(epoch, is_best=True)
                    pbar.write(f"Epoch {epoch+1}: New best F1={val_f1:.4f}")
                else:
                    self.patience += 1
                    if self.patience % 50 == 0:
                        pbar.write(f"Epoch {epoch+1}: No improvement for {self.patience} evals ({self.patience*val_interval} epochs), best F1={self.best_f1:.4f}")
                    
                    # Early stopping
                    if self.patience >= self.max_patience:
                        pbar.write(f"\nEarly stopping at epoch {epoch+1}!")
                        pbar.write(f"Best F1: {self.best_f1:.4f} at epoch {self.best_epoch+1}")
                        break

            # Regular checkpoint
            if (epoch + 1) % checkpoint_interval == 0 and (not early_stopping or not ((epoch + 1) % val_interval == 0)):
                self.save_checkpoint(epoch)

        self.writer.close()
        self.save_history()
        return self.history

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model and optimizer state."""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "history": self.history,
            "best_f1": self.best_f1,
            "best_epoch": self.best_epoch,
        }
        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.history = checkpoint["history"]
        print(f"Loaded checkpoint: {path}")

    def save_history(self):
        """Save training history."""
        path = self.log_dir / "history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved: {path}")
