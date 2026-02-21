"""
Training loop for PUA-PFN (Parallel Universe PFN).

Key changes from original C-PFN trainer:
- NLL loss on bar-distribution (replaces MSE)
- Sparsity loss: penalises non-zero predictions for non-descendants
- Rich SCM prior via randomise_prior=True in generate_full_multiverse
- Evaluates both train-mode and infer-mode F1
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
from typing import Dict, Optional
from tqdm import tqdm

from cpfn.models import MultiverseTransformer
from cpfn.data import generate_full_multiverse
from cpfn.evaluation import CausalDiscoveryEvaluator


# ──────────────────────────────────────────────── #
#  Losses                                          #
# ──────────────────────────────────────────────── #

def sparsity_loss(
    pred_means: torch.Tensor,
    obs_means:  torch.Tensor,
    true_targets: torch.Tensor,
    threshold: float = 0.3,
) -> torch.Tensor:
    """
    Push model predictions towards zero for non-causal (non-descendant) variables.

    For each variable whose true interventional value is close to the
    observational value (i.e., no causal effect), penalise the model
    for predicting a large deviation from the observational baseline.

    Args:
        pred_means:   [K, s, f]  model's mean predictions per universe
        obs_means:    [s, f]     observational means (broadcast over K)
        true_targets: [K, s, f]  ground-truth interventional values
        threshold:    if |true_delta| < threshold → treat as zero-effect

    Returns:
        scalar sparsity penalty
    """
    true_deltas = (true_targets - obs_means.unsqueeze(0)).abs()  # [K, s, f]
    pred_deltas = (pred_means   - obs_means.unsqueeze(0)).abs()  # [K, s, f]

    non_causal_mask = true_deltas < threshold
    penalty = pred_deltas[non_causal_mask].pow(2).mean()
    return penalty


# ──────────────────────────────────────────────── #
#  Trainer                                         #
# ──────────────────────────────────────────────── #

class Trainer:
    """
    Trainer for PUA-PFN (Parallel Universe PFN).

    Training objectives:
    1. NLL loss — bar-distribution over interventional outcomes
    2. Sparsity loss — suppress predictions for non-descendants
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
        lambda_sparsity: float = 0.05,
    ):
        self.model           = model.to(device)
        self.n_features      = n_features
        self.n_samples       = n_samples
        self.device          = device
        self.lambda_sparsity = lambda_sparsity

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            foreach=False,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2
        )

        # Fix bar-distribution borders once (stable NLL scale across all batches).
        # Range [-30, 30] covers: do_val in [2,15] + mechanism amplification + obs noise.
        self.model.bar_head.init_fixed_borders(
            y_min=-30.0, y_max=30.0, device=torch.device(device)
        )

        # Logging
        self.log_dir        = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer  = SummaryWriter(str(self.log_dir))
        self.history = {"epoch": [], "loss": [], "nll": [], "sparsity": [], "lr": [], "val_f1": []}

        # Early stopping
        self.best_f1      = 0.0
        self.best_epoch   = 0
        self.patience     = 0
        self.max_patience = 200
        self.evaluator    = None

    # ------------------------------------------------------------------ #
    #  Single epoch                                                        #
    # ------------------------------------------------------------------ #

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        # ── Generate diverse SCM ────────────────────────────────────── #
        m_data, _ = generate_full_multiverse(
            self.n_samples,
            self.n_features,
            randomise_prior=True,          # rich prior — all hyperparams randomised
            device=self.device,
        )

        # ── Targets ─────────────────────────────────────────────────── #
        # Ground truth interventional universes: [K, n_samples, n_features]
        targets = m_data[1:]                  # [K, s, f]
        targets_flat = targets.reshape(self.n_features, self.n_samples * self.n_features)
        # Note: bar-dist borders are fixed globally (set in __init__); no per-batch update.

        # ── Forward ─────────────────────────────────────────────────── #
        self.optimizer.zero_grad()
        logits = self.model(m_data)           # [K, s*f, n_bins]

        # ── NLL loss ─────────────────────────────────────────────────── #
        nll = self.model.bar_head.nll_loss(
            logits.reshape(self.n_features * self.n_samples * self.n_features, -1),
            targets_flat.reshape(-1),
        )

        # ── Sparsity loss ────────────────────────────────────────────── #
        pred_means = self.model.bar_head.mean(logits)                # [K, s*f]
        pred_means = pred_means.view(self.n_features, self.n_samples, self.n_features)
        obs_world  = m_data[0]                                        # [s, f]
        sp_loss = sparsity_loss(pred_means, obs_world, targets)

        # ── Total loss ───────────────────────────────────────────────── #
        loss = nll + self.lambda_sparsity * sp_loss

        # ── Backward ─────────────────────────────────────────────────── #
        loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        except RuntimeError as e:
            if "CUDA" not in str(e):
                raise
        try:
            self.optimizer.step()
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"Warning: CUDA error in optimizer.step(): {e}")
            else:
                raise

        self.scheduler.step()

        return {
            "loss":     loss.item(),
            "nll":      nll.item(),
            "sparsity": sp_loss.item(),
        }

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def validate(self, n_eval_samples: int = 15) -> float:
        self.model.eval()
        if self.evaluator is None:
            self.evaluator = CausalDiscoveryEvaluator(self.model, device=self.device)

        with torch.no_grad():
            metrics = self.evaluator.evaluate(
                n_samples=n_eval_samples,
                n_features=self.n_features,
                do_val=10.0,
                verbose=False,
                mode="train",
            )

        f1 = metrics["f1"]
        self.model.train()
        self.evaluator = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return f1

    # ------------------------------------------------------------------ #
    #  Full training loop                                                  #
    # ------------------------------------------------------------------ #

    def train(
        self,
        num_epochs: int,
        log_interval: int = 100,
        checkpoint_interval: int = 500,
        val_interval: int = 100,
        early_stopping: bool = True,
    ) -> Dict:
        print(f"Starting PUA-PFN training for {num_epochs} epochs on {self.device}")
        print(f"n_features={self.n_features}, n_samples={self.n_samples}")
        print(f"lambda_sparsity={self.lambda_sparsity}, early_stopping={early_stopping}")

        pbar = tqdm(range(num_epochs), desc="Training")
        for epoch in pbar:
            stats = self.train_epoch(epoch)
            loss, nll, sp = stats["loss"], stats["nll"], stats["sparsity"]

            self.history["epoch"].append(epoch)
            self.history["loss"].append(loss)
            self.history["nll"].append(nll)
            self.history["sparsity"].append(sp)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            if (epoch + 1) % log_interval == 0:
                pbar.set_postfix({"NLL": f"{nll:.4f}", "sp": f"{sp:.4f}"})
                self.writer.add_scalar("train/loss",     loss, epoch)
                self.writer.add_scalar("train/nll",      nll,  epoch)
                self.writer.add_scalar("train/sparsity", sp,   epoch)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch)

            if early_stopping and (epoch + 1) % val_interval == 0:
                val_f1 = self.validate()
                self.history["val_f1"].append(val_f1)
                self.writer.add_scalar("val/f1", val_f1, epoch)

                if val_f1 > self.best_f1:
                    self.best_f1    = val_f1
                    self.best_epoch = epoch
                    self.patience   = 0
                    self.save_checkpoint(epoch, is_best=True)
                    pbar.write(f"Epoch {epoch+1}: New best F1={val_f1:.4f}")
                else:
                    self.patience += 1
                    if self.patience % 50 == 0:
                        pbar.write(
                            f"Epoch {epoch+1}: No improvement for "
                            f"{self.patience} evals, best F1={self.best_f1:.4f}"
                        )
                    if self.patience >= self.max_patience:
                        pbar.write(f"\nEarly stopping at epoch {epoch+1}!")
                        break

            if (epoch + 1) % checkpoint_interval == 0:
                if not (early_stopping and (epoch + 1) % val_interval == 0):
                    self.save_checkpoint(epoch)

        self.writer.close()
        self.save_history()
        return self.history

    # ------------------------------------------------------------------ #
    #  Checkpointing                                                       #
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "history":         self.history,
            "best_f1":         self.best_f1,
            "best_epoch":      self.best_epoch,
        }
        path = (
            self.checkpoint_dir / "best_model.pt"
            if is_best
            else self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.history = ckpt["history"]
        print(f"Loaded checkpoint: {path}")

    def save_history(self):
        path = self.log_dir / "history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved: {path}")
