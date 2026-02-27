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
) -> torch.Tensor:
    """
    Stable sparsity penalty: push predictions for non-descendants toward obs baseline.

    Uses:
    - Relative threshold: 20% of per-universe interquartile range (scale-adaptive)
    - L1 penalty (not L2) — no squaring, can't explode
    - Hard clamp to [0, 5] — caps contribution from any single batch

    Args:
        pred_means:   [K, s, f]  model mean predictions per universe
        obs_means:    [s, f]     observational world values
        true_targets: [K, s, f]  ground-truth interventional values
    """
    true_deltas = (true_targets - obs_means.unsqueeze(0)).abs()   # [K, s, f]
    pred_deltas = (pred_means   - obs_means.unsqueeze(0)).abs()   # [K, s, f]

    # Stricter scale-adaptive threshold: 5% of the per-universe 75th-percentile delta
    # This forces the model to be much more sensitive to non-causal noise.
    scale = true_deltas.reshape(true_deltas.shape[0], -1).quantile(0.75, dim=-1)  # [K]
    threshold = (0.05 * scale).clamp(min=0.1)   # at least 0.1 absolute

    # Expand threshold to match [K, s, f]
    thr = threshold.view(-1, 1, 1).expand_as(true_deltas)
    non_causal_mask = true_deltas < thr

    if non_causal_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_means.device)

    # L1, not L2 — stable with any prediction magnitude
    penalty = pred_deltas[non_causal_mask].mean()
    return penalty.clamp(max=5.0)


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
        self.lambda_infer    = 1.0   # equal weight with train NLL — forces query encoder to dominate

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
        # [-25, 25]: headroom for do_val<=10 + mechanism amplification,
        # without the saturation at boundary that [-15,15] caused in infer mode.
        self.model.bar_head.init_fixed_borders(
            y_min=-25.0, y_max=25.0, device=torch.device(device)
        )

        # Logging
        self.log_dir        = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer  = SummaryWriter(str(self.log_dir))
        self.history = {
            "epoch": [], "loss": [], "nll": [], "nll_infer": [], 
            "sparsity": [], "gate_bce": [], "gate_density": [], "lr": [], "val_f1": []
        }


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

        # Anneal Gumbel-Sigmoid temperature: 2.0 → 0.1 over 500 epochs
        self.model.causal_gate.anneal_temperature(
            epoch, start_temp=2.0, end_temp=0.1, n_epochs=500
        )

        # ── Generate diverse SCM ────────────────────────────────────── #
        m_data, adj = generate_full_multiverse(
            self.n_samples,
            self.n_features,
            randomise_prior=True,
            device=self.device,
        )

        # ── Targets ─────────────────────────────────────────────────── #
        targets      = m_data[1:]   # [K, s, f]
        targets_flat = targets.reshape(self.n_features, self.n_samples * self.n_features)

        # ── Forward ─────────────────────────────────────────────────── #
        self.optimizer.zero_grad()
        logits, gated_means, gate, obs_ctx = self.model(m_data)   # 4-tuple

        # ── NLL loss ─────────────────────────────────────────────────── #
        nll = self.model.bar_head.nll_loss(
            logits.reshape(self.n_features * self.n_samples * self.n_features, -1),
            targets_flat.reshape(-1),
        )

        # ── Gate BCE loss: directly supervise gate with true adjacency ─ #
        # We have adj from the synthetic SCM — this is the key signal for the gate
        gate_bce = self.model.causal_gate.bce_loss(obs_ctx, adj)

        # ── Gate sparsity: prevent gate from predicting all-ones ──────── #
        gate_sp = self.model.causal_gate.sparsity_loss(obs_ctx, target_density=0.15)

        # ── Output sparsity loss ─────────────────────────────────────── #
        obs_world = m_data[0]
        sp_loss = sparsity_loss(gated_means, obs_world, targets)

        # ── Infer-mode NLL: train the obs-only pathway directly ──────── #
        # Pick one random intervention variable k for this batch.
        # Extract do_val from data: m_data[k+1, :, k] are all set to do_val exactly.
        k_infer     = torch.randint(0, self.n_features, (1,)).item()
        do_val_inf  = m_data[k_infer + 1, 0, k_infer].item()   # exact do_val used
        obs_data    = m_data[0]                                  # [s, f] obs only
        true_int_k  = m_data[k_infer + 1].reshape(-1)           # [s*f] true outcomes

        logits_inf, _ = self.model.infer(obs_data, k_infer, do_val_inf)
        nll_infer = self.model.bar_head.nll_loss(logits_inf, true_int_k)

        # ── Gate entropy: force decisions to be binary (0 or 1) ──────── #
        gate_ent = self.model.causal_gate.entropy_loss(obs_ctx)

        # ── Total loss ───────────────────────────────────────────────── #
        # NLL is primary; infer NLL trains obs-only pathway; BCE/sparsity are auxiliary
        loss = (nll
                + 0.3 * sp_loss        # Doubled: push non-descendants harder to 0
                + self.lambda_infer    * nll_infer
                + 1.0 * gate_bce       # Doubled: penalize incorrect edge logic more
                + 0.2 * gate_sp        # Doubled: encourage lower edge density
                + 0.15 * gate_ent)     # Increased: push harder for binary decisions

        # ── NaN guard ────────────────────────────────────────────────── #
        if not torch.isfinite(loss):
            self.optimizer.zero_grad()
            return {"loss": float("nan"), "nll": float("nan"), "nll_infer": float("nan"),
                    "sparsity": float("nan"), "gate_bce": float("nan"), "gate_density": float("nan")}

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

        gate_density = gate.detach().mean().item()
        stats = {
            "loss":         loss.item(),
            "nll":          nll.item(),
            "nll_infer":    nll_infer.item(),
            "sparsity":     sp_loss.item(),
            "gate_bce":     gate_bce.item(),
            "gate_density": gate_density,
        }

        # Log everything to history
        self.history["epoch"].append(epoch)
        for k, v in stats.items():
            if k in self.history:
                self.history[k].append(v)
        self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

        if (epoch + 1) % 100 == 0:
            self.writer.add_scalar("train/gate_bce", stats["gate_bce"], epoch)
            self.writer.add_scalar("train/gate_density", stats["gate_density"], epoch)
            self.writer.add_scalar("train/nll_infer", stats["nll_infer"], epoch)

        return stats

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def validate(self, n_eval_samples: int = 15, n_trials: int = 5) -> float:
        """
        Average F1 over n_trials independently sampled SCMs.
        A single random SCM can give wildly high/low F1 by luck.
        Averaging over 5 SCMs gives a stable signal for checkpointing.
        """
        self.model.eval()
        if self.evaluator is None:
            self.evaluator = CausalDiscoveryEvaluator(self.model, device=self.device)

        f1_scores = []
        with torch.no_grad():
            for _ in range(n_trials):
                metrics = self.evaluator.evaluate(
                    n_samples=n_eval_samples,
                    n_features=self.n_features,
                    do_val=10.0,
                    verbose=False,
                    mode="infer",
                )
                f1_scores.append(metrics["f1"])

        avg_f1 = sum(f1_scores) / len(f1_scores)
        self.model.train()
        self.evaluator = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return avg_f1

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
