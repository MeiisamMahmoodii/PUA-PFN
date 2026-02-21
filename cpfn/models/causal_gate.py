"""
Gumbel-Sigmoid Causal Gate for PUA-PFN.

Learns a differentiable binary mask over the K×K causal edge matrix.
During training: soft gate via Gumbel-Sigmoid (differentiable, explores structures).
During inference: hard threshold at 0.5 (discrete edge prediction).

This replaces the heuristic gap-threshold in the evaluator with a
principled learned sparse gate — solving the precision problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalGate(nn.Module):
    """
    Learned sparse causal gate using Gumbel-Sigmoid.

    Maintains a [K, K] matrix of edge logits α_ij.
    During training:  gate_ij = sigmoid((α_ij + Gumbel) / τ)  — soft, differentiable
    During inference: gate_ij = (α_ij > 0).float()            — hard binary

    Usage:
        gate = CausalGate(n_features=5)
        # During training (forward pass of transformer):
        soft_mask = gate(hard=False)          # [K, K] in (0,1)
        # At eval time:
        edge_probs = gate.edge_probs()        # [K, K] sigmoid probs (no noise)
        adj_hard   = gate.hard_adjacency()    # [K, K] binary
    """

    def __init__(
        self,
        n_features: int,
        temperature: float = 1.0,
        sparsity_prior: float = -1.0,
    ):
        """
        Args:
            n_features:      Number of causal variables K
            temperature:     Gumbel-Sigmoid temperature τ (anneal toward 0 during training)
            sparsity_prior:  Initial bias on logits — negative = sparse prior (fewer edges)
        """
        super().__init__()
        self.n_features = n_features
        self.temperature = temperature

        # Learnable edge logits — initialised with a sparse prior
        # Use large negative float (not -inf) to stay finite through sigmoid/softmax
        logits = torch.ones(n_features, n_features) * sparsity_prior
        logits.fill_diagonal_(-20.0)   # effectively 0 prob, but no NaN
        self.logits = nn.Parameter(logits)

    def forward(self, hard: bool = False) -> torch.Tensor:
        """
        Returns [K, K] gate values.
        hard=False: soft Gumbel-Sigmoid (training)
        hard=True:  hard 0/1 with straight-through gradient
        """
        # Mask diagonal permanently
        mask = torch.ones_like(self.logits)
        mask.fill_diagonal_(0.0)

        if hard or not self.training:
            return (self.logits > 0).float() * mask

        # Gumbel-Sigmoid: add two Gumbel samples (for binary case)
        gumbel1 = -torch.log(-torch.log(torch.rand_like(self.logits).clamp(1e-8)))
        gumbel2 = -torch.log(-torch.log(torch.rand_like(self.logits).clamp(1e-8)))
        noisy_logits = (self.logits + gumbel1 - gumbel2) / self.temperature

        soft = torch.sigmoid(noisy_logits) * mask

        if hard:
            # Straight-through estimator
            hard_gate = (soft > 0.5).float() * mask
            return hard_gate - soft.detach() + soft
        return soft

    def edge_probs(self) -> torch.Tensor:
        """
        Returns [K, K] edge existence probabilities (no noise, sigmoid of logits).
        Diagonal is zeroed (no self-loops).
        """
        probs = torch.sigmoid(self.logits)
        probs = probs.clone()
        probs.fill_diagonal_(0.0)
        return probs

    def hard_adjacency(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Returns [K, K] hard binary adjacency matrix.
        Use at inference time for causal graph output.
        """
        return (self.edge_probs() > threshold).float()

    def sparsity_loss(self, target_density: float = 0.3) -> torch.Tensor:
        """
        Soft sparsity prior: encourage about target_density fraction of edges to be on.
        Penalises the mean edge probability deviating from target_density.
        This is gentler than an L1 on logits.
        """
        probs = self.edge_probs()
        # Only off-diagonal elements
        mask = 1 - torch.eye(self.n_features, device=probs.device)
        mean_prob = (probs * mask).sum() / mask.sum()
        return (mean_prob - target_density).pow(2)

    def anneal_temperature(self, epoch: int, start_temp: float = 2.0, end_temp: float = 0.2, n_epochs: int = 1000):
        """
        Linearly anneal temperature from start_temp → end_temp over n_epochs.
        Call once per epoch during training.
        """
        frac = min(epoch / n_epochs, 1.0)
        self.temperature = start_temp + frac * (end_temp - start_temp)
