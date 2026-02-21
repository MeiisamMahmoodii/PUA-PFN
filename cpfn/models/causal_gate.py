"""
Data-Conditioned Causal Gate for PUA-PFN.

Unlike a fixed K×K logit matrix, this gate is conditioned on the
encoder's observational representation — so it can predict different
adjacency matrices for different SCMs.

Architecture:
  obs_context [1, s*f, embed_dim]
      → pool → [embed_dim]
      → MLP  → [K*K] logits
      → reshape → [K, K] edge probabilities

Training: Gumbel-Sigmoid (differentiable exploration)
Inference: hard threshold at 0.5
Supervision: BCE against ground-truth adjacency (available at training since
             we generate synthetic SCMs with known structure)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalGate(nn.Module):
    """
    Data-conditioned causal gate using Gumbel-Sigmoid.

    Takes the observational context embedding (from the encoder) and
    predicts a K×K edge probability matrix for the current SCM.

    This solves the key limitation of a fixed gate:
    - Fixed gate converges to marginal edge probability (0.3) for all edges
    - Conditioned gate can distinguish WHICH edges exist in this specific graph

    At training: gate produces soft Gumbel-Sigmoid mask
    At inference: gate produces hard 0/1 adjacency from encoder output
    """

    def __init__(
        self,
        n_features: int,
        embed_dim: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.temperature = temperature

        # MLP: obs_context_pooled → K×K edge logits
        self.edge_predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features * n_features),
        )

        # Diagonal mask buffer (no self-loops, always)
        diag_mask = 1.0 - torch.eye(n_features)
        self.register_buffer("diag_mask", diag_mask)

    def forward(
        self,
        obs_context: torch.Tensor,
        hard: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            obs_context: [1, s*f, embed_dim]  encoder output for obs universe
            hard:        if True, return hard 0/1 gate (straight-through)

        Returns:
            gate: [K, K]  soft (0,1) or hard {0,1} edge mask
        """
        # Pool over tokens: mean across sequence dimension → [embed_dim]
        ctx = obs_context.mean(dim=1).squeeze(0)   # [embed_dim]

        # Predict edge logits
        logits = self.edge_predictor(ctx).view(self.n_features, self.n_features)
        logits = logits * self.diag_mask + (1 - self.diag_mask) * (-20.0)

        if hard or not self.training:
            probs = torch.sigmoid(logits) * self.diag_mask
            return (probs > 0.5).float()

        # Gumbel-Sigmoid: differentiable binary sampling
        g1 = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-8)))
        g2 = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-8)))
        noisy = (logits + g1 - g2) / self.temperature
        soft  = torch.sigmoid(noisy) * self.diag_mask

        if hard:
            hard_gate = (soft > 0.5).float() * self.diag_mask
            return hard_gate - soft.detach() + soft  # straight-through
        return soft

    def edge_probs(self, obs_context: torch.Tensor) -> torch.Tensor:
        """
        [K, K] edge probabilities (no noise) given obs_context.
        Use for evaluation and monitoring.
        """
        ctx    = obs_context.mean(dim=1).squeeze(0)
        logits = self.edge_predictor(ctx).view(self.n_features, self.n_features)
        probs  = torch.sigmoid(logits) * self.diag_mask
        return probs.clamp(0.0, 1.0)

    def hard_adjacency(self, obs_context: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        [K, K] binary adjacency matrix given obs_context.
        Use at inference time.
        """
        return (self.edge_probs(obs_context) > threshold).float()

    def bce_loss(
        self,
        obs_context: torch.Tensor,
        true_adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary cross-entropy loss against ground-truth adjacency.
        This directly supervises the gate to predict the correct graph.

        Args:
            obs_context: [1, s*f, embed_dim]
            true_adj:    [K, K] float (0/1)
        """
        ctx    = obs_context.mean(dim=1).squeeze(0)
        logits = self.edge_predictor(ctx).view(self.n_features, self.n_features)
        # Only compute loss on off-diagonal entries
        off_diag  = self.diag_mask.bool()
        return F.binary_cross_entropy_with_logits(
            logits[off_diag],
            true_adj[off_diag].float(),
        )

    def sparsity_loss(self, obs_context: torch.Tensor, target_density: float = 0.3) -> torch.Tensor:
        """Encourage ~target_density fraction of edges to be predicted ON."""
        probs     = self.edge_probs(obs_context)
        mean_prob = probs.sum() / self.diag_mask.sum()
        return (mean_prob - target_density).pow(2)

    def anneal_temperature(
        self, epoch: int,
        start_temp: float = 2.0,
        end_temp: float = 0.3,
        n_epochs: int = 1000,
    ):
        frac = min(epoch / n_epochs, 1.0)
        self.temperature = start_temp + frac * (end_temp - start_temp)
