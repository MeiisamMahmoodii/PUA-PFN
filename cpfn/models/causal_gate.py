"""
Data-Conditioned Causal Gate for PUA-PFN.

Unlike a fixed K×K logit matrix, this gate is conditioned on the
encoder's observational representation — so it can predict different
adjacency matrices for different SCMs.

Architecture:
  obs_context [1, s*f, embed_dim]
      → pool (global mean or per-variable mean)
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


def _pool_obs_context(
    obs_context: torch.Tensor,
    n_features: int,
    use_per_variable: bool,
) -> torch.Tensor:
    """
    Pool obs_context [1, s*f, embed_dim] to a fixed-size vector.
    - use_per_variable=False: global mean → [embed_dim]
    - use_per_variable=True:  per-variable mean → [n_features, embed_dim] flattened
    """
    if not use_per_variable:
        return obs_context.mean(dim=1).flatten()
    # Reshape to [1, n_samples, n_features, embed_dim], mean over samples
    s_f, edim = obs_context.shape[1], obs_context.shape[2]
    s = s_f // n_features
    ctx = obs_context.view(1, s, n_features, edim).mean(dim=1).flatten()
    return ctx


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
        hidden_dim: int = 128,
        temperature: float = 1.0,
        use_per_variable_pooling: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.temperature = temperature
        self.use_per_variable_pooling = use_per_variable_pooling

        # Input dim: per-variable keeps structure (K*embed_dim), global collapses to embed_dim
        ctx_dim = n_features * embed_dim if use_per_variable_pooling else embed_dim

        # MLP: obs_context_pooled → K×K edge logits
        self.edge_predictor = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, n_features * n_features),
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
        ctx = _pool_obs_context(
            obs_context, self.n_features, self.use_per_variable_pooling
        )

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
        ctx = _pool_obs_context(
            obs_context, self.n_features, self.use_per_variable_pooling
        )
        logits = self.edge_predictor(ctx).view(self.n_features, self.n_features)

        probs  = torch.sigmoid(logits) * self.diag_mask
        return probs.clamp(0.0, 1.0)

    def hard_adjacency(self, obs_context: torch.Tensor, threshold: float = 0.35) -> torch.Tensor:
        """
        [K, K] binary adjacency matrix given obs_context.
        Improved threshold 0.35: reduces False Positives by being more conservative
        than the previous 0.15 baseline.
        """
        return (self.edge_probs(obs_context) > threshold).float()

    def bce_loss(
        self,
        obs_context: torch.Tensor,
        true_adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary cross-entropy loss against ground-truth adjacency.
        Restricts to upper triangle (i<j) to match SCM convention and focus gradients.
        """
        ctx = _pool_obs_context(
            obs_context, self.n_features, self.use_per_variable_pooling
        )
        logits = self.edge_predictor(ctx).view(self.n_features, self.n_features)

        # Upper triangle only (i<j): matches SCM convention, fewer outputs
        upper_tri = torch.triu(torch.ones_like(logits), diagonal=1).bool()
        return F.binary_cross_entropy_with_logits(
            logits[upper_tri],
            true_adj[upper_tri].float(),
        )

    def sparsity_loss(self, obs_context: torch.Tensor, target_density: float = 0.1) -> torch.Tensor:
        """Encourage ~target_density fraction of edges to be predicted ON."""
        probs     = self.edge_probs(obs_context)
        mean_prob = probs.sum() / self.diag_mask.sum()
        return (mean_prob - target_density).pow(2)

    def entropy_loss(self, obs_context: torch.Tensor) -> torch.Tensor:
        """Force probabilities towards 0 or 1. Penalizes uncertainty."""
        probs = self.edge_probs(obs_context)
        # Entropy H = -p*log(p) - (1-p)*log(1-p)
        # Simplified as p*(1-p) which peaks at 0.5
        entropy = probs * (1 - probs)
        return entropy.mean()

    def anneal_temperature(
        self, epoch: int,
        start_temp: float = 2.0,
        end_temp: float = 0.1,  # Lowered from 0.3 for sharper binary decisions
        n_epochs: int = 500,
    ):
        frac = min(epoch / n_epochs, 1.0)
        self.temperature = start_temp + frac * (end_temp - start_temp)
