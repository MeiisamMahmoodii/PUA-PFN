"""
Parallel Universe Embedding Layer with Causal Hints.
Supports both training mode (full multiverse) and inference mode (obs + query token).
"""

import torch
import torch.nn as nn


class ParallelUniverseEmbedding(nn.Module):
    """
    Embeds multiverse data with explicit causal structure information.

    Structure:
    - Value embedding:        Encodes the numerical value
    - Feature embedding:      Which variable (0..K-1)?
    - Universe embedding:     Observational (0) or Interventional (1)?
    - Intervention flag:      Is this variable the intervention target?
    """

    def __init__(self, n_features: int, embed_dim: int = 128):
        super().__init__()
        self.n_features = n_features
        self.embed_dim  = embed_dim

        self.value_encoder    = nn.Linear(1, embed_dim)
        self.feature_embed    = nn.Embedding(n_features, embed_dim)
        self.universe_embed   = nn.Embedding(2, embed_dim)   # {obs=0, int=1}
        self.intervention_flag = nn.Embedding(2, embed_dim)  # {not target=0, target=1}

    def forward(self, m_data: torch.Tensor) -> torch.Tensor:
        """
        Training-time embedding of a full multiverse.

        Args:
            m_data: [n_universes, n_samples, n_features]

        Returns:
            embeddings: [n_universes, n_samples*n_features, embed_dim]
        """
        u, s, f = m_data.shape
        device  = m_data.device

        flat_data = m_data.reshape(-1, 1)

        feat_idx = torch.arange(f, device=device).repeat(u * s)
        univ_idx = torch.cat([
            torch.zeros(s * f, device=device),
            torch.ones((u - 1) * s * f, device=device)
        ]).long()

        # Intervention flag: universe i intervenes on variable i-1
        flags = torch.zeros(u, s, f, device=device)
        for i in range(1, u):
            flags[i, :, i - 1] = 1
        flags = flags.reshape(-1).long()

        combined = (
            self.value_encoder(flat_data)
            + self.feature_embed(feat_idx)
            + self.universe_embed(univ_idx)
            + self.intervention_flag(flags)
        )
        return combined.view(u, s * f, -1)

    def embed_obs(self, obs_data: torch.Tensor) -> torch.Tensor:
        """
        Embed a single observational universe.

        Args:
            obs_data: [n_samples, n_features]

        Returns:
            [1, n_samples*n_features, embed_dim]
        """
        s, f   = obs_data.shape
        device = obs_data.device

        flat_data = obs_data.reshape(-1, 1)
        feat_idx  = torch.arange(f, device=device).repeat(s)
        univ_idx  = torch.zeros(s * f, dtype=torch.long, device=device)
        flags     = torch.zeros(s * f, dtype=torch.long, device=device)

        combined = (
            self.value_encoder(flat_data)
            + self.feature_embed(feat_idx)
            + self.universe_embed(univ_idx)
            + self.intervention_flag(flags)
        )
        return combined.unsqueeze(0)  # [1, s*f, embed_dim]


class InterventionQueryEncoder(nn.Module):
    """
    Encodes an inference-time intervention query into embedding space.

    At inference the model doesn't have real interventional data;
    instead the user specifies:
      - target_var: which variable is intervened on (int, 0..K-1)
      - do_val:     the intervention value (float)

    This module produces a sequence of K tokens (one per variable)
    seeding the decoder, so the cross-universe attention can operate
    exactly as during training.
    """

    def __init__(self, n_features: int, embed_dim: int = 128):
        super().__init__()
        self.n_features = n_features
        # Encodes [one-hot(target) | do_val] → embed_dim
        self.query_proj = nn.Linear(n_features + 1, embed_dim)
        # Per-feature offset so each variable token is distinguishable
        self.feature_embed = nn.Embedding(n_features, embed_dim)

    def forward(
        self,
        target_var: torch.Tensor, # [N] (int)
        do_val: torch.Tensor,     # [N] (float)
        n_samples: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Args:
            target_var: [N] tensor of target variable indices
            do_val:     [N] tensor of intervention values
        Returns:
            query_tokens: [N, n_samples*n_features, embed_dim]
        """
        N = target_var.shape[0]
        
        # Build batch of one-hot + do_val
        # shape: [N, n_features + 1]
        one_hot = torch.zeros(N, self.n_features + 1, device=device)
        one_hot[torch.arange(N), target_var] = 1.0
        one_hot[:, -1] = do_val
        
        # Project: [N, embed_dim]
        query_base = self.query_proj(one_hot)
        
        # Feature embeddings: [n_features, embed_dim]
        feat_idx = torch.arange(self.n_features, device=device)
        f_embeds = self.feature_embed(feat_idx) # [f, embed_dim]
        
        # Combine: query_base [N, 1, embed_dim] + f_embeds [1, f, embed_dim]
        # Result: [N, f, embed_dim]
        feat_tokens = query_base.unsqueeze(1) + f_embeds.unsqueeze(0)
        
        # Expand across samples: [N, s, f, embed_dim]
        tokens = feat_tokens.unsqueeze(1).expand(-1, n_samples, -1, -1)
        tokens = tokens.reshape(N, n_samples * self.n_features, -1)
        
        return tokens # [N, s*f, embed_dim]
