"""
Parallel Universe Embedding Layer with Causal Hints
"""

import torch
import torch.nn as nn


class ParallelUniverseEmbedding(nn.Module):
    """
    Embeds multiverse data with explicit causal structure information.

    Structure:
    - Value embedding: Encodes the numerical value
    - Feature embedding: Which variable (0-4)?
    - Universe embedding: Observational (0) or Interventional (1)?
    - Intervention flag embedding: Is this variable the intervention target?
    """

    def __init__(self, n_features: int, embed_dim: int = 128):
        super().__init__()
        self.value_encoder = nn.Linear(1, embed_dim)
        self.feature_embed = nn.Embedding(n_features, embed_dim)
        self.universe_embed = nn.Embedding(2, embed_dim)  # {obs, int}
        self.intervention_flag = nn.Embedding(2, embed_dim)  # {not intervened, intervened}

    def forward(self, m_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m_data: [n_universes, n_samples, n_features]

        Returns:
            embeddings: [n_universes, n_samples*n_features, embed_dim]
        """
        u, s, f = m_data.shape
        device = m_data.device

        # Flatten to process all tokens
        flat_data = m_data.reshape(-1, 1)

        # Create index arrays
        feat_idx = torch.arange(f, device=device).repeat(u * s)
        univ_idx = torch.cat(
            [torch.zeros(s * f, device=device), torch.ones((u - 1) * s * f, device=device)]
        ).long()

        # Intervention flag: In universe i, variable i-1 is the intervention target
        flags = torch.zeros(u, s, f, device=device)
        for i in range(1, u):
            flags[i, :, i - 1] = 1
        flags = flags.reshape(-1).long()

        # Combine embeddings additively
        val_emb = self.value_encoder(flat_data)
        feat_emb = self.feature_embed(feat_idx)
        univ_emb = self.universe_embed(univ_idx)
        flag_emb = self.intervention_flag(flags)

        combined = val_emb + feat_emb + univ_emb + flag_emb
        return combined.view(u, s * f, -1)
