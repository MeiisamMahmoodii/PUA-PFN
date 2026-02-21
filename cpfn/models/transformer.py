"""
Multiverse Transformer: The core C-PFN architecture
"""

import torch
import torch.nn as nn
from cpfn.models.embedding import ParallelUniverseEmbedding
from cpfn.models.blocks import CrossUniverseBlock


class MultiverseTransformer(nn.Module):
    """
    Foundation model for causal inference via parallel universe reasoning.

    Architecture:
    1. Embedding layer: Encodes multiverse data with causal hints
    2. Observational encoder: Deep transformer on Universe 0
    3. Interventional decoder: Cross-universe attention blocks for Universes 1-D
    4. Output head: Maps to intervention effects
    """

    def __init__(
        self,
        n_features: int = 5,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_decoder_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.embedding = ParallelUniverseEmbedding(n_features, embed_dim)

        # Encoder for observational universe (the "base case")
        self.obs_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads, batch_first=True
            ),
            num_layers=3,
        )

        # Decoder: Multiple cross-universe blocks
        self.int_decoder = nn.ModuleList(
            [CrossUniverseBlock(embed_dim, n_heads) for _ in range(n_decoder_layers)]
        )

        self.output_head = nn.Linear(embed_dim, 1)

    def forward(self, m_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m_data: [n_universes, n_samples, n_features]

        Returns:
            predictions: [n_features, n_samples*n_features, 1]
                         Predicted values for each interventional universe
        """
        u, s, f = m_data.shape

        # Embed all universes
        embedded = self.embedding(m_data)

        # Encode observational universe
        u_obs = self.obs_encoder(embedded[0].unsqueeze(0))

        # Get interventional universes
        u_int = embedded[1:]

        # Replicate observational context for each interventional universe
        obs_context = u_obs.repeat(u - 1, 1, 1)

        # Apply cross-universe blocks
        for layer in self.int_decoder:
            u_int = layer(obs_context, u_int)

        # Output
        return self.output_head(u_int)
