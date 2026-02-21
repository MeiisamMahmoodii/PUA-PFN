"""
Cross-Universe Attention Blocks
"""

import torch.nn as nn


class CrossUniverseBlock(nn.Module):
    """
    Attention block that compares interventional universe against observational context.

    Structure:
    1. Intra-attention: Self-attention within the interventional universe
    2. Cross-attention: Query interventional universe against observational universe
    3. Feed-forward: Standard transformer FFN
    """

    def __init__(self, embed_dim: int, n_heads: int = 4):
        super().__init__()
        self.intra_attention = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, u_obs, u_int):
        """
        Args:
            u_obs: [batch_size, seq_len, embed_dim] observational universe context
            u_int: [batch_size, seq_len, embed_dim] interventional universe to transform

        Returns:
            transformed u_int
        """
        # Intra-attention: self-attention within interventional universe
        attn_intra, _ = self.intra_attention(u_int, u_int, u_int)
        u_int = self.norm1(u_int + attn_intra)

        # Cross-attention: interventional queries observational
        attn_cross, _ = self.cross_attention(u_int, u_obs, u_obs)
        u_int = self.norm2(u_int + attn_cross)

        # Feed-forward
        return u_int + self.ffn(u_int)
