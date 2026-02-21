"""
Multiverse Transformer: The core PUA-PFN architecture.

Training:  full parallel universes → learn causal intervention structure
Inference: observational data + (target_var, do_val) query → p(y | do(X=v))
"""

import torch
import torch.nn as nn
from cpfn.models.embedding import ParallelUniverseEmbedding, InterventionQueryEncoder
from cpfn.models.blocks import CrossUniverseBlock
from cpfn.models.bar_distribution import BarDistribution
from cpfn.models.causal_gate import CausalGate


class MultiverseTransformer(nn.Module):
    """
    Foundation model for causal inference via parallel universe reasoning.

    Architecture:
    1. Embedding:             Encodes multiverse data with causal hints
    2. Observational encoder: Deep self-attention on Universe 0 (the base world)
    3. Interventional decoder: Cross-universe attention blocks (parallel worlds → obs)
    4. Bar-distribution head: Full p(y | do(X=v)) distribution output

    Two forward modes:
    ─────────────────
    forward(m_data)             — training: full multiverse input
    infer(obs_data, var, val)   — inference: obs data + intervention query
    """

    def __init__(
        self,
        n_features: int = 5,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_decoder_layers: int = 4,
        n_bins: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_bins     = n_bins

        # --- Embedding ---
        self.embedding       = ParallelUniverseEmbedding(n_features, embed_dim)
        self.query_encoder   = InterventionQueryEncoder(n_features, embed_dim)

        # --- Encoder: observational universe ---
        self.obs_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                batch_first=True,
                dropout=dropout,
            ),
            num_layers=3,
        )

        # --- Decoder: cross-universe attention blocks ---
        self.int_decoder = nn.ModuleList([
            CrossUniverseBlock(embed_dim, n_heads)
            for _ in range(n_decoder_layers)
        ])

        # --- Output: bar-distribution ---
        self.bar_head = BarDistribution(embed_dim, n_bins=n_bins)

        # --- Causal Gate: learned K×K edge mask (Gumbel-Sigmoid) ---
        self.causal_gate = CausalGate(
            n_features=n_features,
            temperature=2.0,       # will be annealed during training
            sparsity_prior=-1.0,   # initialise biased toward sparse graph
        )

    # ------------------------------------------------------------------ #
    #  Training forward pass — full multiverse                            #
    # ------------------------------------------------------------------ #
    def forward(self, m_data: torch.Tensor) -> torch.Tensor:
        """
        Training mode: all parallel universes are available.

        Args:
            m_data: [n_universes, n_samples, n_features]
                    universe 0  = observational
                    universes 1…K = do(X_{i-1} = v)

        Returns:
            logits: [n_features, n_samples*n_features, n_bins]
                    Bar-distribution logits for each interventional universe.
        """
        u, s, f = m_data.shape

        # Embed all universes
        embedded = self.embedding(m_data)           # [u, s*f, embed_dim]

        # Encode observational universe
        obs_ctx = self.obs_encoder(embedded[0].unsqueeze(0))  # [1, s*f, embed_dim]

        # Interventional universes
        u_int = embedded[1:]                        # [K, s*f, embed_dim]

        # Replicate obs context for each interventional universe
        obs_repeated = obs_ctx.expand(u - 1, -1, -1)  # [K, s*f, embed_dim]

        # Cross-universe decoder
        for layer in self.int_decoder:
            u_int = layer(obs_repeated, u_int)      # [K, s*f, embed_dim]

        # Bar-distribution logits: [K, s*f, n_bins]
        logits = self.bar_head(u_int)

        # Apply Gumbel-Sigmoid causal gate to predicted means
        # gate: [K, K] — gate[i, j] = prob that X_i → X_j exists
        # We gate the mean prediction for each (intervention i, variable j) pair
        gate = self.causal_gate(hard=False)         # [K, K] soft mask
        means = self.bar_head.mean(logits)          # [K, s*f]
        means_2d = means.view(f, s, f)              # [K, s, f] = [interv, samples, var]
        # gate[i, j]: intervening on i → variable j is causal
        # broadcast gate over the samples dimension
        gated_means = means_2d * gate.unsqueeze(1)  # [K, s, f]

        return logits, gated_means, gate

    # ------------------------------------------------------------------ #
    #  Inference forward pass — obs data + query token                    #
    # ------------------------------------------------------------------ #
    def infer(
        self,
        obs_data: torch.Tensor,
        target_var: int,
        do_val: float,
    ) -> torch.Tensor:
        """
        Inference mode: no interventional data needed.

        Args:
            obs_data:   [n_samples, n_features]  observational data
            target_var: int  which variable is intervened on (0-indexed)
            do_val:     float  the do() value

        Returns:
            logits: [n_samples*n_features, n_bins]
                    Bar-distribution logits for p(y | do(X_{target}=do_val), obs)
            mean_pred: [n_samples, n_features]
                    E[y] reshaped back to sample×feature grid
        """
        s, f   = obs_data.shape
        device = obs_data.device

        # Encode observational data
        obs_emb = self.embedding.embed_obs(obs_data)        # [1, s*f, embed_dim]
        obs_ctx = self.obs_encoder(obs_emb)                 # [1, s*f, embed_dim]

        # Build intervention query tokens (no real int data needed)
        q_tokens = self.query_encoder(target_var, do_val, s, device)  # [1, s*f, embed_dim]

        # Cross-universe decoder: query tokens attend to observational context
        u_int = q_tokens
        for layer in self.int_decoder:
            u_int = layer(obs_ctx, u_int)               # [1, s*f, embed_dim]

        logits    = self.bar_head(u_int.squeeze(0))     # [s*f, n_bins]
        mean_pred = self.bar_head.mean(logits).view(s, f)  # [s, f]

        return logits, mean_pred
