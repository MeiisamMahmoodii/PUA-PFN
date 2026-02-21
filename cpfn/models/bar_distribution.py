"""
Bar-Distribution: discretised density head for distributional prediction.

Implements the same bar-distribution output used by Do-PFN / TabPFN.
Discretises the output range into N_BINS equal-width bins.

IMPORTANT: Borders must be fixed at training start via init_fixed_borders().
Changing borders each batch causes NLL scale to jump → divergence.

Reference: TabPFN v2 / Do-PFN (arXiv 2506.06039)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BarDistribution(nn.Module):
    """
    Discretised density output head.

    Maps transformer embeddings → logits over N_BINS bins,
    trained with NLL loss. Supports mean prediction and sampling.

    FIXED borders: call init_fixed_borders() once before training.
    Per-batch borders (set_borders) are only for evaluation.
    """

    def __init__(self, embed_dim: int, n_bins: int = 64):
        super().__init__()
        self.n_bins = n_bins
        self.logit_head = nn.Linear(embed_dim, n_bins)
        # Will be set via init_fixed_borders() or set_borders()
        self.register_buffer("borders", torch.zeros(n_bins + 1))
        self._borders_set = False

    def init_fixed_borders(
        self,
        y_min: float = -30.0,
        y_max: float = 30.0,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Set globally fixed borders covering all SCM output values.
        Call ONCE before training. Default [-30, 30] covers:
          - do_val up to 15 plus mechanism amplification
          - negative observational values from noise
        """
        self.borders = torch.linspace(y_min, y_max, self.n_bins + 1, device=device)
        self._borders_set = True

    def set_borders(self, y_min: float, y_max: float, device: torch.device):
        """
        Set borders to a specific range — use for evaluation, not training.
        Adds 10% margin so targets aren't clipped to boundary bins.
        """
        margin = (y_max - y_min) * 0.1 + 1e-3
        self.borders = torch.linspace(
            y_min - margin, y_max + margin, self.n_bins + 1, device=device
        )
        self._borders_set = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., embed_dim]  →  logits: [..., n_bins]"""
        return self.logit_head(x)

    def nll_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Soft-target NLL loss.

        Args:
            logits:  [..., n_bins]
            targets: [...] float values within the border range
        """
        bin_idx, weight = self._soft_target(targets)
        log_probs = F.log_softmax(logits, dim=-1)

        left_nll  = -weight * log_probs.gather(
            -1, bin_idx.clamp(0, self.n_bins - 1).unsqueeze(-1)
        ).squeeze(-1)
        right_nll = -(1 - weight) * log_probs.gather(
            -1, (bin_idx + 1).clamp(0, self.n_bins - 1).unsqueeze(-1)
        ).squeeze(-1)
        return (left_nll + right_nll).mean()

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """E[y] = Σ p(bin) * centre(bin)"""
        probs       = F.softmax(logits, dim=-1)
        bin_centres = (self.borders[:-1] + self.borders[1:]) / 2
        return (probs * bin_centres).sum(dim=-1)

    def _soft_target(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        borders   = self.borders.to(y.device)
        y_clamped = y.clamp(borders[0].item(), borders[-1].item())
        bin_idx   = torch.bucketize(y_clamped, borders[1:-1])
        left      = borders[bin_idx.clamp(0, self.n_bins - 1)]
        right     = borders[(bin_idx + 1).clamp(0, self.n_bins)]
        width     = (right - left).clamp(min=1e-8)
        alpha     = 1.0 - (y_clamped - left) / width
        return bin_idx, alpha
