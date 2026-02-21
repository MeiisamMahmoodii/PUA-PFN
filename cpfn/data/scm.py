"""
Structural Causal Model (SCM) generation for diverse training data.
Richly randomised prior matching Do-PFN's diversity.
"""

import torch
import torch.nn as nn
import random
from typing import Tuple


class CausalMechanism(nn.Module):
    """
    A neural network representing a single causal mechanism.
    Maps parents -> child variable via randomly-initialised (frozen) weights.
    Uses Kaiming initialisation and a randomly chosen nonlinearity,
    matching the ANM prior from Do-PFN.
    """

    NONLINEARITIES = [
        nn.ReLU(),
        nn.Tanh(),
        # x^2 approximated via a custom module
    ]

    def __init__(self, n_parents: int, nonlinearity: str = "relu"):
        super().__init__()
        self.nonlinearity_name = nonlinearity

        if n_parents == 0:
            self.model = None
        else:
            act = {
                "relu": nn.ReLU(),
                "tanh": nn.Tanh(),
                "square": _Square(),
            }[nonlinearity]

            self.model = nn.Sequential(
                nn.Linear(n_parents, 16),
                act,
                nn.Linear(16, 1),
            )
            # Kaiming init (matches Do-PFN weight prior)
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.zeros_(m.bias)
            # Freeze: mechanism is fixed once created
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        x:     [n_samples, n_parents]
        noise: [n_samples]
        Returns: [n_samples]  clamped to [-20, 20] for training stability
        """
        if self.model is None:
            return noise.clamp(-20, 20)
        out = self.model(x).view(-1) + noise
        return out.clamp(-20, 20)


class _Square(nn.Module):
    """Scaled element-wise square: tanh(x)^2, bounded to (-1, 1)."""
    def forward(self, x):
        return torch.tanh(x) ** 2


def generate_full_multiverse(
    n_samples: int,
    n_features: int,
    do_val: float = 5.0,
    edge_prob: float = 0.3,
    device: str = "cpu",
    randomise_prior: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a multiverse: observational + one interventional universe per variable.

    When randomise_prior=True (training), all SCM parameters are re-sampled
    each call from a rich prior matching Do-PFN's diversity:
      - edge_prob   ~ Uniform(0.1, 0.5)
      - do_val      ~ Uniform(2, 15)
      - nonlinearity ~ choice(relu, tanh, square)
      - noise std   ~ Uniform(0.5, 3.0)
      - noise dist  ~ choice(gaussian, laplace)

    Args:
        n_samples:       Samples per universe
        n_features:      Number of causal variables
        do_val:          Intervention value (overridden if randomise_prior=True)
        edge_prob:       DAG edge probability (overridden if randomise_prior=True)
        device:          Torch device
        randomise_prior: Whether to randomise all SCM hyperparameters

    Returns:
        multiverse: [n_features+1, n_samples, n_features]
                    - Universe 0:   observational
                    - Universes 1…K: do(X_i = do_val)
        adj:        [n_features, n_features] adjacency matrix
    """
    if randomise_prior:
        edge_prob  = random.uniform(0.1, 0.5)
        do_val     = random.uniform(2.0, 10.0)   # reduced ceiling: keep within [-20, 20] borders
        sigma_exo  = random.uniform(0.5, 2.0)    # reduced ceiling to stay in range
        noise_type = random.choice(["gaussian", "laplace"])
        gamma      = random.choice(["relu", "tanh", "square"])
    else:
        sigma_exo  = 1.0
        noise_type = "gaussian"
        gamma      = "relu"

    # --- Random DAG (upper-triangular = topological order) ---
    adj = (torch.rand(n_features, n_features, device=device) < edge_prob).triu(diagonal=1)

    # --- Causal mechanisms (one per variable) ---
    mechanisms = [
        CausalMechanism(int(adj[:, i].sum()), nonlinearity=gamma).to(device)
        for i in range(n_features)
    ]

    # --- Exogenous noise ---
    if noise_type == "gaussian":
        exogenous_noise = torch.randn(n_samples, n_features, device=device) * sigma_exo
    else:  # laplace
        exogenous_noise = (
            torch.distributions.Laplace(0, sigma_exo)
            .sample((n_samples, n_features))
            .to(device)
        )

    # --- Multiverse tensor ---
    multiverse = torch.zeros(n_features + 1, n_samples, n_features, device=device)

    # Universe 0: Observational
    for i in range(n_features):
        parents = torch.where(adj[:, i])[0]
        noise   = exogenous_noise[:, i]
        if len(parents) > 0:
            multiverse[0, :, i] = mechanisms[i](multiverse[0, :, parents], noise)
        else:
            multiverse[0, :, i] = noise

    # Universes 1..K: do(X_{u-1} = do_val)
    for u_idx in range(1, n_features + 1):
        target = u_idx - 1
        for i in range(n_features):
            parents = torch.where(adj[:, i])[0]
            noise   = exogenous_noise[:, i]
            if i == target:
                multiverse[u_idx, :, i] = do_val
            elif len(parents) > 0:
                multiverse[u_idx, :, i] = mechanisms[i](
                    multiverse[u_idx, :, parents], noise
                )
            else:
                multiverse[u_idx, :, i] = noise

    return multiverse, adj
