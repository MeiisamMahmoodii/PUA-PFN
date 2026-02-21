"""
Structural Causal Model (SCM) generation for diverse training data.
"""

import torch
import torch.nn as nn
from typing import Tuple


class CausalMechanism(nn.Module):
    """
    A neural network representing a single causal mechanism.
    Maps parents -> child variable via learned (frozen) weights.
    """

    def __init__(self, n_parents: int):
        super().__init__()
        if n_parents == 0:
            self.model = None
        else:
            self.model = nn.Sequential(
                nn.Linear(n_parents, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
            # Freeze parameters: model structure is fixed across all SCMs
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        x: [n_samples, n_parents] parent values
        noise: [n_samples] exogenous noise
        Returns: [n_samples] child variable values
        """
        if self.model is None:
            return noise
        return self.model(x).view(-1) + noise


def generate_full_multiverse(
    n_samples: int,
    n_features: int,
    do_val: float = 5.0,
    edge_prob: float = 0.3,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a multiverse: observational + interventional universes under random SCM.

    Args:
        n_samples: Number of samples per universe
        n_features: Number of variables
        do_val: Value to set intervened variables to
        edge_prob: Probability of edge in random DAG
        device: Device to generate tensors on

    Returns:
        multiverse: [n_features+1, n_samples, n_features]
                   - Universe 0: observational data
                   - Universes 1-n_features: interventional on each variable
        adj: [n_features, n_features] adjacency matrix of the causal graph
    """
    # Generate random acyclic adjacency matrix
    adj = (torch.rand(n_features, n_features, device=device) < edge_prob).triu(
        diagonal=1
    )

    # Create mechanism for each variable
    mechanisms = [
        CausalMechanism(int(adj[:, i].sum())).to(device) for i in range(n_features)
    ]

    # Generate exogenous noise
    exogenous_noise = torch.randn(n_samples, n_features, device=device)

    # Initialize multiverse tensor
    multiverse = torch.zeros(n_features + 1, n_samples, n_features, device=device)

    # Universe 0: Observational (natural distribution)
    for i in range(n_features):
        parents = torch.where(adj[:, i])[0]
        noise = exogenous_noise[:, i]
        if len(parents) > 0:
            p_data = multiverse[0, :, parents]
            multiverse[0, :, i] = mechanisms[i](p_data, noise)
        else:
            multiverse[0, :, i] = noise

    # Universes 1-D: Interventional (intervene on each variable)
    for u_idx in range(1, n_features + 1):
        target_node = u_idx - 1
        for i in range(n_features):
            parents = torch.where(adj[:, i])[0]
            noise = exogenous_noise[:, i]

            if i == target_node:
                # Set intervened variable to do_val
                multiverse[u_idx, :, i] = do_val
            else:
                # Propagate causal effects
                if len(parents) > 0:
                    p_data = multiverse[u_idx, :, parents]
                    multiverse[u_idx, :, i] = mechanisms[i](p_data, noise)
                else:
                    multiverse[u_idx, :, i] = noise

    return multiverse, adj
