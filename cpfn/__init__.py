"""
Causal Prior Function Network (C-PFN)
A foundation model for causal inference via parallel universes.
"""

__version__ = "0.1.0"

from cpfn.models.embedding import ParallelUniverseEmbedding
from cpfn.models.blocks import CrossUniverseBlock
from cpfn.models.transformer import MultiverseTransformer
from cpfn.data.scm import CausalMechanism, generate_full_multiverse
from cpfn.training.trainer import Trainer
from cpfn.evaluation.evaluator import CausalDiscoveryEvaluator

__all__ = [
    "ParallelUniverseEmbedding",
    "CrossUniverseBlock",
    "MultiverseTransformer",
    "CausalMechanism",
    "generate_full_multiverse",
    "Trainer",
    "CausalDiscoveryEvaluator",
]
