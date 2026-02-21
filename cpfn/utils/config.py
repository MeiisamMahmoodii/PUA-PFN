"""
Configuration management for C-PFN experiments
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Config:
    """Configuration for C-PFN training and evaluation."""

    # Model architecture
    n_features: int = 5
    embed_dim: int = 128
    n_heads: int = 8
    n_decoder_layers: int = 4

    # Training
    num_epochs: int = 1500
    n_samples: int = 30
    learning_rate: float = 1e-4
    batch_size: int = 1  # Meta-learning: one SCM per step

    # Data generation
    edge_prob: float = 0.3
    do_val_range: tuple = (2.0, 15.0)

    # Evaluation
    eval_n_samples: int = 30
    eval_do_val: float = 10.0

    # Logging
    log_interval: int = 100
    checkpoint_interval: int = 500
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"

    # Device
    device: str = "cuda"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: str):
        """Save config to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load config from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
