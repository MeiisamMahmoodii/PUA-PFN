#!/usr/bin/env python3
"""
Quick test script to verify C-PFN installation and basic functionality
"""

import torch
from cpfn.models import MultiverseTransformer
from cpfn.data import generate_full_multiverse
from cpfn.evaluation import CausalDiscoveryEvaluator

print("=" * 70)
print("C-PFN Quick Test")
print("=" * 70)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n✓ Device: {device}")

# Create model
print("\n✓ Creating MultiverseTransformer...")
model = MultiverseTransformer(n_features=5, embed_dim=128)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Generate data
print("\n✓ Generating multiverse...")
m_data, adj = generate_full_multiverse(n_samples=10, n_features=5, device=device)
print(f"  Multiverse shape: {m_data.shape}")
print(f"  Adjacency shape: {adj.shape}")
print(f"  Causal edges: {torch.nonzero(adj).tolist()}")

# Forward pass
print("\n✓ Running forward pass...")
model.to(device)
m_data = m_data.to(device)
with torch.no_grad():
    output = model(m_data)
print(f"  Output shape: {output.shape}")

# Quick evaluation
print("\n✓ Running blind inference evaluation...")
evaluator = CausalDiscoveryEvaluator(model, device=device)
metrics = evaluator.evaluate(n_samples=10, n_features=5, do_val=10.0, verbose=False)
print(f"  F1 Score: {metrics['f1']:.4f}")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall: {metrics['recall']:.4f}")

print("\n" + "=" * 70)
print("✓ All tests passed!")
print("=" * 70)
print("\nNext steps:")
print("  1. Train model: python scripts/train.py --num-epochs 1500")
print("  2. Evaluate: python scripts/evaluate.py --checkpoint <path> --num-evals 5")
