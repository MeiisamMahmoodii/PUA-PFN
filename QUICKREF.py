#!/usr/bin/env python3
"""
Quick Reference: C-PFN Command Cheatsheet
"""

import sys

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    C-PFN QUICK REFERENCE                                  ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─ SETUP ─────────────────────────────────────────────────────────────────┐
│ cd /home/meisam/code/PUA-PFN                                             │
│ source .venv/bin/activate                                                │
│ python scripts/test.py              # Verify installation              │
└─────────────────────────────────────────────────────────────────────────┘

┌─ TRAINING ──────────────────────────────────────────────────────────────┐
│ # Quick test (100 epochs, ~15 min)                                      │
│ python scripts/train.py --num-epochs 100                                │
│                                                                          │
│ # Full training (1500 epochs, ~2-4 hours)                               │
│ python scripts/train.py --num-epochs 1500 --n-samples 30 --device cuda  │
│                                                                          │
│ # Resume training from checkpoint                                       │
│ python scripts/train.py --resume checkpoints/checkpoint_epoch_1000.pt   │
│                         --num-epochs 2000                               │
│                                                                          │
│ # Custom configuration                                                  │
│ python scripts/train.py --config my_config.json --num-epochs 1500       │
└─────────────────────────────────────────────────────────────────────────┘

┌─ EVALUATION ────────────────────────────────────────────────────────────┐
│ # Single evaluation (verbose)                                           │
│ python scripts/evaluate.py --checkpoint checkpoints/checkpoint_epoch_1500.pt
│                            --config config.json --num-evals 1           │
│                                                                          │
│ # Multiple evaluations (for averaging)                                  │
│ python scripts/evaluate.py --checkpoint checkpoints/checkpoint_epoch_1500.pt
│                            --num-evals 10                               │
│                                                                          │
│ # Evaluate multiple checkpoints                                         │
│ for epoch in 500 1000 1500; do                                          │
│   python scripts/evaluate.py \                                          │
│     --checkpoint checkpoints/checkpoint_epoch_$epoch.pt \               │
│     --num-evals 5                                                       │
│ done                                                                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─ MONITORING ────────────────────────────────────────────────────────────┐
│ # Open TensorBoard (in new terminal)                                    │
│ tensorboard --logdir logs                                               │
│ # Then open http://localhost:6006 in browser                           │
│                                                                          │
│ # Check training history                                                │
│ python -c "import json; print(json.dumps(json.load(open('logs/history.json')),
│                              indent=2))"                               │
└─────────────────────────────────────────────────────────────────────────┘

┌─ CONFIGURATION ─────────────────────────────────────────────────────────┐
│ # View config                                                            │
│ cat config.json                                                          │
│                                                                          │
│ # Modify config programmatically                                        │
│ python -c "from cpfn.utils import Config; c = Config.load('config.json')
│ c.num_epochs = 5000; c.save('config.json')"                             │
└─────────────────────────────────────────────────────────────────────────┘

┌─ COMMON PARAMETER COMBINATIONS ─────────────────────────────────────────┐
│ QUICK TEST (verify setup):                                              │
│   python scripts/train.py --num-epochs 100 --log-interval 10            │
│                                                                          │
│ PRODUCTION TRAINING (default):                                          │
│   python scripts/train.py --num-epochs 1500 --device cuda               │
│                                                                          │
│ RESEARCH TRAINING (extended):                                           │
│   python scripts/train.py --num-epochs 5000 --n-samples 50 --device cuda
│                                                                          │
│ LOW MEMORY (CPU):                                                       │
│   python scripts/train.py --embed-dim 64 --n-samples 20 --device cpu    │
│                                                                          │
│ FAST ITERATION (testing):                                               │
│   python scripts/train.py --num-epochs 500 --log-interval 50 --device cuda
└─────────────────────────────────────────────────────────────────────────┘

┌─ FILES & DIRECTORIES ───────────────────────────────────────────────────┐
│ config.json                    Configuration (saved automatically)       │
│ logs/                          TensorBoard logs                          │
│ logs/history.json              Training history (JSON)                   │
│ checkpoints/                   Model checkpoints                         │
│ cpfn/                          Main package source code                  │
│ scripts/                       CLI entry points                          │
│ README_CPFN.md                 Full architecture guide                   │
│ TRAINING_GUIDE.md              Step-by-step training guide               │
└─────────────────────────────────────────────────────────────────────────┘

┌─ TROUBLESHOOTING ───────────────────────────────────────────────────────┐
│ PROBLEM: Module not found                                               │
│ FIX:     source .venv/bin/activate                                      │
│                                                                          │
│ PROBLEM: Out of GPU memory                                              │
│ FIX:     --embed-dim 64 or --n-samples 20                               │
│                                                                          │
│ PROBLEM: Training on CPU (too slow)                                     │
│ FIX:     --device cuda                                                  │
│                                                                          │
│ PROBLEM: Loss not decreasing                                            │
│ FIX:     --learning-rate 5e-5 (smaller LR)                              │
│                                                                          │
│ PROBLEM: Loss is NaN                                                    │
│ FIX:     --learning-rate 1e-5 (much smaller LR)                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─ PYTHON API USAGE ──────────────────────────────────────────────────────┐
│ from cpfn.models import MultiverseTransformer                           │
│ from cpfn.data import generate_full_multiverse                          │
│ from cpfn.training import Trainer                                       │
│ from cpfn.evaluation import CausalDiscoveryEvaluator                    │
│ from cpfn.utils import Config, get_device                               │
│                                                                          │
│ # Create model                                                          │
│ model = MultiverseTransformer(n_features=5, embed_dim=128)              │
│                                                                          │
│ # Generate data                                                         │
│ m_data, adj = generate_full_multiverse(n_samples=30, n_features=5)      │
│                                                                          │
│ # Create trainer                                                        │
│ trainer = Trainer(model, n_features=5, device='cuda')                   │
│ history = trainer.train(num_epochs=1500)                                │
│                                                                          │
│ # Evaluate                                                              │
│ evaluator = CausalDiscoveryEvaluator(model, device='cuda')              │
│ metrics = evaluator.evaluate(n_samples=30, n_features=5)                │
└─────────────────────────────────────────────────────────────────────────┘

┌─ EXPECTED RESULTS ──────────────────────────────────────────────────────┐
│ LOSS:          10 → 1 → 0.1 → 0.01 → 0.001 (over 1500 epochs)           │
│ F1 SCORE:      0.0 → 0.3 → 0.5 → 0.7 → 0.8+ (over 1500 epochs)         │
│ TRAINING TIME: ~2-4 hours on GPU, ~24+ hours on CPU                     │
│ CHECKPOINTS:   Saved every 500 epochs (resume-able)                    │
│ METRICS:       TensorBoard at http://localhost:6006                     │
└─────────────────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════════════════════╗
║ For detailed guides, see:                                                  ║
║   - README_CPFN.md (Architecture & overview)                              ║
║   - TRAINING_GUIDE.md (Step-by-step training)                             ║
║   - MIGRATION_SUMMARY.md (Project structure)                              ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
