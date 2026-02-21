# C-PFN Training Guide

## Overview

This guide walks you through training a Causal Prior Function Network (C-PFN) from scratch.

## Step 1: Quick Verification (5 minutes)

First, verify that everything is installed correctly:

```bash
cd /home/meisam/code/PUA-PFN
source .venv/bin/activate
python scripts/test.py
```

Expected output:
```
======================================================================
C-PFN Quick Test
======================================================================

✓ Device: cuda
✓ Creating MultiverseTransformer...
  Parameters: 2,837,889
✓ Generating multiverse...
  Multiverse shape: torch.Size([6, 10, 5])
  ...
✓ All tests passed!
```

## Step 2: Start Training (varies by epochs)

### Quick Training (testing, ~30 minutes on GPU)

```bash
python scripts/train.py \
  --num-epochs 100 \
  --n-features 5 \
  --n-samples 30 \
  --log-interval 10
```

### Full Training (production, ~2-4 hours on GPU)

```bash
python scripts/train.py \
  --num-epochs 1500 \
  --n-features 5 \
  --n-samples 30 \
  --device cuda
```

### Extended Training (research, ~8+ hours on GPU)

```bash
python scripts/train.py \
  --num-epochs 5000 \
  --n-features 5 \
  --n-samples 50 \
  --learning-rate 1e-4 \
  --device cuda
```

## Step 3: Monitor Training

While training runs, you can monitor progress in another terminal:

```bash
tensorboard --logdir logs
```

Open http://localhost:6006 in your browser.

Metrics to watch:
- **train/loss**: Should decrease steadily
- **train/lr**: Learning rate schedule
- Checkpoints saved to `checkpoints/` directory

## Step 4: Evaluate the Model

After training completes (or at any checkpoint):

```bash
# Single evaluation (verbose)
python scripts/evaluate.py \
  --checkpoint checkpoints/checkpoint_epoch_1500.pt \
  --config config.json \
  --n-samples 30 \
  --num-evals 1

# Multiple evaluations (for averaging metrics)
python scripts/evaluate.py \
  --checkpoint checkpoints/checkpoint_epoch_1500.pt \
  --config config.json \
  --num-evals 10
```

Expected output (from untrained model):
```
==============================================================================
BLIND CAUSAL INFERENCE EVALUATION (n_samples=30)
==============================================================================
True DAG Edges: [(0, 2), (0, 4), (1, 3), (1, 4)]

--- Intervention on X0 ---
Variable | True Δ | Pred Δ | True Effect?
--------------------------------------------------
X0       |  2.2543 |  0.0001 | YES
X1       |  0.0000 |  0.0001 | NO
X2       |  3.1420 |  0.0002 | YES
...

==============================================================================
CAUSAL DISCOVERY METRICS
==============================================================================
Precision: 0.5000
Recall:    0.5000
F1 Score:  0.5000
==============================================================================
```

## Step 5: Resume Training

If training is interrupted, resume from the last checkpoint:

```bash
python scripts/train.py \
  --resume checkpoints/checkpoint_epoch_1000.pt \
  --num-epochs 1500
```

## Training Configuration

All parameters are saved to `config.json`:

```json
{
  "n_features": 5,
  "embed_dim": 128,
  "n_heads": 8,
  "n_decoder_layers": 4,
  "num_epochs": 1500,
  "n_samples": 30,
  "learning_rate": 0.0001,
  ...
}
```

To use a custom config:

```bash
python scripts/train.py --config my_config.json
```

## Expected Training Progress

### Loss Trajectory
- Epoch 0: ~10-15 (random predictions)
- Epoch 100: ~1-2 (starting to learn)
- Epoch 500: ~0.01-0.1 (good progress)
- Epoch 1000: ~0.001-0.01 (convergence)
- Epoch 1500: ~0.001 (stable)

### Evaluation Metrics (F1 Score)
- Untrained model: 0.0-0.2
- 500 epochs: 0.3-0.5
- 1000 epochs: 0.5-0.7
- 1500 epochs: 0.7-0.9 (expected)

## Performance Tips

### Speed up training
```bash
# Use more samples per epoch (more stable gradients)
python scripts/train.py --n-samples 50

# Increase checkpoint interval if storage is limited
python scripts/train.py --checkpoint-interval 1000

# Reduce logging overhead
python scripts/train.py --log-interval 500
```

### Better convergence
```bash
# Smaller learning rate for fine-tuning
python scripts/train.py --learning-rate 5e-5

# More training epochs
python scripts/train.py --num-epochs 5000
```

### GPU memory optimization
```bash
# If you run out of memory, this shouldn't happen with current setup
# But you can reduce embedding dimension
python scripts/train.py --embed-dim 64
```

## Output Files

After training, you'll have:

```
logs/
├── events.out.tfevents.*  # TensorBoard logs
└── history.json           # Training history (JSON)

checkpoints/
├── checkpoint_epoch_500.pt
├── checkpoint_epoch_1000.pt
├── checkpoint_epoch_1500.pt
└── ...

config.json               # Saved configuration
```

## Interpreting Results

### Good Signs
- Loss decreases monotonically
- Evaluation F1 increases with epochs
- No NaN or Inf values
- Convergence around epoch 1000-1500

### Troubleshooting

**Problem**: Loss doesn't decrease
- Solution: Check learning rate, try 5e-5 or 1e-3
- Solution: Verify CUDA is being used (check device output)

**Problem**: Loss is NaN/Inf
- Solution: Reduce learning rate (try 1e-5)
- Solution: Check for numerical instability in embedding

**Problem**: Slow training on CPU
- Solution: Use GPU with `--device cuda`
- Solution: Reduce n_samples if memory is limited

**Problem**: Out of memory
- Solution: Reduce embed_dim to 64
- Solution: Reduce n_samples to 20
- Solution: Use CPU (slower but will work)

## Next Steps

After successful training:

1. **Analyze results**
   ```bash
   tensorboard --logdir logs
   ```

2. **Run more evaluation**
   ```bash
   python scripts/evaluate.py \
     --checkpoint checkpoints/checkpoint_epoch_1500.pt \
     --num-evals 20
   ```

3. **Implement extensions** (see README_CPFN.md)
   - Add distribution outputs
   - Train on variable n_features
   - Add unconfoundedness robustness

## Support

For issues or questions:
1. Check the README_CPFN.md for architecture details
2. Review training logs: `tensorboard --logdir logs`
3. Check config.json for parameter values
4. Verify checkpoint files exist before evaluation
