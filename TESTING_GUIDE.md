# C-PFN Testing Guide

Complete testing procedures for the Causal Prior Function Network implementation.

---

## Table of Contents

1. [Quick Verification Test](#quick-verification-test)
2. [Unit Testing](#unit-testing)
3. [Integration Testing](#integration-testing)
4. [Training & Evaluation Testing](#training--evaluation-testing)
5. [Performance Testing](#performance-testing)
6. [Debugging Tips](#debugging-tips)

---

## Quick Verification Test

**Purpose**: Verify installation and basic functionality (30 seconds)

### Command

```bash
cd /home/meisam/code/PUA-PFN
source .venv/bin/activate
python scripts/test.py
```

### What It Tests

✅ Device detection (CUDA/CPU)
✅ Model creation (2.8M parameters)
✅ Data generation (multiverse creation)
✅ Forward pass (output shape)
✅ Evaluation framework (F1/Precision/Recall)

### Expected Output

```
======================================================================
C-PFN Quick Test
======================================================================

✓ Device: cuda
✓ Creating MultiverseTransformer...
  Parameters: 2,837,889
✓ Generating multiverse...
  Multiverse shape: torch.Size([6, 10, 5])
  Adjacency shape: torch.Size([5, 5])
  Causal edges: [[0, 2], [0, 4], [1, 3], [1, 4]]
✓ Running forward pass...
  Output shape: torch.Size([5, 50, 1])
✓ Running blind inference evaluation...
  F1 Score: 0.0000
  Precision: 0.0000
  Recall: 0.0000

======================================================================
✓ All tests passed!
======================================================================
```

### Troubleshooting Quick Test

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'cpfn'` | Run `cd /home/meisam/code/PUA-PFN` first |
| `CUDA out of memory` | Use CPU: `CUDA_VISIBLE_DEVICES="" python scripts/test.py` |
| `torch not found` | Run `source .venv/bin/activate` |

---

## Unit Testing

Test individual components in isolation.

### 1. Test Data Generation

```bash
python << 'EOF'
import torch
from cpfn.data import generate_full_multiverse

# Test basic generation
m_data, adj = generate_full_multiverse(
    n_samples=20,
    n_features=5,
    device='cuda'
)

print(f"✓ Multiverse shape: {m_data.shape}")
print(f"  Expected: [6, 20, 5] (6 universes, 20 samples, 5 features)")
print(f"✓ Adjacency shape: {adj.shape}")
print(f"  Expected: [5, 5]")
print(f"✓ Edge density: {adj.sum().item() / (5*5):.2%}")
print(f"  Expected: ~30% (edge_prob=0.3)")

# Verify observational universe is same in all
print(f"✓ Observational universe (u=0) variance: {m_data[0].var():.4f}")
EOF
```

**Expected**: Multiverse [6, 20, 5], adjacency [5, 5], ~30% edges

### 2. Test Embedding Layer

```bash
python << 'EOF'
import torch
from cpfn.models.embedding import ParallelUniverseEmbedding

# Create embedding
embed = ParallelUniverseEmbedding(
    n_features=5,
    embed_dim=64
)

# Create dummy data [6, 10, 5] (6 universes, 10 samples, 5 features)
m_data = torch.randn(6, 10, 5)

# Forward pass
embedded = embed(m_data)

print(f"✓ Input shape: {m_data.shape}")
print(f"✓ Output shape: {embedded.shape}")
print(f"  Expected: [6, 50, 64] (50 = 10 samples * 5 features)")
print(f"✓ Embedding has gradients: {embedded.requires_grad}")
EOF
```

**Expected**: Output [6, 50, 64], requires_grad=True

### 3. Test Transformer Model

```bash
python << 'EOF'
import torch
from cpfn.models import MultiverseTransformer

# Create model
model = MultiverseTransformer(
    n_features=5,
    embed_dim=128,
    n_heads=8,
    n_decoder_layers=4
)

# Create dummy data
m_data = torch.randn(6, 10, 5)

# Forward pass
output = model(m_data)

print(f"✓ Input shape: {m_data.shape}")
print(f"✓ Output shape: {output.shape}")
print(f"  Expected: [5, 50, 1]")
print(f"  (5 features, 10*5 predictions, 1 value)")
print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"✓ Trainable: {all(p.requires_grad for p in model.parameters())}")
EOF
```

**Expected**: Output [5, 50, 1], all parameters trainable, ~2.8M params

### 4. Test Evaluator

```bash
python << 'EOF'
import torch
from cpfn.models import MultiverseTransformer
from cpfn.evaluation import CausalDiscoveryEvaluator

# Create model
model = MultiverseTransformer(n_features=5, embed_dim=128)
model.eval()

# Create evaluator
evaluator = CausalDiscoveryEvaluator(model, device='cuda')

# Run evaluation
metrics = evaluator.evaluate(
    n_samples=30,
    n_features=5,
    do_val=10.0,
    verbose=True
)

print(f"\n✓ F1 Score: {metrics['f1']:.4f}")
print(f"✓ Precision: {metrics['precision']:.4f}")
print(f"✓ Recall: {metrics['recall']:.4f}")
print(f"✓ TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
EOF
```

**Expected**: Metrics dict with f1, precision, recall, tp, fp, fn (all 0 for untrained)

---

## Integration Testing

Test components working together.

### 1. Test Full Training Loop (1 epoch)

```bash
python << 'EOF'
import torch
from torch.optim import Adam
from cpfn.models import MultiverseTransformer
from cpfn.data import generate_full_multiverse
from cpfn.utils import Config

# Create config
config = Config(num_epochs=1, n_samples=20, learning_rate=1e-4)

# Create model and optimizer
model = MultiverseTransformer(
    n_features=config.n_features,
    embed_dim=config.embed_dim
)
model.to(config.device)
optimizer = Adam(model.parameters(), lr=config.learning_rate)

print("✓ Running 1 training epoch...")

# One epoch
model.train()
m_data, adj = generate_full_multiverse(
    n_samples=config.n_samples,
    n_features=config.n_features,
    device=config.device
)

# Forward pass
output = model(m_data)
loss = output.mean()  # Simple loss for testing

# Backward pass
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

print(f"✓ Loss: {loss.item():.4f}")
print(f"✓ Model training mode: {model.training}")
print(f"✓ Gradients computed: {any(p.grad is not None for p in model.parameters())}")
EOF
```

**Expected**: Loss computed, gradients exist, model is trainable

### 2. Test Checkpoint Save/Load

```bash
python << 'EOF'
import torch
from pathlib import Path
from cpfn.models import MultiverseTransformer
from cpfn.training import Trainer
from cpfn.utils import Config

# Create config and trainer
config = Config(num_epochs=1, n_samples=20)
trainer = Trainer(config)

print("✓ Saving checkpoint...")
checkpoint_path = Path("test_checkpoint.pt")
trainer.save_checkpoint(checkpoint_path, epoch=10)

print("✓ Loading checkpoint...")
trainer.load_checkpoint(checkpoint_path)

print(f"✓ Checkpoint file size: {checkpoint_path.stat().st_size / 1024:.1f} KB")
print(f"✓ Contains: model, optimizer, scheduler, history")

# Clean up
checkpoint_path.unlink()
print("✓ Cleanup complete")
EOF
```

**Expected**: Checkpoint saved/loaded successfully (~50-100 MB)

---

## Training & Evaluation Testing

Test training and evaluation scripts.

### 1. Quick Training Test (10 epochs)

```bash
# Create test directory
mkdir -p test_runs/quick_train
cd test_runs/quick_train

# Run 10 epochs
python ../../scripts/train.py \
  --num-epochs 10 \
  --n-samples 20 \
  --log-dir ./logs \
  --checkpoint-dir ./checkpoints \
  --device cuda

# Check output
ls -lah checkpoints/
ls -lah logs/
```

**Expected Output**:
- Checkpoint saved at epoch 10
- TensorBoard logs in `logs/` directory
- Training loss should decrease slightly (untrained model, random data)

### 2. Full Training Test (100 epochs)

```bash
# This takes ~10-15 minutes on GPU
python scripts/train.py \
  --num-epochs 100 \
  --device cuda

# Monitor with TensorBoard in another terminal:
tensorboard --logdir logs
```

**Expected**:
- Checkpoints saved at epochs 50 and 100
- Loss curve visible in TensorBoard
- Training time ~10-15 min for 100 epochs on GPU

### 3. Evaluation Test

```bash
# After training (or use checkpoint from training)
python scripts/evaluate.py \
  --checkpoint checkpoints/checkpoint_epoch_100.pt \
  --num-evals 5 \
  --device cuda
```

**Expected Output**:
```
Evaluating checkpoint: checkpoints/checkpoint_epoch_100.pt

Run 1:
  F1: 0.2345
  Precision: 0.2567
  Recall: 0.2143

...

Average across 5 runs:
  F1: 0.2450 ± 0.0123
  Precision: 0.2567 ± 0.0145
  Recall: 0.2312 ± 0.0156
```

---

## Performance Testing

Test model performance and speed.

### 1. Throughput Test

```bash
python << 'EOF'
import torch
import time
from cpfn.models import MultiverseTransformer
from cpfn.data import generate_full_multiverse

model = MultiverseTransformer(n_features=5, embed_dim=128)
model.to('cuda')
model.eval()

# Warmup
with torch.no_grad():
    m_data, _ = generate_full_multiverse(n_samples=30, n_features=5, device='cuda')
    _ = model(m_data)

# Benchmark
num_runs = 100
start = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        m_data, _ = generate_full_multiverse(n_samples=30, n_features=5, device='cuda')
        _ = model(m_data)
elapsed = time.time() - start

print(f"✓ Average inference time: {elapsed/num_runs*1000:.2f} ms")
print(f"✓ Throughput: {num_runs/elapsed:.1f} forward passes/sec")
EOF
```

**Expected**: ~30-50 ms per forward pass on modern GPU

### 2. Memory Usage Test

```bash
python << 'EOF'
import torch
from cpfn.models import MultiverseTransformer
from cpfn.data import generate_full_multiverse

torch.cuda.reset_peak_memory_stats()

model = MultiverseTransformer(n_features=5, embed_dim=128)
model.to('cuda')

m_data, _ = generate_full_multiverse(n_samples=50, n_features=5, device='cuda')

with torch.no_grad():
    output = model(m_data)

print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"✓ Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print(f"✓ Model size on disk: ~45 MB")
EOF
```

**Expected**: ~2 GB GPU memory peak for forward pass

### 3. Scaling Test

```bash
python << 'EOF'
import torch
import time
from cpfn.models import MultiverseTransformer
from cpfn.data import generate_full_multiverse

model = MultiverseTransformer(n_features=5, embed_dim=128)
model.to('cuda')
model.eval()

print("Testing different batch sizes:")
for n_samples in [10, 30, 50, 100]:
    start = time.time()
    with torch.no_grad():
        m_data, _ = generate_full_multiverse(
            n_samples=n_samples, 
            n_features=5, 
            device='cuda'
        )
        output = model(m_data)
    elapsed = time.time() - start
    print(f"  n_samples={n_samples:3d}: {elapsed*1000:6.2f} ms, output={output.shape}")
EOF
```

**Expected**: Linear scaling with number of samples

---

## Debugging Tips

### Enable Verbose Output

```bash
# Training with debug info
python << 'EOF'
import logging
logging.basicConfig(level=logging.DEBUG)

from cpfn.training import Trainer
from cpfn.utils import Config

config = Config(num_epochs=1)
trainer = Trainer(config)
trainer.train()
EOF
```

### Check Model Architecture

```bash
python << 'EOF'
from cpfn.models import MultiverseTransformer

model = MultiverseTransformer(n_features=5, embed_dim=128)

print("Model Architecture:")
print(model)
print("\nParameter Count by Layer:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.numel():,}")
EOF
```

### Inspect Gradients During Training

```bash
python << 'EOF'
import torch
from cpfn.models import MultiverseTransformer
from cpfn.data import generate_full_multiverse

model = MultiverseTransformer(n_features=5, embed_dim=128)
model.to('cuda')

m_data, _ = generate_full_multiverse(n_samples=30, n_features=5, device='cuda')

output = model(m_data)
loss = output.mean()
loss.backward()

print("Gradient Statistics:")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"  {name}: grad_mean={param.grad.mean():.4f}, grad_std={param.grad.std():.4f}")
EOF
```

### Test on CPU (for debugging)

```bash
python << 'EOF'
import torch
from cpfn.models import MultiverseTransformer
from cpfn.data import generate_full_multiverse

# Force CPU
device = 'cpu'

model = MultiverseTransformer(n_features=5, embed_dim=64)  # Smaller for CPU
model.to(device)

m_data, _ = generate_full_multiverse(n_samples=10, n_features=5, device=device)

output = model(m_data)
print(f"✓ CPU mode works: output shape {output.shape}")
EOF
```

### Monitor GPU Memory

```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

### Verify Data Shapes

```bash
python << 'EOF'
import torch
from cpfn.data import generate_full_multiverse

m_data, adj = generate_full_multiverse(n_samples=30, n_features=5)

print(f"Multiverse tensor:")
print(f"  Shape: {m_data.shape} = [universes, samples, features]")
print(f"  U0 (observational): {m_data[0].shape}")
print(f"  U1-U5 (interventional): {m_data[1:].shape}")
print(f"  Min/Max: {m_data.min():.2f} / {m_data.max():.2f}")

print(f"\nAdjacency matrix:")
print(f"  Shape: {adj.shape}")
print(f"  Edges: {adj.sum().item()}")
print(f"  Density: {adj.sum().item() / (5*5):.2%}")
EOF
```

---

## Test Coverage Summary

| Component | Test | Time | Status |
|-----------|------|------|--------|
| Data generation | `test.py` | <1s | ✅ Quick |
| Embedding layer | Unit test | <1s | ✅ Quick |
| Transformer | Unit test | <1s | ✅ Quick |
| Evaluator | Unit test | 5s | ✅ Quick |
| Training loop | Integration | 2-5 min | ⏱️ Medium |
| Full training | 100 epochs | 10-15 min | ⏱️ Medium |
| Evaluation | 10 runs | 2-3 min | ⏱️ Medium |
| Performance | Throughput | <1s | ✅ Quick |

---

## Continuous Testing Workflow

### Before Committing Code

```bash
# 1. Quick test
python scripts/test.py

# 2. Unit tests
python << 'EOF'
# Run all unit tests above
EOF

# 3. Quick training (1-2 min)
python scripts/train.py --num-epochs 10 --device cuda

# 4. If passing: proceed to commit
```

### Weekly Full Testing

```bash
# Full training run
python scripts/train.py --num-epochs 1500 --device cuda

# Evaluation
python scripts/evaluate.py \
  --checkpoint checkpoints/checkpoint_epoch_1500.pt \
  --num-evals 10

# Check results and metrics
```

---

## Common Test Failures

| Error | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: CUDA out of memory` | Batch too large | Reduce `n_samples` or use CPU |
| `ModuleNotFoundError` | Wrong directory | Run from project root |
| `Shape mismatch` | Wrong config | Check n_features, embed_dim match |
| `NaN loss` | Learning rate too high | Reduce `learning_rate` |
| `No module named cpfn` | Virtual env not activated | `source .venv/bin/activate` |

---

## Next Steps

✅ Run `python scripts/test.py` to verify setup
✅ Run 10-epoch training test
✅ Monitor with TensorBoard
✅ Run evaluation on trained checkpoint
✅ Proceed to full 1500-epoch training

Good luck! 🚀
