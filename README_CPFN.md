# Causal Prior Function Network (C-PFN)

A foundation model for causal inference via parallel universe reasoning.

## Quick Start

### 1. Training

Train the model from scratch:

```bash
python scripts/train.py \
  --num-epochs 1500 \
  --n-features 5 \
  --n-samples 30 \
  --device auto
```

Options:
- `--config config.json`: Config file path (created automatically)
- `--num-epochs`: Number of training epochs
- `--n-features`: Number of variables in SCMs
- `--n-samples`: Number of samples per SCM
- `--learning-rate`: Learning rate
- `--device`: cuda/cpu/auto
- `--resume <checkpoint>`: Resume from checkpoint

### 2. Evaluation

Evaluate causal discovery on blind inference:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/checkpoint_epoch_1000.pt \
  --config config.json \
  --num-evals 5 \
  --device auto
```

Options:
- `--checkpoint`: Path to trained model
- `--config`: Config file
- `--n-samples`: Samples for evaluation
- `--n-features`: Number of features
- `--do-val`: Intervention value
- `--num-evals`: Number of evaluation runs (for averaging)

## Project Structure

```
cpfn/
├── data/              # SCM generation
│   ├── scm.py        # CausalMechanism, generate_full_multiverse
│   └── __init__.py
├── models/            # Architecture components
│   ├── embedding.py   # ParallelUniverseEmbedding
│   ├── blocks.py      # CrossUniverseBlock
│   ├── transformer.py # MultiverseTransformer
│   └── __init__.py
├── training/          # Training loop
│   ├── trainer.py     # Trainer class
│   └── __init__.py
├── evaluation/        # Evaluation metrics
│   ├── evaluator.py   # CausalDiscoveryEvaluator
│   └── __init__.py
├── utils/             # Utilities
│   ├── config.py      # Configuration management
│   ├── device.py      # Device utilities
│   └── __init__.py
└── __init__.py

scripts/
├── train.py           # Full training script
├── evaluate.py        # Evaluation script
└── __init__.py
```

## Architecture

### ParallelUniverseEmbedding
- Encodes multiverse data with 4 embedding dimensions:
  - Value: The numerical value
  - Feature: Which variable (0-4)?
  - Universe: Observational (0) or Interventional (1)?
  - Intervention Flag: Is this the intervention target?

### CrossUniverseBlock
- Intra-attention: Self-attention within interventional universe
- Cross-attention: Compare interventional against observational
- FFN: Standard transformer feed-forward

### MultiverseTransformer
- Embedding layer: Encodes all universes
- Obs encoder: Deep transformer on observational universe (Universe 0)
- Int decoder: 4× CrossUniverseBlock layers for interventional universes
- Output head: Predicts intervention effects

## Training Details

### Meta-Learning Approach
- Each epoch generates a **new random SCM** with:
  - Random acyclic graph (DAG) with 30% edge probability
  - Random causal mechanisms (frozen neural networks)
  - Random exogenous noise
  - Random intervention value (uniform [2, 15])

- **Model never sees the same "physics" twice** → learns general causal reasoning

### Loss Function
- MSE loss between predicted and true interventional universe values
- Target shape: `[n_features, n_samples * n_features, 1]`

### Optimization
- Adam optimizer with learning rate 1e-4
- Cosine annealing warmup restart scheduler
- Gradient clipping (max norm 1.0)

## Evaluation

### Blind Inference Test
- Generate new SCM the model has never seen
- Create "blind" input: only intervened variables visible (others zeroed)
- Model must predict all variable values under intervention
- Compare to ground truth

### Metrics
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1**: Harmonic mean of precision and recall
- **Edge Detection**: True/False positive/negative edges

## Key Features

✅ **Meta-Learning**: Diverse SCMs during training
✅ **Causal-Aware**: Explicit intervention flags in embedding
✅ **Cross-Universe Reasoning**: Attention between observational and interventional
✅ **Blind Inference**: Zero-shot causal discovery without graph knowledge
✅ **Uncertainty Quantification**: Supports extension with distribution outputs
✅ **Checkpoint Management**: Save/load models and training state
✅ **Tensorboard Logging**: Monitor training progress

## Extensions (Aligned with do-PFN)

To extend this to match do-PFN's approach:

1. **Distribution Outputs**
   ```python
   # Instead of point predictions, output mean + variance
   output_mean = self.output_head_mean(u_int)
   output_std = self.output_head_std(u_int)
   ```

2. **Variable n_features**
   ```python
   # Train on 3, 5, 7, 10 features
   # Allows generalization to unseen feature counts
   ```

3. **Unconfoundedness Robustness**
   ```python
   # Train on SCMs with unobserved confounders
   # Learn implicit adjustment sets
   ```

4. **Formal Theory**
   - Prove consistency: with infinite data, model converges to true CID
   - Sample complexity bounds
   - Identifiability analysis

## Citation

If you use this code, please cite:

```bibtex
@article{pfn2024,
  title={Causal Prior Function Networks},
  author={Your Name},
  year={2024}
}
```

## References

- [do-PFN: In-Context Learning for Causal Effect Estimation](https://arxiv.org/abs/2506.06039)
- [TabPFN: In-Context Learning for Tabular Data](https://arxiv.org/abs/2207.01848)
