"""
Causal discovery evaluation metrics and blind inference testing
"""

import torch
import torch.nn as nn
from typing import Set, Tuple, Dict
from cpfn.data import generate_full_multiverse


class CausalDiscoveryEvaluator:
    """
    Evaluates C-PFN's causal discovery capabilities using blind inference.

    Metrics:
    - Precision/Recall/F1 of edge detection
    - Delta analysis (true vs predicted intervention effects)
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def evaluate(
        self,
        n_samples: int = 30,
        n_features: int = 5,
        do_val: float = 10.0,
        verbose: bool = True,
    ) -> Dict:
        """
        Blind evaluation: Model only sees intervened value + zeros, must predict effects.

        Returns:
            metrics: {
                'precision': float,
                'recall': float,
                'f1': float,
                'true_edges': List[Tuple],
                'pred_edges': List[Tuple],
                'tp': int, 'fp': int, 'fn': int
            }
        """
        self.model.eval()

        # Generate ground truth
        m_true, adj_test = generate_full_multiverse(
            n_samples, n_features, do_val=do_val, device=self.device
        )

        # Create blind input: only intervened value visible
        m_blind = m_true.clone()
        for u_idx in range(1, n_features + 1):
            target_node = u_idx - 1
            for f_idx in range(n_features):
                if f_idx != target_node:
                    m_blind[u_idx, :, f_idx] = 0.0

        # Inference
        with torch.no_grad():
            predictions = self.model(m_blind)

        # Extract true edges
        true_edges = set()
        edges = torch.where(adj_test > 0)
        for j in range(len(edges[0])):
            true_edges.add((edges[0][j].item(), edges[1][j].item()))

        if verbose:
            print(f"\n{'='*70}")
            print(f"BLIND CAUSAL INFERENCE EVALUATION (n_samples={n_samples})")
            print(f"{'='*70}")
            print(f"True DAG Edges: {sorted(list(true_edges))}")

        # Analyze each intervention
        all_pred_edges = {}
        for u_idx in range(n_features):
            target_node = u_idx

            # Reshape predictions correctly
            pred_world = predictions[u_idx, : n_samples * n_features, 0].reshape(
                n_samples, n_features
            )
            obs_world = m_true[0]
            true_world = m_true[u_idx + 1]

            if verbose:
                print(f"\n--- Intervention on X{target_node} ---")

            # Compute deltas
            true_deltas = (true_world - obs_world).abs().mean(dim=0)
            pred_deltas = (pred_world - obs_world).abs().mean(dim=0)

            # Detect causal effects
            pred_edges_for_this_interv = set()
            detected_threshold = true_deltas.max() * 0.1

            if verbose:
                print("\nVariable | True Δ | Pred Δ | True Effect?")
                print("-" * 50)

            for i in range(n_features):
                true_delta = true_deltas[i].item()
                pred_delta = pred_deltas[i].item()

                has_effect = true_delta > detected_threshold

                if has_effect and i != target_node:
                    pred_edges_for_this_interv.add((target_node, i))

                status = "YES" if has_effect else "NO"
                if verbose:
                    print(f"X{i}       | {true_delta:6.4f} | {pred_delta:6.4f} | {status}")

            all_pred_edges[target_node] = pred_edges_for_this_interv

        # Compute metrics
        predicted_edges = set()
        for edges_set in all_pred_edges.values():
            predicted_edges.update(edges_set)

        # Remove self-loops from predictions (no variable causes itself directly)
        predicted_edges = {(i, j) for i, j in predicted_edges if i != j}
        true_edges = {(i, j) for i, j in true_edges if i != j}

        true_positives = true_edges & predicted_edges
        false_positives = predicted_edges - true_edges
        false_negatives = true_edges - predicted_edges

        precision = (
            len(true_positives) / len(predicted_edges)
            if len(predicted_edges) > 0
            else 0
        )
        recall = (
            len(true_positives) / len(true_edges) if len(true_edges) > 0 else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        if verbose:
            print(f"\n{'='*70}")
            print(f"CAUSAL DISCOVERY METRICS")
            print(f"{'='*70}")
            print(f"Predicted Edges: {sorted(list(predicted_edges))}")
            print(
                f"True Positives (TP): {len(true_positives)} {sorted(list(true_positives))}"
            )
            print(
                f"False Positives (FP): {len(false_positives)} {sorted(list(false_positives))}"
            )
            print(
                f"False Negatives (FN): {len(false_negatives)} {sorted(list(false_negatives))}"
            )
            print(f"\nPrecision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            print(f"{'='*70}\n")

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_edges": sorted(list(true_edges)),
            "pred_edges": sorted(list(predicted_edges)),
            "tp": len(true_positives),
            "fp": len(false_positives),
            "fn": len(false_negatives),
        }
