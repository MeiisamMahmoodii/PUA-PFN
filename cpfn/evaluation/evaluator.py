"""
Causal discovery evaluation — PUA-PFN.

Two evaluation paths:
1. train_mode_evaluate()  — uses full multiverse (training-time format)
2. infer_mode_evaluate()  — uses obs data + query token (inference-time format)

Edge detection uses the MODEL'S predicted deltas (not ground truth),
fixing the original evaluator bug.
"""

import torch
import torch.nn as nn
from typing import Dict
from cpfn.data import generate_full_multiverse


class CausalDiscoveryEvaluator:
    """
    Evaluates PUA-PFN on causal effect prediction and causal graph discovery.

    Metrics:
    - Precision / Recall / F1 of edge detection
    - Delta analysis: true Δ vs predicted Δ (mean of bar-distribution)
    - NLL of predicted bar-distribution over true interventional outcomes
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model  = model.to(device)
        self.device = device

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        n_samples: int = 30,
        n_features: int = 5,
        do_val: float = 10.0,
        verbose: bool = True,
        mode: str = "train",  # "train" | "infer"
    ) -> Dict:
        """
        Run one evaluation episode.

        Args:
            mode: "train" uses the full multiverse forward pass.
                  "infer" uses only obs data + query token (inference mode).
        """
        if mode == "train":
            return self._train_mode_evaluate(n_samples, n_features, do_val, verbose)
        elif mode == "infer":
            return self._infer_mode_evaluate(n_samples, n_features, do_val, verbose)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'infer'.")

    # ------------------------------------------------------------------ #
    #  Training-mode evaluation (full multiverse)                         #
    # ------------------------------------------------------------------ #

    def _train_mode_evaluate(self, n_samples, n_features, do_val, verbose) -> Dict:
        self.model.eval()

        # Generate ground truth
        m_true, adj = generate_full_multiverse(
            n_samples, n_features, do_val=do_val,
            randomise_prior=False, device=self.device
        )

        # Set bar-distribution borders based on target range
        targets_flat = m_true[1:].reshape(-1)
        self.model.bar_head.set_borders(
            targets_flat.min().item(),
            targets_flat.max().item(),
            device=self.device,
        )

        with torch.no_grad():
            logits = self.model(m_true)   # [K, s*f, n_bins]

        return self._compute_metrics(
            logits, m_true, adj, n_samples, n_features, verbose, mode="train"
        )

    # ------------------------------------------------------------------ #
    #  Inference-mode evaluation (obs data + query token only)            #
    # ------------------------------------------------------------------ #

    def _infer_mode_evaluate(self, n_samples, n_features, do_val, verbose) -> Dict:
        self.model.eval()

        m_true, adj = generate_full_multiverse(
            n_samples, n_features, do_val=do_val,
            randomise_prior=False, device=self.device
        )

        obs_data = m_true[0]  # [n_samples, n_features]

        # Set bar-distribution borders
        targets_flat = m_true[1:].reshape(-1)
        self.model.bar_head.set_borders(
            targets_flat.min().item(),
            targets_flat.max().item(),
            device=self.device,
        )

        # Collect per-variable logits using inference mode
        all_logits = []
        with torch.no_grad():
            for var_idx in range(n_features):
                logits_var, _ = self.model.infer(obs_data, var_idx, do_val)
                all_logits.append(logits_var.unsqueeze(0))  # [1, s*f, n_bins]

        logits = torch.cat(all_logits, dim=0)  # [K, s*f, n_bins]

        return self._compute_metrics(
            logits, m_true, adj, n_samples, n_features, verbose, mode="infer"
        )

    # ------------------------------------------------------------------ #
    #  Shared metric computation                                          #
    # ------------------------------------------------------------------ #

    def _compute_metrics(
        self, logits, m_true, adj, n_samples, n_features, verbose, mode
    ) -> Dict:
        true_edges = set()
        edges = torch.where(adj > 0)
        for j in range(len(edges[0])):
            true_edges.add((edges[0][j].item(), edges[1][j].item()))
        true_edges = {(i, k) for i, k in true_edges if i != k}

        if verbose:
            print(f"\n{'='*70}")
            print(f"PUA-PFN EVALUATION [{mode.upper()} MODE]  (n_samples={n_samples})")
            print(f"{'='*70}")
            print(f"True DAG Edges: {sorted(list(true_edges))}")

        obs_world = m_true[0]   # [n_samples, n_features]
        all_pred_edges = {}
        total_nll = 0.0

        for u_idx in range(n_features):
            target_node = u_idx
            true_world  = m_true[u_idx + 1]  # [n_samples, n_features]

            # Get mean prediction from bar-distribution
            logits_u   = logits[u_idx]   # [s*f, n_bins]
            mean_pred  = self.model.bar_head.mean(logits_u)   # [s*f]
            pred_world = mean_pred.view(n_samples, n_features)

            # Bar-distribution NLL on true interventional outcomes
            nll = self.model.bar_head.nll_loss(
                logits_u,
                true_world.reshape(-1)
            )
            total_nll += nll.item()

            # Deltas
            true_deltas = (true_world - obs_world).abs().mean(dim=0)   # [f]
            pred_deltas = (pred_world - obs_world).abs().mean(dim=0)   # [f]

            # Edge detection: find the natural gap in PREDICTED deltas.
            # Sort all non-self predicted deltas; use the midpoint between the
            # two largest and two smallest as threshold. This is robust whether
            # the model has learned to predict low for the intervened var or not.
            non_self_deltas = torch.stack([
                pred_deltas[i] for i in range(n_features) if i != target_node
            ])
            sorted_deltas, _ = non_self_deltas.sort(descending=True)

            if len(sorted_deltas) >= 2:
                # Gap-based threshold: halfway between highest and lowest pred delta
                # among non-self variables
                gap_threshold = (sorted_deltas[0] + sorted_deltas[-1]) / 2.0
                pred_threshold = gap_threshold.item()
            else:
                pred_threshold = non_self_deltas.mean().item() * 0.5

            pred_threshold = max(pred_threshold, 0.05)

            pred_edges_this = set()
            for i in range(n_features):
                if i != target_node and pred_deltas[i].item() > pred_threshold:
                    pred_edges_this.add((target_node, i))

            all_pred_edges[target_node] = pred_edges_this

            if verbose:
                print(f"\n--- Intervention on X{target_node} ---")
                print("Variable | True Δ | Pred Δ | Predicted edge?")
                print("-" * 54)
                for i in range(n_features):
                    td = true_deltas[i].item()
                    pd = pred_deltas[i].item()
                    edge = "YES" if (target_node, i) in pred_edges_this else "NO "
                    mark = " ←" if i == target_node else ""
                    print(f"  X{i}    | {td:6.4f} | {pd:6.4f} | {edge}{mark}")

        # Aggregate edges
        predicted_edges = set()
        for es in all_pred_edges.values():
            predicted_edges.update(es)
        predicted_edges = {(i, j) for i, j in predicted_edges if i != j}

        tp = true_edges & predicted_edges
        fp = predicted_edges - true_edges
        fn = true_edges - predicted_edges

        precision = len(tp) / len(predicted_edges) if predicted_edges else 0.0
        recall    = len(tp) / len(true_edges)      if true_edges      else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        avg_nll   = total_nll / n_features

        if verbose:
            print(f"\n{'='*70}")
            print("CAUSAL DISCOVERY METRICS")
            print(f"{'='*70}")
            print(f"Predicted Edges:     {sorted(list(predicted_edges))}")
            print(f"True Positives  (TP): {len(tp)}  {sorted(list(tp))}")
            print(f"False Positives (FP): {len(fp)}  {sorted(list(fp))}")
            print(f"False Negatives (FN): {len(fn)}  {sorted(list(fn))}")
            print(f"\nPrecision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            print(f"Avg NLL:   {avg_nll:.4f}")
            print(f"{'='*70}\n")

        return {
            "precision":    precision,
            "recall":       recall,
            "f1":           f1,
            "nll":          avg_nll,
            "true_edges":   sorted(list(true_edges)),
            "pred_edges":   sorted(list(predicted_edges)),
            "tp":           len(tp),
            "fp":           len(fp),
            "fn":           len(fn),
        }
