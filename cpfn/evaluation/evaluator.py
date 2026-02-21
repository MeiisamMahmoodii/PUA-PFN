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

    def __init__(self, model: nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
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

        m_true, adj = generate_full_multiverse(
            n_samples, n_features, do_val=do_val,
            randomise_prior=False, device=self.device
        )

        targets_flat = m_true[1:].reshape(-1)
        self.model.bar_head.set_borders(
            targets_flat.min().item(),
            targets_flat.max().item(),
            device=self.device,
        )

        with torch.no_grad():
            logits, gated_means, gate, obs_ctx = self.model(m_true)   # 4-tuple

        # Edge prediction: use data-conditioned gate (principled, learned)
        pred_adj = self.model.causal_gate.hard_adjacency(obs_ctx)     # [K, K]
        predicted_edges = {
            (i, j)
            for i in range(n_features)
            for j in range(n_features)
            if i != j and pred_adj[i, j].item() > 0.5
        }

        return self._compute_metrics(
            logits, m_true, adj, n_samples, n_features, verbose,
            mode="train", predicted_edges=predicted_edges,
            obs_ctx=obs_ctx
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
        self, logits, m_true, adj, n_samples, n_features, verbose, mode,
        predicted_edges=None, obs_ctx=None
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

        obs_world   = m_true[0]
        total_nll   = 0.0
        # For infer mode or verbose delta table, track per-intervention edges
        gap_pred_edges = {}

        for u_idx in range(n_features):
            target_node = u_idx
            true_world  = m_true[u_idx + 1]

            logits_u   = logits[u_idx]
            mean_pred  = self.model.bar_head.mean(logits_u)
            pred_world = mean_pred.view(n_samples, n_features)

            nll = self.model.bar_head.nll_loss(logits_u, true_world.reshape(-1))
            total_nll += nll.item()

            true_deltas = (true_world - obs_world).abs().mean(dim=0)
            pred_deltas = (pred_world - obs_world).abs().mean(dim=0)

            # Gap threshold (used in infer mode, and for verbose display)
            non_self_deltas = torch.stack([
                pred_deltas[i] for i in range(n_features) if i != target_node
            ])
            sorted_d, _ = non_self_deltas.sort(descending=True)
            if len(sorted_d) >= 2:
                gap_threshold = (sorted_d[0] + sorted_d[-1]) / 2.0
            else:
                gap_threshold = non_self_deltas.mean() * 0.5
            gap_threshold = max(gap_threshold.item(), 0.05)

            gap_edges_this = set()
            for i in range(n_features):
                if i != target_node and pred_deltas[i].item() > gap_threshold:
                    gap_edges_this.add((target_node, i))
            gap_pred_edges[target_node] = gap_edges_this

            if verbose:
                show_gate = (mode == "train" and obs_ctx is not None
                             and hasattr(self.model, "causal_gate"))
                gate_probs = (
                    self.model.causal_gate.edge_probs(obs_ctx).detach()
                    if show_gate else None
                )

                print(f"\n--- Intervention on X{target_node} ---")
                header = "Variable | True Δ | Pred Δ | Gate P" if show_gate else "Variable | True Δ | Pred Δ"
                print(header)
                print("-" * (60 if show_gate else 40))
                for i in range(n_features):
                    td = true_deltas[i].item()
                    pd = pred_deltas[i].item()
                    mark = " ←" if i == target_node else ""
                    if show_gate:
                        gp = gate_probs[target_node, i].item() if i != target_node else float("nan")
                        in_gate = (predicted_edges is not None and (target_node, i) in predicted_edges)
                        g_str = f"{gp:.3f} {'✓' if in_gate else ' '}" if i != target_node else "  self"
                        print(f"  X{i}    | {td:6.4f} | {pd:6.4f} | {g_str}{mark}")
                    else:
                        in_gap = (target_node, i) in gap_edges_this
                        print(f"  X{i}    | {td:6.4f} | {pd:6.4f} | {'YES' if in_gap else 'NO '}{mark}")

        # Final predicted edges: use gate if provided (train), else gap (infer)
        if predicted_edges is None:
            predicted_edges = set()
            for es in gap_pred_edges.values():
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
