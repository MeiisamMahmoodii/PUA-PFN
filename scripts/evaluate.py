#!/usr/bin/env python3
"""
Evaluation script for C-PFN causal discovery
"""

import argparse
import torch
from pathlib import Path

from cpfn.models import MultiverseTransformer
from cpfn.evaluation import CausalDiscoveryEvaluator
from cpfn.utils import Config, get_device


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate C-PFN on blind causal inference task"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=30,
        help="Number of samples for evaluation",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=5,
        help="Number of features",
    )
    parser.add_argument(
        "--do-val",
        type=float,
        default=10.0,
        help="Intervention value",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--num-evals",
        type=int,
        default=1,
        help="Number of evaluations to run",
    )

    args = parser.parse_args()

    # Load config
    device = get_device(args.device)
    if Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = Config(
            n_features=args.n_features,
            device=device,
        )

    # Load model
    model = MultiverseTransformer(
        n_features=config.n_features,
        embed_dim=config.embed_dim,
        n_heads=config.n_heads,
        n_decoder_layers=config.n_decoder_layers,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded model from {args.checkpoint}")

    # Evaluate
    evaluator = CausalDiscoveryEvaluator(model, device=device)

    print(f"\nRunning {args.num_evals} evaluations...")
    all_metrics = []
    for i in range(args.num_evals):
        print(f"\n[Evaluation {i+1}/{args.num_evals}]")
        metrics = evaluator.evaluate(
            n_samples=args.n_samples,
            n_features=config.n_features,
            do_val=args.do_val,
            verbose=(i == 0),  # Verbose output for first evaluation
        )
        all_metrics.append(metrics)

    # Summary
    if args.num_evals > 1:
        print(f"\n{'='*70}")
        print(f"SUMMARY ({args.num_evals} evaluations)")
        print(f"{'='*70}")
        avg_f1 = sum(m["f1"] for m in all_metrics) / len(all_metrics)
        avg_precision = (
            sum(m["precision"] for m in all_metrics) / len(all_metrics)
        )
        avg_recall = sum(m["recall"] for m in all_metrics) / len(all_metrics)

        print(f"Average F1:        {avg_f1:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall:    {avg_recall:.4f}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
