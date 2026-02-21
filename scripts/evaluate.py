"""
Evaluation script for PUA-PFN (Parallel Universe PFN).
Supports both training-mode (full multiverse) and inference-mode (obs + query).
"""

import argparse
import torch
from pathlib import Path

from cpfn.models import MultiverseTransformer
from cpfn.evaluation import CausalDiscoveryEvaluator
from cpfn.utils import Config, get_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate PUA-PFN on causal inference")
    parser.add_argument("--config",     type=str, default="config.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-samples",  type=int, default=30)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--do-val",     type=float, default=10.0)
    parser.add_argument("--device",     type=str, default="auto")
    parser.add_argument("--num-evals",  type=int, default=1)
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "infer", "both"],
        help=(
            "train: use full multiverse (training format).\n"
            "infer: use obs data + query token only (inference format).\n"
            "both:  run and compare both modes."
        ),
    )
    args = parser.parse_args()

    device = get_device(args.device)
    if Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = Config(n_features=args.n_features, device=device)

    model = MultiverseTransformer(
        n_features=config.n_features,
        embed_dim=config.embed_dim,
        n_heads=config.n_heads,
        n_decoder_layers=config.n_decoder_layers,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded model from {args.checkpoint}")

    evaluator = CausalDiscoveryEvaluator(model, device=device)
    modes = ["train", "infer"] if args.mode == "both" else [args.mode]

    for mode in modes:
        all_metrics = []
        print(f"\nRunning {args.num_evals} evaluations in [{mode.upper()}] mode...")
        for i in range(args.num_evals):
            print(f"\n[Evaluation {i+1}/{args.num_evals}]")
            metrics = evaluator.evaluate(
                n_samples=args.n_samples,
                n_features=config.n_features,
                do_val=args.do_val,
                verbose=(i == 0),
                mode=mode,
            )
            all_metrics.append(metrics)

        if args.num_evals > 1:
            print(f"\n{'='*70}")
            print(f"SUMMARY — [{mode.upper()} MODE] ({args.num_evals} evaluations)")
            print(f"{'='*70}")
            avg = lambda key: sum(m[key] for m in all_metrics) / len(all_metrics)
            print(f"Average F1:        {avg('f1'):.4f}")
            print(f"Average Precision: {avg('precision'):.4f}")
            print(f"Average Recall:    {avg('recall'):.4f}")
            print(f"Average NLL:       {avg('nll'):.4f}")
            print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
