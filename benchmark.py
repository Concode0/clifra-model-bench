#!/usr/bin/env python3
"""Correctness checks and paired timing for a fixed geometric model."""

import argparse
import importlib

import torch

from bench.runner import run, select_device

MODELS = {
    "gca_mlp": "models.gca_mlp.benchmark",
    "gca_gnn": "models.gca_gnn.benchmark",
    "clifford_resnet2d": "models.clifford_resnet2d.benchmark",
    "clifford_fno2d": "models.clifford_fno2d.benchmark",
    "clifford_fno3d": "models.clifford_fno3d.benchmark",
    "cgenn": "models.cgenn.benchmark",
    "gatr": "models.gatr.benchmark",
}


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--model", choices=MODELS, default="gca_mlp")
    known, _ = pre.parse_known_args()
    model_benchmark = importlib.import_module(MODELS[known.model])
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--backend", choices=("both", "reference", "clifra"), default="both")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--mode",
        choices=("forward", "forward+backward", "both"),
        default="both",
        help="time inference forward, or a new grad-enabled forward plus loss and backward",
    )
    parser.add_argument("--compile", action="store_true", help="time torch.compile steady state")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--cpu-interop-threads", type=int)
    model_benchmark.add_arguments(parser)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters <= 0:
        parser.error("--warmup must be nonnegative and --iters must be positive")
    model_benchmark.check_arguments(parser, args)
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    if args.cpu_interop_threads is not None:
        torch.set_num_interop_threads(args.cpu_interop_threads)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    run(model_benchmark.prepare(args, device), args, device)


if __name__ == "__main__":
    main()
