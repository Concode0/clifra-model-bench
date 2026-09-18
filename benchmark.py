#!/usr/bin/env python3
"""Correctness checks and paired timing for the two GCA-MLP implementations."""

from __future__ import annotations

import argparse
import contextlib
import math
import platform
import dataclasses
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from clifra_model import ClifraGCAMLP
from reference import ReferenceGCAMLP


@dataclasses.dataclass
class BenchmarkTarget:
    name: str
    model: torch.nn.Module
    to_model_domain: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    to_standard_domain: Callable[[torch.Tensor], torch.Tensor] = lambda x: x


class ConventionAdapter:
    """Algebra isomorphism from cliffordlayers' null-first basis to Clifra."""

    def __init__(self, reference_algebra, clifra_algebra) -> None:
        reference_bitmaps = [int(value) for value in reference_algebra.bbo.index_to_bitmap]
        clifra_basis = clifra_algebra.layout().basis_indices
        clifra_position = {blade: position for position, blade in enumerate(clifra_basis)}
        clifra_for_reference = []
        signs = []
        for bitmap in reference_bitmaps:
            # Reference generators (e0,e1,e2,e3) map to Clifra's (e4,e1,e2,e3),
            # where e4 is null. Reordering the wedge orientation may add a sign.
            targets = [3 if bit == 0 else bit - 1 for bit in range(4) if bitmap & (1 << bit)]
            inversions = sum(
                targets[left] > targets[right]
                for left in range(len(targets))
                for right in range(left + 1, len(targets))
            )
            blade = sum(1 << bit for bit in targets)
            clifra_for_reference.append(clifra_position[blade])
            signs.append(-1.0 if inversions % 2 else 1.0)

        self.clifra_for_reference = tuple(clifra_for_reference)
        self.signs_for_reference = tuple(signs)
        reference_for_clifra = [0] * 16
        signs_for_clifra = [0.0] * 16
        for reference_lane, clifra_lane in enumerate(clifra_for_reference):
            reference_for_clifra[clifra_lane] = reference_lane
            signs_for_clifra[clifra_lane] = signs[reference_lane]
        self.reference_for_clifra = tuple(reference_for_clifra)
        self.signs_for_clifra = tuple(signs_for_clifra)

    def to_clifra(self, values: torch.Tensor) -> torch.Tensor:
        indices = torch.tensor(self.reference_for_clifra, device=values.device)
        signs = values.new_tensor(self.signs_for_clifra)
        return torch.index_select(values, -1, indices) * signs

    def to_reference(self, values: torch.Tensor) -> torch.Tensor:
        indices = torch.tensor(self.clifra_for_reference, device=values.device)
        signs = values.new_tensor(self.signs_for_reference)
        return torch.index_select(values, -1, indices) * signs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--backend", choices=("both", "reference", "clifra"), default="both")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--items", type=int, default=64)
    parser.add_argument("--in-channels", type=int, default=8)
    parser.add_argument("--out-channels", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--act-agg", choices=("linear", "sum", "mean"), default="linear")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--mode", choices=("forward", "backward", "both"), default="both")
    parser.add_argument("--compile", action="store_true", help="time torch.compile steady state")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--cpu-interop-threads", type=int)
    args = parser.parse_args()
    positive = (
        "batch",
        "items",
        "in_channels",
        "out_channels",
        "hidden_channels",
        "hidden_layers",
        "iters",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    return args


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def git_commit(path: Path) -> str:
    if not (path / ".git").exists():
        return "not a git checkout"
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def print_provenance(args: argparse.Namespace, device: torch.device, counts: dict[str, int]) -> None:
    local_clifra = Path(__file__).resolve().parent.parent / "clifra"
    print("=== Provenance ===")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"PyTorch: {torch.__version__}")
    print(f"Clifra: {package_version('clifra')}")
    print(f"Clifra git commit: {git_commit(local_clifra)}")
    print(f"cliffordlayers: {package_version('cliffordlayers')}")
    print(f"platform: {platform.platform()} / {platform.machine()}")
    if platform.system() == "Darwin":
        print(f"macOS: {platform.mac_ver()[0]}")
    print(f"device: {device}")
    print("dtype: torch.float32")
    print(f"CPU intra-op threads: {torch.get_num_threads()}")
    print(f"CPU inter-op threads: {torch.get_num_interop_threads()}")
    print(
        "model: "
        f"in={args.in_channels}, out={args.out_channels}, hidden={args.hidden_channels}, "
        f"hidden_layers={args.hidden_layers}, act_agg={args.act_agg}, "
        f"batch={args.batch}, items={args.items}"
    )
    print("parameter counts: " + ", ".join(f"{name}={count:,}" for name, count in counts.items()))
    print(f"execution: {'compiled' if args.compile else 'eager'}, mode={args.mode}")


def transfer_reference_to_clifra(
    reference: ReferenceGCAMLP, clifra: ClifraGCAMLP, adapter: ConventionAdapter
) -> None:
    """Transfer every learnable parameter through the coefficient isomorphism."""
    with torch.no_grad():
        for source, target in zip(reference.linears, clifra.linears):
            target.weight.copy_(source.weight)
            target.embed_e0.copy_(source.embed_e0)
            target._action.zero_()
            target_positions = {
                blade: position for position, blade in enumerate(target.action_basis_indices)
            }
            for source_position, reference_lane in enumerate(source.action_blades):
                clifra_lane = adapter.clifra_for_reference[reference_lane]
                sign = adapter.signs_for_reference[reference_lane]
                target_position = target_positions[clifra_lane]
                target._action[..., target_position].copy_(sign * source._action[..., source_position])
        for source, target in zip(reference.activations, clifra.activations):
            if hasattr(source, "conv"):
                target.conv.weight.copy_(
                    adapter.to_clifra(source.conv.weight.squeeze(1)).unsqueeze(1)
                )
                target.conv.bias.copy_(source.conv.bias)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def assert_finite_backward(name: str, model: torch.nn.Module, inputs: torch.Tensor) -> None:
    model.zero_grad(set_to_none=True)
    probe = inputs.detach().clone().requires_grad_(True)
    model(probe).square().mean().backward()
    if probe.grad is None or not torch.isfinite(probe.grad).all():
        raise AssertionError(f"{name}: input gradient is missing or non-finite")
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad and (
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
        ):
            raise AssertionError(f"{name}: non-finite or missing gradient for {parameter_name}")
    model.zero_grad(set_to_none=True)


def validate_correctness(
    targets: list[BenchmarkTarget],
    inputs: dict[str, torch.Tensor],
    adapter: ConventionAdapter | None,
    expected_shape: tuple[int, ...],
) -> None:
    print("\n=== Correctness ===")
    if adapter is not None and "reference" in inputs:
        roundtrip = adapter.to_reference(adapter.to_clifra(inputs["reference"]))
        torch.testing.assert_close(roundtrip, inputs["reference"], rtol=0.0, atol=0.0)
        print("convention roundtrip: PASS")

    outputs = {}
    for target in targets:
        name = target.name
        model = target.model
        output = model(inputs[name])
        if output.shape != expected_shape:
            raise AssertionError(f"{name}: got shape {tuple(output.shape)}, expected {expected_shape}")
        if not torch.isfinite(output).all():
            raise AssertionError(f"{name}: forward output is non-finite")
        assert_finite_backward(name, model, inputs[name])
        outputs[name] = output.detach()
        print(f"{name}: shape, finite forward, finite backward PASS")

    # Check whole-model equivalence between the first target and the rest in standard domain
    if len(targets) > 1:
        base_target = targets[0]
        base_standard_output = base_target.to_standard_domain(outputs[base_target.name])
        for target in targets[1:]:
            standard_output = target.to_standard_domain(outputs[target.name])
            whole_error = float((base_standard_output - standard_output).abs().max().detach().cpu())
            torch.testing.assert_close(standard_output, base_standard_output, rtol=1e-3, atol=1e-4)
            print(f"{target.name} matches {base_target.name} standard-domain output: PASS (max={whole_error:.3e})")

    # Legacy per-layer validation specific to reference vs clifra
    models_dict = {t.name: t.model for t in targets}
    if adapter is not None and set(models_dict) == {"reference", "clifra"} and "reference" in inputs:
        generator = torch.Generator(device="cpu").manual_seed(1729)
        max_linear_error = 0.0
        max_activation_error = 0.0
        for reference_layer, clifra_layer in zip(
            models_dict["reference"].linears, models_dict["clifra"].linears
        ):
            source = torch.randn(3, reference_layer.in_features, 16, generator=generator).to(
                inputs["reference"].device
            )
            expected = adapter.to_clifra(reference_layer(source))
            actual = clifra_layer(adapter.to_clifra(source))
            max_linear_error = max(
                max_linear_error, float((expected - actual).abs().max().detach().cpu())
            )
            torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-5)
        for reference_activation, clifra_activation in zip(
            models_dict["reference"].activations, models_dict["clifra"].activations
        ):
            source = torch.randn(
                3, models_dict["reference"].linears[0].out_features, 16, generator=generator
            ).to(inputs["reference"].device)
            expected = adapter.to_clifra(reference_activation(source))
            actual = clifra_activation(adapter.to_clifra(source))
            max_activation_error = max(
                max_activation_error, float((expected - actual).abs().max().detach().cpu())
            )
            torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-5)
        print(
            "transferred layer equivalence: PASS "
            f"(linear max={max_linear_error:.3e}, activation max={max_activation_error:.3e})"
        )

    print(
        "equivariance: not asserted; upstream documents GCA-MLP as using PGA "
        "representations but not E(3)-equivariant"
    )


def execute_once(model: torch.nn.Module, inputs: torch.Tensor, mode: str) -> torch.Tensor:
    outputs = model(inputs)
    if mode == "backward":
        outputs.square().mean().backward()
    return outputs


def compile_for_mode(
    targets: list[BenchmarkTarget],
    inputs: dict[str, torch.Tensor],
    mode: str,
    device: torch.device,
) -> list[BenchmarkTarget]:
    compiled = []
    print(f"\n=== torch.compile preparation: {mode} ===")
    for target in targets:
        name = target.name
        model = target.model
        try:
            start = time.perf_counter_ns()
            candidate = torch.compile(model)
            api_ms = (time.perf_counter_ns() - start) / 1e6
            model.zero_grad(set_to_none=True)
            inputs[name].grad = None
            with torch.inference_mode():
                expected = model(inputs[name]).detach()
            synchronize(device)
            start = time.perf_counter_ns()
            context = torch.inference_mode() if mode == "forward" else contextlib.nullcontext()
            with context:
                actual = execute_once(candidate, inputs[name], mode)
            synchronize(device)
            first_ms = (time.perf_counter_ns() - start) / 1e6
            torch.testing.assert_close(actual.detach(), expected, rtol=1e-3, atol=1e-4)
            compiled.append(dataclasses.replace(target, model=candidate))
            print(
                f"{name}: compile API={api_ms:.3f} ms, first call={first_ms:.3f} ms, "
                "output check=PASS"
            )
        except Exception as error:  # torch.compile failures vary by backend.
            print(f"{name}: COMPILE FAILED: {type(error).__name__}: {error}")
        finally:
            model.zero_grad(set_to_none=True)
            inputs[name].grad = None
    return compiled


def timed_samples(
    targets: list[BenchmarkTarget],
    inputs: dict[str, torch.Tensor],
    mode: str,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, list[float]]:
    samples = {target.name: [] for target in targets}
    context = torch.inference_mode() if mode == "forward" else contextlib.nullcontext()
    with context:
        for phase_iterations, record in ((warmup, False), (iterations, True)):
            for iteration in range(phase_iterations):
                order = targets if iteration % 2 == 0 else list(reversed(targets))
                for target in order:
                    name = target.name
                    model = target.model
                    if mode == "backward":
                        model.zero_grad(set_to_none=True)
                        inputs[name].grad = None
                    synchronize(device)
                    start = time.perf_counter_ns()
                    execute_once(model, inputs[name], mode)
                    synchronize(device)
                    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                    if record:
                        samples[name].append(elapsed_ms)
    for target in targets:
        target.model.zero_grad(set_to_none=True)
        inputs[target.name].grad = None
    return samples


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def report_samples(
    samples: dict[str, list[float]], args: argparse.Namespace, counts: dict[str, int], mode: str
) -> None:
    print(f"\n=== {mode} ({'compiled' if args.compile else 'eager'}) ===")
    print("backend       min_ms     p25_ms  median_ms     p75_ms    samples/s       MV/s     params")
    medians = {}
    for name, values in samples.items():
        median = statistics.median(values)
        medians[name] = median
        samples_per_second = args.batch * 1000.0 / median
        multivectors_per_second = args.batch * args.items * args.out_channels * 1000.0 / median
        print(
            f"{name:<11} {min(values):>9.3f} {percentile(values, 0.25):>10.3f} "
            f"{median:>10.3f} {percentile(values, 0.75):>10.3f} "
            f"{samples_per_second:>12,.1f} {multivectors_per_second:>10,.0f} "
            f"{counts[name]:>10,}"
        )
    if set(medians) == {"reference", "clifra"}:
        ratio = medians["reference"] / medians["clifra"]
        print(f"speed ratio reference_median / clifra_median = {ratio:.3f}")
        print("> 1.0 = Clifra faster; < 1.0 = reference faster")


def main() -> None:
    args = parse_args()
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    if args.cpu_interop_threads is not None:
        torch.set_num_interop_threads(args.cpu_interop_threads)
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    configuration = dict(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        hidden_channels=args.hidden_channels,
        hidden_layers=args.hidden_layers,
        act_agg=args.act_agg,
    )
    reference = ReferenceGCAMLP(**configuration)
    clifra = ClifraGCAMLP(**configuration)
    adapter = ConventionAdapter(reference.pga, clifra.algebra)
    transfer_reference_to_clifra(reference, clifra, adapter)

    all_targets = {
        "reference": BenchmarkTarget(
            name="reference",
            model=reference.to(device)
        ),
        "clifra": BenchmarkTarget(
            name="clifra",
            model=clifra.to(device),
            to_model_domain=adapter.to_clifra,
            to_standard_domain=adapter.to_reference
        )
    }

    selected_names = ["clifra", "reference"] if args.backend == "both" else [args.backend]
    targets = [all_targets[name] for name in selected_names]
    counts = {t.name: parameter_count(t.model) for t in targets}
    if len(set(counts.values())) != 1:
        raise AssertionError(f"parameter counts differ: {counts}")

    standard_input = torch.randn(
        args.batch, args.items, args.in_channels, 16, device=device, dtype=torch.float32
    )

    # Precompute inputs for each target in their respective domain
    inputs = {
        target.name: target.to_model_domain(standard_input).detach().requires_grad_(True)
        for target in targets
    }

    print_provenance(args, device, counts)
    validate_correctness(
        targets,
        inputs,
        adapter,
        (args.batch, args.items, args.out_channels, 16),
    )

    modes = ["forward", "backward"] if args.mode == "both" else [args.mode]
    for mode in modes:
        measured_targets = targets
        if args.compile:
            measured_targets = compile_for_mode(targets, inputs, mode, device)
            if not measured_targets:
                print(f"No backend compiled successfully for {mode}; eager correctness remains valid.")
                continue
        samples = timed_samples(
            measured_targets,
            inputs,
            mode,
            args.warmup,
            args.iters,
            device,
        )
        report_samples(samples, args, counts, mode)


if __name__ == "__main__":
    main()
