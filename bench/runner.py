"""Architecture-independent eager and compiled benchmark execution."""

import contextlib
import logging
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import clifra
import torch

from .core import (
    PreparedBenchmark,
    PreparedCall,
    assert_close_tree,
    check_finite,
    detach_tree,
    parameter_count,
    tensors,
)


class _RecompileLimitTracker(logging.Handler):
    def __init__(self):
        super().__init__()
        self.hit_limit = False

    def emit(self, record: logging.LogRecord) -> None:
        if "recompile_limit" in record.getMessage():
            self.hit_limit = True


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
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


def _version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _commit(path: Path) -> str:
    if not (path / ".git").exists():
        return "not a git checkout"
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return commit + (" (dirty)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _source_commit(module_file: str) -> str:
    """Report a checkout only for an imported source tree, never a virtualenv."""
    source = Path(module_file).resolve()
    if "site-packages" in source.parts:
        return "installed package"
    for parent in source.parents:
        if (parent / ".git").exists():
            return _commit(parent)
    return "installed package"


def print_provenance(
    prepared: PreparedBenchmark, args, device: torch.device, counts: dict[str, int]
) -> None:
    repository = Path(__file__).resolve().parents[1]
    print("=== Provenance ===")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"PyTorch: {torch.__version__}")
    print(f"Clifra: {_version('clifra')}")
    clifra_path = Path(clifra.__file__).resolve()
    print(f"Clifra source: {clifra_path}")
    print(f"Clifra git commit: {_source_commit(str(clifra_path))}")
    for package in prepared.provenance_packages:
        print(f"{package}: {_version(package)}")
    if prepared.reference_source:
        print(f"reference source: {prepared.reference_source}")
    print(f"benchmark git commit: {_commit(repository)}")
    print(f"platform: {platform.platform()} / {platform.machine()}")
    if platform.system() == "Darwin":
        print(f"macOS: {platform.mac_ver()[0]}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(device)}")
        print(f"PyTorch CUDA build: {torch.version.cuda}")
        print(f"cuDNN: {torch.backends.cudnn.version()}")
    dtypes = {
        str(parameter.dtype) for call in prepared.calls for parameter in call.model.parameters()
    }
    print(f"parameter dtypes: {', '.join(sorted(dtypes))}")
    print(f"CPU intra-op threads: {torch.get_num_threads()}")
    print(f"CPU inter-op threads: {torch.get_num_interop_threads()}")
    thread_env = {
        name: os.environ[name]
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
        if name in os.environ
    }
    print(f"thread environment: {thread_env or 'unset'}")
    print(f"model: {prepared.description}")
    print("parameter counts: " + ", ".join(f"{name}={count:,}" for name, count in counts.items()))
    print(f"execution: {'compiled' if args.compile else 'eager'}, mode={args.mode}")


def validate(prepared: PreparedBenchmark) -> None:
    print("\n=== Correctness ===")
    outputs = {}
    for call in prepared.calls:
        outputs[call.name] = check_finite(call)
        print(f"{call.name}: finite forward, finite backward PASS")
    prepared.validate(outputs)


def _check_placement(call: PreparedCall, device: torch.device) -> None:
    values = [*call.model.parameters(), *call.model.buffers(), *tensors((call.args, call.kwargs))]
    for value in values:
        if value.device.type != device.type or (
            device.index is not None and value.device.index != device.index
        ):
            raise RuntimeError(f"{call.name}: prepared tensor on {value.device}, expected {device}")


def _execute(call: PreparedCall, mode: str):
    output = call.model(*call.args, **call.kwargs)
    if mode == "forward+backward":
        call.loss(output).backward()
    return output


def _gradients(call: PreparedCall) -> list[torch.Tensor | None]:
    values = [*tensors((call.args, call.kwargs)), *call.model.parameters()]
    return [value.grad.detach().clone() if value.grad is not None else None for value in values]


def compile_calls(calls: list[PreparedCall], mode: str, device: torch.device) -> list[PreparedCall]:
    compiled = []
    print(f"\n=== torch.compile preparation: {mode} ===")
    for call in calls:
        tracker = _RecompileLimitTracker()
        dynamo_logger = logging.getLogger("torch._dynamo")
        dynamo_logger.addHandler(tracker)
        try:
            from torch._dynamo.utils import counters

            # Dynamo counters are process global; isolate this backend's first call.
            counters.clear()
            start = time.perf_counter_ns()
            candidate = torch.compile(call.model)
            api_ms = (time.perf_counter_ns() - start) / 1e6
            call.reset_grad()
            if mode == "forward":
                with torch.inference_mode():
                    expected = detach_tree(call.model(*call.args, **call.kwargs))
                expected_gradients = None
            else:
                expected = detach_tree(_execute(call, mode))
                expected_gradients = _gradients(call)
                call.reset_grad()
            synchronize(device)
            start = time.perf_counter_ns()
            context = torch.inference_mode() if mode == "forward" else contextlib.nullcontext()
            compiled_call = replace(call, model=candidate)
            with context:
                actual = _execute(compiled_call, mode)
            synchronize(device)
            first_ms = (time.perf_counter_ns() - start) / 1e6
            assert_close_tree(detach_tree(actual), expected, rtol=1e-3, atol=1e-4)
            if expected_gradients is not None:
                assert_close_tree(
                    _gradients(compiled_call), expected_gradients, rtol=1e-3, atol=1e-4
                )
            compiled.append(compiled_call)
            checked = "output/gradients" if expected_gradients is not None else "output"
            graphs = counters["stats"]["unique_graphs"]
            breaks = sum(counters["graph_break"].values())
            print(
                f"{call.name}: compile API={api_ms:.3f} ms, first call={first_ms:.3f} ms, "
                f"{checked} check=PASS, graphs={graphs}, graph_breaks={breaks}, "
                f"recompile_limit_warning={tracker.hit_limit}"
            )
        except Exception as error:  # Compilation failures vary by device and backend.
            print(f"{call.name}: COMPILE FAILED: {type(error).__name__}: {error}")
        finally:
            dynamo_logger.removeHandler(tracker)
            call.reset_grad()
    return compiled


def timed_samples(
    calls: list[PreparedCall], mode: str, warmup: int, iterations: int, device: torch.device
) -> dict[str, list[float]]:
    samples = {call.name: [] for call in calls}
    context = torch.inference_mode() if mode == "forward" else contextlib.nullcontext()
    with context:
        for phase_iterations, record in ((warmup, False), (iterations, True)):
            for iteration in range(phase_iterations):
                order = calls if iteration % 2 == 0 else list(reversed(calls))
                for call in order:
                    if mode == "forward+backward":
                        call.reset_grad()
                    synchronize(device)
                    start = time.perf_counter_ns()
                    output = call.model(*call.args, **call.kwargs)
                    if mode == "forward+backward":
                        call.loss(output).backward()
                    synchronize(device)
                    if record:
                        samples[call.name].append((time.perf_counter_ns() - start) / 1e6)
    for call in calls:
        call.reset_grad()
    return samples


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return (
        ordered[lower]
        if lower == upper
        else ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    )


def report(
    samples: dict[str, list[float]],
    work_units: dict[str, float],
    counts: dict[str, int],
    mode: str,
    compiled: bool,
) -> None:
    print(f"\n=== {mode} ({'compiled' if compiled else 'eager'}) ===")
    units = list(work_units)
    print(
        "backend       min_ms     p25_ms  median_ms     p75_ms  "
        + " ".join(f"{unit + '/s':>12}" for unit in units)
        + "     params"
    )
    medians = {}
    for name, values in samples.items():
        median = statistics.median(values)
        medians[name] = median
        rates = " ".join(f"{amount * 1000 / median:>12,.1f}" for amount in work_units.values())
        print(
            f"{name:<11} {min(values):>9.3f} {_percentile(values, 0.25):>10.3f} {median:>10.3f} {_percentile(values, 0.75):>10.3f} {rates} {counts[name]:>10,}"
        )
    if {"reference", "clifra"} <= medians.keys():
        print(
            f"speed ratio reference_median / clifra_median = {medians['reference'] / medians['clifra']:.3f}"
        )
        print("> 1.0 = Clifra faster; < 1.0 = reference faster")


def run(prepared: PreparedBenchmark, args, device: torch.device) -> None:
    for call in prepared.calls:
        _check_placement(call, device)
    counts = {call.name: parameter_count(call.model) for call in prepared.calls}
    if len(set(counts.values())) != 1:
        raise AssertionError(f"parameter counts differ: {counts}")
    print_provenance(prepared, args, device, counts)
    validate(prepared)
    modes = ["forward", "forward+backward"] if args.mode == "both" else [args.mode]
    for mode in modes:
        calls = compile_calls(prepared.calls, mode, device) if args.compile else prepared.calls
        if not calls:
            print(f"No backend compiled successfully for {mode}; eager correctness remains valid.")
            continue
        samples = timed_samples(calls, mode, args.warmup, args.iters, device)
        report(samples, prepared.work_units, counts, mode, args.compile)
