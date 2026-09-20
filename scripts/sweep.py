"""Run the fixed paired-model workload sweep and retain every raw benchmark log."""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    name: str
    default: tuple[str, ...]
    scaled: tuple[str, ...]


CASES = (
    Case(
        "gca_mlp",
        ("--model", "gca_mlp"),
        ("--batch", "8", "--items", "128", "--hidden-channels", "64"),
    ),
    Case(
        "gca_gnn",
        ("--model", "gca_gnn"),
        (
            "--graphs",
            "2",
            "--nodes",
            "32",
            "--edges",
            "128",
            "--node-channels",
            "16",
            "--message-channels",
            "16",
            "--mlp-hidden-channels",
            "32",
        ),
    ),
    Case(
        "clifford_resnet2d",
        ("--model", "clifford_resnet2d"),
        ("--grid", "48", "--hidden-channels", "8", "--blocks", "3"),
    ),
    Case(
        "clifford_fno2d",
        ("--model", "clifford_fno2d"),
        ("--grid", "48", "--hidden-channels", "8", "--blocks", "3", "--modes", "16"),
    ),
    Case(
        "clifford_fno3d",
        ("--model", "clifford_fno3d"),
        ("--grid", "20", "--hidden-channels", "4", "--blocks", "3", "--modes", "8"),
    ),
    Case(
        "cgenn/o3",
        ("--model", "cgenn", "--variant", "o3"),
        ("--batch", "4", "--hidden-features", "32", "--layers", "4"),
    ),
    Case(
        "cgenn/o5_regression", ("--model", "cgenn", "--variant", "o5_regression"), ("--batch", "4")
    ),
    Case(
        "cgenn/hulls",
        ("--model", "cgenn", "--variant", "hulls"),
        ("--batch", "4", "--hidden-features", "16", "--layers", "4"),
    ),
    Case(
        "cgenn/nbody",
        ("--model", "cgenn", "--variant", "nbody"),
        ("--batch", "3", "--nodes", "6", "--hidden-features", "16", "--layers", "3"),
    ),
    Case(
        "cgenn/lorentz",
        ("--model", "cgenn", "--variant", "lorentz"),
        (
            "--batch",
            "3",
            "--nodes",
            "6",
            "--hidden-features",
            "16",
            "--hidden-scalar-features",
            "8",
            "--decoder-features",
            "16",
            "--layers",
            "3",
        ),
    ),
    Case(
        "gatr",
        ("--model", "gatr"),
        (
            "--items",
            "16",
            "--hidden-mv-channels",
            "8",
            "--hidden-s-channels",
            "32",
            "--blocks",
            "3",
            "--heads",
            "4",
        ),
    ),
)


def _capture(command: list[str]) -> str:
    return subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _tables(log: str) -> dict[str, dict[str, str]]:
    """Extract the runner's reported medians without recomputing benchmark results."""
    tables: dict[str, dict[str, str]] = {}
    phase = None
    for line in log.splitlines():
        match = re.fullmatch(r"=== (forward|forward\+backward) \((eager|compiled)\) ===", line)
        if match:
            phase = match.group(1)
            tables[phase] = {}
            continue
        if phase and (match := re.match(r"^(reference|clifra)\s+\S+\s+\S+\s+(\d+\.\d+)", line)):
            tables[phase][match.group(1)] = match.group(2)
        elif phase and (
            match := re.match(r"^speed ratio reference_median / clifra_median = (\d+\.\d+)", line)
        ):
            tables[phase]["ratio"] = match.group(1)
    return tables


def _compile_info(log: str, phase: str) -> tuple[str, list[str]]:
    section = log.split(f"=== torch.compile preparation: {phase} ===", 1)
    if len(section) != 2:
        return "—", []
    body = section[1].split("\n=== ", 1)[0]
    captures = []
    failures = []
    for line in body.splitlines():
        if match := re.match(
            r"^(reference|clifra): .*check=PASS, graphs=(\d+), graph_breaks=(\d+), recompile_limit_warning=(\w+)",
            line,
        ):
            captures.append(
                f"{match[1]} {match[2]}g/{match[3]}b" + (" ⚠" if match[4] == "True" else "")
            )
        elif match := re.match(r"^(reference|clifra): COMPILE FAILED: (.*)", line):
            failures.append(f"{phase} {match[1]}: {match[2]}")
    return "; ".join(captures) or "—", failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--cpu-interop-threads", type=int, default=4)
    parser.add_argument(
        "--summarize-only", action="store_true", help="rebuild the document from existing raw logs"
    )
    args = parser.parse_args()
    if min(args.iters, args.cpu_threads, args.cpu_interop_threads) <= 0 or args.warmup < 0:
        parser.error("iterations and thread counts must be positive; warmup must be nonnegative")

    output = args.output_dir.resolve()
    logs = output / "logs" / args.device
    logs.mkdir(parents=True, exist_ok=True)
    document = output / f"{args.device}.md"
    if args.summarize_only:
        if not document.exists():
            parser.error(f"missing existing summary: {document}")
        previous = document.read_text().splitlines()
        started, source = previous[2:4]
    else:
        commit = _capture(["git", "rev-parse", "HEAD"])
        started = f"Run started: {datetime.now(timezone.utc).isoformat(timespec='seconds')}."
        source = f"Source commit: `{commit}`. Host: `{platform.platform()}`."
    lines = [
        f"# {args.device.upper()} paired-model sweep",
        "",
        started,
        source,
        f"Device: `{args.device}`; float32; seed 0; CPU intra-op/inter-op threads: {args.cpu_threads}/{args.cpu_interop_threads}.",
        f"Each command uses `uv run --locked benchmark.py`, {args.warmup} warmup and {args.iters} timed iterations per mode.",
        "The default is the repository's small correctness workload. The scaled setting changes the listed architecture or workload flags.",
        "Latency is median milliseconds. Ratio is reference median / Clifra median; greater than 1 favors Clifra.",
        "Forward uses inference mode; forward+backward includes a fresh forward, loss, and backward.",
        "Compiled latency is shown only after output validation, plus gradient validation for forward+backward. Graph notation is graphs/breaks; ⚠ marks a recompile-limit warning.",
        "These are single-host sweep measurements, not cross-device speed ratios. Raw logs retain full provenance, correctness output, latency distributions, and throughput.",
        "",
    ]
    total = len(CASES) * 4
    index = 0
    for case in CASES:
        lines.extend((f"## {case.name}", ""))
        lines.append(
            f"Default flags: `{' '.join(case.default)}`. Scaled additions: `{' '.join(case.scaled)}`."
        )
        lines.extend(
            (
                "",
                "| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |",
                "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
            )
        )
        notes = []
        for workload, extra in (("default", ()), ("scaled", case.scaled)):
            for execution in ("eager", "compiled"):
                index += 1
                name = case.name.replace("/", "_")
                path = logs / f"{name}--{workload}--{execution}.log"
                command = [
                    "uv",
                    "run",
                    "--locked",
                    "benchmark.py",
                    *case.default,
                    *extra,
                    "--device",
                    args.device,
                    "--backend",
                    "both",
                    "--mode",
                    "both",
                    "--warmup",
                    str(args.warmup),
                    "--iters",
                    str(args.iters),
                    "--cpu-threads",
                    str(args.cpu_threads),
                    "--cpu-interop-threads",
                    str(args.cpu_interop_threads),
                ]
                if execution == "compiled":
                    command.append("--compile")
                print(
                    f"[{index}/{total}] {args.device} {case.name} {workload} {execution}",
                    flush=True,
                )
                if args.summarize_only:
                    log = path.read_text()
                    exit_match = re.search(r"\nexit_code=(\d+)\s*$", log)
                    if exit_match is None:
                        raise ValueError(f"missing exit code in {path}")
                    returncode = int(exit_match.group(1))
                else:
                    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                    returncode = result.returncode
                    log = "$ " + " ".join(command) + "\n\n" + result.stdout + result.stderr
                    path.write_text(log + f"\nexit_code={returncode}\n")
                reported = _tables(log)
                for phase in ("forward", "forward+backward"):
                    medians = reported.get(phase, {})
                    reference = medians.get("reference", "—")
                    clifra = medians.get("clifra", "—")
                    ratio = medians.get("ratio", "—") if reference != "—" and clifra != "—" else "—"
                    capture, failures = (
                        _compile_info(log, phase) if execution == "compiled" else ("—", [])
                    )
                    notes.extend(f"{workload} {failure}" for failure in failures)
                    if returncode != 0:
                        notes.append(
                            f"{workload} {execution}: process exited {returncode}; inspect raw log"
                        )
                    link = path.relative_to(output).as_posix()
                    lines.append(
                        f"| {workload} | {execution} | {phase} | {reference} | {clifra} | {ratio} | {capture} | [log]({link}) |"
                    )
        lines.append("")
        if notes:
            lines.append("Validation and compiler notes:")
            lines.extend(f"- {note}" for note in dict.fromkeys(notes))
            lines.append("")
        document.write_text("\n".join(lines) + "\n")
    print(f"Wrote {document}", flush=True)


if __name__ == "__main__":
    main()
