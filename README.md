# Clifra model benchmarks

[Clifra](https://github.com/Concode0/clifra) is a general Clifford-algebra computation layer for PyTorch rather than a neural-network library. I wanted to see how far that abstraction carries when used to reproduce published geometric-algebra networks.

This repository is one such experiment. It implements seven benchmark targets spanning GCA, CliffordLayers, CGENN, and GATr with both a reference implementation and a Clifra-native implementation, aligns their conventions and parameters, checks numerical equivalence, and then compares their ordinary PyTorch execution on CPU, MPS, and CUDA.

The goal is broader than asking whether Clifra is faster. These ports test whether established model structures can be expressed faithfully on top of Clifra's general computation layer, and whether doing so introduces meaningful execution costs or opportunities.

Before timing begins, every comparison checks basis roundtrips, parameter-count parity, finite outputs and gradients, model-specific equivalence, and compiled-versus-eager agreement.

The timed region is the steady-state

```python
model(*args, **kwargs)
```

call, plus a scalar loss and backward pass in forward+backward mode. Model construction, representation conversion, parameter transfer, compilation, correctness checks, and first calls are excluded from timing.

These are execution comparisons of matched fixed architectures, not task-performance or training-quality benchmarks.

## Models tested

| Model               | Reference                                                                                                                                                          | Scope                                                                         |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| `gca_mlp`           | [Qualcomm GATr `gcan.py`](https://github.com/qualcomm-ai-research/geometric-algebra-transformer/blob/6afc26f/gatr/baselines/gcan.py)                               | GCA-MLP with per-item channels; `--flatten` matches the n-body channel layout |
| `gca_gnn`           | same                                                                                                                                                               | GCA-GNN message passing over PyG `MessagePassing`                             |
| `clifford_resnet2d` | [CliffordLayers `74799cf`](https://github.com/microsoft/cliffordlayers/tree/74799cf) · [Brandstetter et al.](https://arxiv.org/abs/2209.04934)                     | 2D scalar/vector-field Clifford ResNet                                        |
| `clifford_fno2d`    | same                                                                                                                                                               | 2D Clifford Fourier neural operator                                           |
| `clifford_fno3d`    | same                                                                                                                                                               | 3D vector/bivector Maxwell Clifford Fourier model                             |
| `cgenn`             | [CGENN `08f2f2c`](https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks/tree/08f2f2c) · [Ruhe et al.](https://arxiv.org/abs/2305.11141)          | Five variants: O(3), O(5) regression, convex hulls, n-body, and Lorentz       |
| `gatr`              | [Qualcomm GATr `6afc26f`](https://github.com/qualcomm-ai-research/geometric-algebra-transformer/tree/6afc26f) · [Brehmer et al.](https://arxiv.org/abs/2305.18415) | Geometric Algebra Transformer with PGA attention                              |

The reference side follows pinned upstream implementations or source revisions. Where a direct package path is impractical, the repository keeps an execution-equivalent port of the pinned source and records that choice in the benchmark provenance.

## Results summary

The table below reports **reference median / Clifra median** latency ratios on each model's default workload.

Values above 1 favor Clifra; values below 1 favor the reference. Each cell is **forward / forward+backward**.

A `—` means that a valid paired timing was unavailable because one side failed compilation or validation for that cell.

| Model                 |   CPU eager | CPU compiled |  CUDA eager | CUDA compiled |   MPS eager | MPS compiled |
| --------------------- | ----------: | -----------: | ----------: | ------------: | ----------: | -----------: |
| `gca_mlp`             |  32× / 131× |   33× / 164× | 3.1× / 1.7× |   1.8× / 1.4× | 8.3× / 2.1× |  2.3× / 2.2× |
| `gca_gnn`             |   13× / 33× |    11× / 36× | 3.0× / 1.7× |   1.8× / 1.5× | 3.8× / 1.3× |  1.6× / 1.8× |
| `clifford_resnet2d`   | 1.1× / 1.2× |     — / 1.0× | 1.3× / 1.3× |         — / — | 1.1× / 1.2× |        — / — |
| `clifford_fno2d`      | 1.2× / 1.5× |  1.4× / 1.1× | 1.4× / 1.3× |   0.9× / 1.0× | 1.3× / 1.2× |  1.2× / 1.0× |
| `clifford_fno3d`      | 1.1× / 1.7× |  1.8× / 1.3× | 3.6× / 2.8× |         — / — | 2.0× / 1.6× |        — / — |
| `cgenn/o3`            | 4.2× / 2.7× |  1.5× / 1.2× | 2.3× / 1.7× |   1.2× / 0.9× | 2.0× / 1.2× |  1.6× / 1.0× |
| `cgenn/o5_regression` | 2.2× / 1.4× |  1.4× / 0.9× | 2.4× / 1.8× |   1.4× / 1.0× | 2.1× / 3.6× |  1.8× / 1.2× |
| `cgenn/hulls`         | 2.5× / 1.7× |  1.5× / 1.0× | 2.6× / 2.0× |   1.5× / 1.0× | 2.2× / 2.6× |  1.9× / 1.1× |
| `cgenn/nbody`         | 3.0× / 2.1× |  1.6× / 1.3× | 2.3× / 1.7× |   1.3× / 1.0× | 3.9× / 1.8× |  2.0× / 1.3× |
| `cgenn/lorentz`       | 2.1× / 1.8× |  1.8× / 1.4× | 2.0× / 1.5× |   1.4× / 1.1× | 4.3× / 2.8× |  1.8× / 2.0× |
| `gatr`                | 0.6× / 0.9× |  1.5× / 1.7× | 0.8× / 1.1× |   1.6× / 1.6× | 1.8× / 2.0× |     1.8× / — |

The default configurations are intentionally small, correctness-oriented workloads rather than reproductions of full paper training runs. Scaled workloads are included in the full reports.

Large ratios in some GCA cases reflect differences in how the reference implementation and Clifra execute the same fixed algebraic structure. They should not be interpreted as a general claim that Clifra is uniformly faster across geometric-algebra workloads.

Full latency tables, distributions, architecture settings, provenance, and compiler diagnostics are recorded in the per-device reports:

* [CPU results](results/2026-09-20/cpu.md) — Apple M5 Pro, 4 intra-op / 4 inter-op threads
* [CUDA results](results/2026-09-20/cuda.md) — NVIDIA RTX PRO 4000
* [MPS results](results/2026-09-20/mps.md) — Apple M5 Pro MPS

Raw benchmark output is preserved under [`results/2026-09-20/logs/`](results/2026-09-20/logs/).

### Notes

* **Clifford ResNet 2D compiled:** compiled-output validation failures occur for several backend/mode combinations, so compiled timings are reported only where a validated pair is available.

* **Clifford FNO 3D compiled reference:** the reference compiler path fails on CUDA with a driver error and on MPS with a device mismatch. Clifra produces compiled timings on those backends, but no paired ratio is reported.

* **GATr eager forward:** the reference is faster than Clifra on CPU and CUDA in eager forward execution. After `torch.compile`, Clifra is faster in these measurements and captures as 1 graph / 0 breaks, compared with 10 graphs / 5 breaks for the reference.

* **GATr MPS compiled forward+backward:** the reference compiled backward fails gradient validation, so only the Clifra timing is available.

## Running it yourself

```bash
uv sync --locked

uv run --locked benchmark.py --model gca_mlp --device auto
uv run --locked benchmark.py --model cgenn --variant o3 --device cpu
uv run --locked benchmark.py --model gatr --device cuda --compile
```

`--device` accepts `cpu`, `mps`, `cuda`, and `auto`, with `auto` trying CUDA → MPS → CPU.
`--mode forward` measures inference only. `--mode forward+backward` measures a new grad-enabled forward, scalar loss, and backward pass.
`--compile` additionally measures the validated `torch.compile` steady state.

Run `--help` for model-specific architecture and workload options.

## Repository layout

Each comparison lives under `models/<name>/`:

```text
models/<name>/
├── reference.py
├── clifra.py
└── benchmark.py
```

The reference and Clifra implementations remain separate. Configuration, convention mapping, parameter alignment, and correctness checks live in the benchmark module.

The shared runner receives native models and native inputs, keeping graph construction, representation conversion, and other setup work outside the measured call.

Adding another model comparison requires implementing that three-file structure and registering it in the root `benchmark.py`.
