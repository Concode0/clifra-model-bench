# Clifra model benchmarks

This standalone repository asks, for each fixed geometric-algebra neural network,
how an established implementation and a Clifra-native implementation
compare in correctness, source complexity, eager execution, compiled execution, forward
throughput, and forward+backward throughput?

## Benchmarks

| Model | Comparison |
| --- | --- |
| `gca_mlp` | GCA-MLP with per-item channels by default; `--flatten` reproduces the official n-body channel layout. |
| `gca_gnn` | GCA-GNN message passing with GCA-MLPs for messages and node updates. |
| `clifford_resnet2d` | 2D scalar/vector-field Clifford ResNet from *Clifford Neural Layers for PDE Modeling*. |
| `clifford_fno2d` | 2D Clifford Fourier neural operator variant from the same work. |
| `clifford_fno3d` | 3D vector/bivector Maxwell Clifford Fourier model from the same work. |
| `cgenn` | Five Clifford Group Equivariant Network variants spanning O(3), O(5), and O(1,3). |
| `gatr` | Full Geometric Algebra Transformer with PGA attention and geometric MLP blocks. |

The GCA reference models follow [Qualcomm GATr's pinned `gcan.py`](https://github.com/qualcomm-ai-research/geometric-algebra-transformer/blob/6afc26f26b8fcf51136ae8c1d264a36e14b6e497/gatr/baselines/gcan.py)
and use [pinned CliffordLayers](https://github.com/microsoft/cliffordlayers/tree/74799cf4588a065916305bfcf2f030d84918f0ad).
The official [GCA-MLP](https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/blob/6afc26f26b8fcf51136ae8c1d264a36e14b6e497/config/model/gcamlp_nbody.yaml)
and [GCA-GNN](https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/blob/6afc26f26b8fcf51136ae8c1d264a36e14b6e497/config/model/gcagnn_nbody.yaml)
n-body configs use larger widths and, for GCA-MLP, flatten four items into one
channel axis. The benchmark CLI can express these settings.
GCA-GNN follows the [original GCAN work](https://proceedings.mlr.press/v202/ruhe23a/ruhe23a.pdf)
and uses PyTorch Geometric `MessagePassing`. Its default workload is two disjoint graphs
with 16 nodes and 64 directed edges each, 4 input/output channels, 8 node/message
channels, 16 MLP hidden channels, two MLP hidden layers, and three message-passing steps.
Its edges are unique and loop-free. The official n-body wrapper uses a complete
loop-free graph; set `--edges` to `nodes*(nodes-1)` to match that topology.
The PDE references use the [official CliffordLayers models at the same pinned revision](https://github.com/microsoft/cliffordlayers/tree/74799cf4588a065916305bfcf2f030d84918f0ad),
from [Brandstetter et al.](https://arxiv.org/abs/2209.04934). Their fixed defaults use
signature `(2,0)`, one 32×32 scalar/vector field, four hidden multivector channels,
two blocks, and 16 modes per axis for the Fourier model.
The 2D ResNet uses Clifford group normalization; the 2D Fourier model does not.
On PyTorch 2.14 MPS, the pinned ResNet normalization's per-pixel triangular
solve can return zeros. Its MPS reference wrapper uses an equivalent batched
solve, verified against upstream outputs and gradients on CPU; the wrapper is
identified in benchmark provenance and remains inside timing.
The 3D Maxwell benchmark uses signature `(3,0)`, one 16³ vector/bivector field,
two hidden channels and Fourier blocks, eight modes per axis, and padding two.
The [pinned CliffordLayers model examples](https://github.com/microsoft/cliffordlayers/blob/74799cf4588a065916305bfcf2f030d84918f0ad/docs/reference/models.md)
use larger widths, four stages, and potentially different input/output channels;
`--in-channels`, `--out-channels`, `--hidden-channels`, `--blocks`, and Fourier
`--modes` allow those execution architectures. The defaults are small correctness
workloads, not paper training scales. The [paper's architecture appendix](https://arxiv.org/abs/2209.04934)
reports Clifford ResNet 2D at 8 blocks/64 hidden channels, Clifford FNO 2D at
8 blocks/48 hidden channels/16 modes, and Maxwell FNO 3D at 4 blocks/32 hidden
channels/6 modes.
`cgenn` follows the [official source at commit `08f2f2c`](https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks/tree/08f2f2cbf5f7a723ea8539daf9f20a24cde73b1b)
for [Ruhe et al.](https://arxiv.org/abs/2305.11141). Its reference code is an
execution-only port of that pinned revision, with the upstream algebra `b()` indentation repaired.
The [official README commands and model configs](https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks/tree/08f2f2cbf5f7a723ea8539daf9f20a24cde73b1b/configs/model)
give the full experiment widths and depths; this benchmark defaults to smaller widths.
`gatr` uses the [official Qualcomm package at commit `6afc26f`](https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/tree/6afc26f26b8fcf51136ae8c1d264a36e14b6e497)
for [Brehmer et al.](https://arxiv.org/abs/2305.18415). Its [n-body configuration](https://github.com/Qualcomm-AI-research/geometric-algebra-transformer/blob/6afc26f26b8fcf51136ae8c1d264a36e14b6e497/config/model/gatr_nbody.yaml)
uses 10 blocks, 16 hidden MV channels, and 128 hidden scalar channels. The pinned
commit records a 1.4.3 update in its changelog, while package `__version__` and
installed metadata remain 1.4.2.
Both GATr implementations use PyTorch SDPA; uv replaces the pinned package's
unconditional xFormers and NumPy `<1.25` dependency declarations.

## Run

```bash
uv sync --locked
uv run --locked benchmark.py --model gca_mlp --device auto
uv run --locked benchmark.py --model gca_gnn --device cpu
uv run --locked benchmark.py --model gca_gnn --device cuda --compile
uv run --locked benchmark.py --model clifford_resnet2d --device cpu
uv run --locked benchmark.py --model clifford_fno2d --device cpu --compile
uv run --locked benchmark.py --model clifford_fno3d --device cpu --compile
uv run --locked benchmark.py --model cgenn --variant o3 --device cpu
uv run --locked benchmark.py --model gatr --device cpu
```

`--device` accepts `cpu`, `mps`, `cuda`, and `auto` (CUDA, then MPS, then CPU).
Explicit unavailable devices fail. `--help` lists model configuration, seed, warmup,
iteration, mode, and CPU thread options. `--mode forward` times inference;
`--mode forward+backward` times a new grad-enabled forward, scalar loss, and backward.

## Method

The benchmark checks basis roundtrips, parameter counts, finite outputs and gradients,
model-specific equivalence, and compiled versus eager outputs and gradients before
timing. Graphs, native inputs, parameters, and Clifra plans are prepared first. The
steady-state timed call is `model(*args, **kwargs)`, plus the model's scalar loss and
backward pass in forward+backward mode. Timing excludes construction, conversion, parameter
transfer, correctness checks, compilation, first calls, and gradient reset. CPU uses
wall-clock time; MPS and CUDA use the same clock with device synchronization around
each sample. Output includes latency distributions, model-specific work rates, and
software, device, and source provenance. Compiled preparation also reports captured
graph and graph-break counts for each implementation.
If compilation fails or compiled correctness differs, that backend has no
compiled timing result. Current PyTorch 2.14 MPS limitations include the ResNet
triangular-solve compiler path, the official FNO3D reference compiler path, and
the GATr reference's compiled forward+backward validation.

## Layout

Each comparison lives under `models/<name>/` as `reference.py`, `clifra.py`, and
`benchmark.py`. I keep the two model implementations separate, with configuration,
convention mapping, parameter alignment, and correctness checks in the benchmark file.
The shared runner receives native models and inputs, so graph construction and
representation conversion stay outside the measured call. Adding a comparison needs
one entry in the root `benchmark.py` registry.
