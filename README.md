# Clifra GCA-MLP benchmark

This standalone repository asks one question: for one fixed geometric-algebra neural
network, how do an established implementation and a clean Clifra-native implementation
compare in correctness, source complexity, eager execution, compiled execution, forward
throughput, and forward+backward throughput?

## Model and provenance

The only model is **GCA-MLP** with `flatten=False`, PGA signature `(0, +, +, +)`, all 16
input blades, action blades `(0, 5, 6, 7, 8, 9, 10, 15)`, and the upstream sequence of
`PGAConjugateLinear` and `MultiVectorAct` layers. The default configuration is 8 input
channels, 8 output channels, 32 hidden channels, 3 hidden layers, and `act_agg="linear"`.
It has 24,776 trainable parameters in both implementations.

`reference.py` directly uses Microsoft's official `cliffordlayers` primitives, following
Qualcomm's `gatr/baselines/gcan.py`. `clifra_model.py` derives the same conjugation maps
with Clifra public plans during construction. It lowers `k x reverse(k)` to 36 fixed
quadratic operator maps and performs only tensor contractions at runtime.

Pinned/tested provenance:

- Qualcomm GATr source commit: `6afc26f26b8fcf51136ae8c1d264a36e14b6e497`
- `cliffordlayers` commit used by GATr: `74799cf4588a065916305bfcf2f030d84918f0ad`
  (`0.1.3.dev7+g74799cf45`)
- local editable Clifra: `2.0.1`, commit
  `07b811792adfc18de43d06b844172a2bb7176120`
- Python `3.12.13`, PyTorch `2.14.0`, NumPy `2.5.3`

The editable Clifra source is expected at `../clifra`, as recorded in `pyproject.toml` and
`uv.lock`.

## Install and run

```bash
uv sync
uv run benchmark.py --backend both --device cpu
uv run benchmark.py --backend both --device mps
uv run benchmark.py --backend both --device mps --compile
```

The CLI also exposes batch/items, all channel counts, hidden layers, activation
aggregation, warmup/iteration counts, forward/backward mode, seed, and explicit CPU
thread controls. Use `uv run benchmark.py --help` for all options.

## Methodology and correctness

Each invocation first checks exact convention roundtrip, shape, finite forward and
backward values, all trainable gradients, transferred layer equivalence, and transferred
whole-model equivalence. The reference null-first shortlex basis and Clifra's canonical
null-last bitmask basis differ by both permutation and orientation signs; conversion is
performed in `benchmark.py` before timing. The same conversion transfers actions and
activation weights. GCA-MLP is not claimed to be E(3)-equivariant: the upstream source
explicitly says it is not, so the harness does not assert that stronger property.

Timing uses `perf_counter_ns()`, synchronizes CUDA/MPS around every sample, alternates
backend order in warmup and measurement, uses inference mode for forward, and times
forward + mean-square scalar loss + backward for backward mode. Gradient reset, model
construction, imports, correctness checks, Clifra planning/static lowering, convention
conversion, compilation, and compile first calls are excluded. `samples/s` counts batch
samples; `MV/s` counts output multivectors (`batch * items * out_channels`). The reported
ratio is `reference_median / clifra_median`: **> 1 means Clifra is faster; < 1 means the
reference is faster**.

## Observed results

Apple arm64, macOS 26.6.2, float32, PyTorch intra/inter-op threads 5/15. Forward used 3
warmups + 10 iterations on CPU and 5 + 15 on MPS. Values are median milliseconds.

| Device | Batch×items | Reference | Clifra | Ref/Clifra |
|---|---:|---:|---:|---:|
| CPU | 1×16 | 13.186 | 7.343 | 1.796 |
| CPU | 1×64 | 20.458 | 13.472 | 1.519 |
| CPU | 1×256 | 37.106 | 30.543 | 1.215 |
| CPU | 8×16 | 25.212 | 19.233 | 1.311 |
| CPU | 8×64 | 56.985 | 50.057 | 1.138 |
| CPU | 8×256 | 207.454 | 201.615 | 1.029 |
| CPU | 32×16 | 58.205 | 50.990 | 1.141 |
| CPU | 32×64 | 201.733 | 193.020 | 1.045 |
| CPU | 32×256 | 826.077 | 796.551 | 1.037 |
| MPS | 1×16 | 4.797 | 0.604 | 7.940 |
| MPS | 1×64 | 5.080 | 0.696 | 7.294 |
| MPS | 1×256 | 5.899 | 0.896 | 6.586 |
| MPS | 8×16 | 5.273 | 0.706 | 7.471 |
| MPS | 8×64 | 6.585 | 1.347 | 4.887 |
| MPS | 8×256 | 11.763 | 3.540 | 3.323 |
| MPS | 32×16 | 6.561 | 1.308 | 5.018 |
| MPS | 32×64 | 11.712 | 3.578 | 3.274 |
| MPS | 32×256 | 37.329 | 12.933 | 2.886 |

Forward+backward used the same default model at 64 items:

| Device | Batch×items | Reference ms | Clifra ms | Ref/Clifra |
|---|---:|---:|---:|---:|
| CPU | 1×64 | 108.458 | 93.473 | 1.160 |
| CPU | 8×64 | 712.212 | 694.166 | 1.026 |
| CPU | 32×64 | 2770.980 | 2759.909 | 1.004 |
| MPS | 1×64 | 12.236 | 8.437 | 1.450 |
| MPS | 8×64 | 12.028 | 7.514 | 1.601 |
| MPS | 32×64 | 24.735 | 11.366 | 2.176 |

Selected `torch.compile` result at 8×64 (compilation and first calls excluded):

| Device/mode | Reference ms | Clifra ms | Ref/Clifra |
|---|---:|---:|---:|
| CPU forward | 54.769 | 50.682 | 1.081 |
| CPU forward+backward | 703.123 | 684.986 | 1.026 |
| MPS forward | 1.446 | 1.111 | 1.302 |
| MPS forward+backward | 2.989 | 2.449 | 1.221 |

The main eager difference is structural rather than a generic algebra-runtime cost. The
reference layer reconstructs full left/right Clifford kernels and embeds/gathers full
multivectors on every call. Clifra plans once, stores a 36-map conjugation basis, and
forms one dense runtime kernel. A layer microbenchmark on MPS found the hidden linear
2.5–3.2× faster and activation 1.7–4.3× faster over 16–8192 folded items. The CPU gap
narrows as tensor work dominates; compilation substantially narrows the MPS gap by
fusing away much of the reference materialization overhead.

## Reproduce the main sweep

```bash
for d in cpu mps; do
  for b in 1 8 32; do
    for n in 16 64 256; do
      uv run benchmark.py --device "$d" --batch "$b" --items "$n" \
        --mode forward --warmup 5 --iters 15
    done
  done
  for b in 1 8 32; do
    uv run benchmark.py --device "$d" --batch "$b" --items 64 \
      --mode backward --warmup 5 --iters 15
  done
done
```
