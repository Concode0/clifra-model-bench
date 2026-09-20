# CUDA paired-model sweep

Run started: 2026-09-20T11:14:06+00:00.
Source commit: `38d8296c3fd39c77c23a3765ef112e703c8a6de7`. Host: `Linux-6.8.0-117-generic-x86_64-with-glibc2.35`.
Device: `cuda`; float32; seed 0; CPU intra-op/inter-op threads: 4/4.
Each command uses `uv run --locked benchmark.py`, 5 warmup and 20 timed iterations per mode.
The default is the repository's small correctness workload. The scaled setting changes the listed architecture or workload flags.
Latency is median milliseconds. Ratio is reference median / Clifra median; greater than 1 favors Clifra.
Forward uses inference mode; forward+backward includes a fresh forward, loss, and backward.
Compiled latency is shown only after output validation, plus gradient validation for forward+backward. Graph notation is graphs/breaks; ⚠ marks a recompile-limit warning.
These are single-host sweep measurements, not cross-device speed ratios. Raw logs retain full provenance, correctness output, latency distributions, and throughput.

## gca_mlp

Default flags: `--model gca_mlp`. Scaled additions: `--batch 8 --items 128 --hidden-channels 64`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 4.808 | 1.528 | 3.145 | — | [log](logs/cuda/gca_mlp--default--eager.log) |
| default | eager | forward+backward | 11.498 | 6.798 | 1.691 | — | [log](logs/cuda/gca_mlp--default--eager.log) |
| default | compiled | forward | 0.987 | 0.561 | 1.760 | clifra 1g/0b; reference 1g/0b | [log](logs/cuda/gca_mlp--default--compiled.log) |
| default | compiled | forward+backward | 3.121 | 2.155 | 1.448 | clifra 1g/0b; reference 1g/0b | [log](logs/cuda/gca_mlp--default--compiled.log) |
| scaled | eager | forward | 4.930 | 1.521 | 3.241 | — | [log](logs/cuda/gca_mlp--scaled--eager.log) |
| scaled | eager | forward+backward | 12.701 | 6.839 | 1.857 | — | [log](logs/cuda/gca_mlp--scaled--eager.log) |
| scaled | compiled | forward | 1.023 | 0.609 | 1.681 | clifra 1g/0b; reference 1g/0b | [log](logs/cuda/gca_mlp--scaled--compiled.log) |
| scaled | compiled | forward+backward | 3.759 | 2.253 | 1.669 | clifra 1g/0b; reference 1g/0b | [log](logs/cuda/gca_mlp--scaled--compiled.log) |

## gca_gnn

Default flags: `--model gca_gnn`. Scaled additions: `--graphs 2 --nodes 32 --edges 128 --node-channels 16 --message-channels 16 --mlp-hidden-channels 32`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 22.392 | 7.563 | 2.961 | — | [log](logs/cuda/gca_gnn--default--eager.log) |
| default | eager | forward+backward | 49.043 | 28.601 | 1.715 | — | [log](logs/cuda/gca_gnn--default--eager.log) |
| default | compiled | forward | 3.907 | 2.140 | 1.825 | clifra 1g/0b; reference 1g/0b | [log](logs/cuda/gca_gnn--default--compiled.log) |
| default | compiled | forward+backward | 11.310 | 7.575 | 1.493 | clifra 1g/0b; reference 1g/0b | [log](logs/cuda/gca_gnn--default--compiled.log) |
| scaled | eager | forward | 21.962 | 7.310 | 3.004 | — | [log](logs/cuda/gca_gnn--scaled--eager.log) |
| scaled | eager | forward+backward | 51.111 | 29.776 | 1.717 | — | [log](logs/cuda/gca_gnn--scaled--eager.log) |
| scaled | compiled | forward | 3.923 | 2.129 | 1.842 | clifra 1g/0b; reference 1g/0b | [log](logs/cuda/gca_gnn--scaled--compiled.log) |
| scaled | compiled | forward+backward | 11.461 | 7.521 | 1.524 | clifra 1g/0b; reference 1g/0b | [log](logs/cuda/gca_gnn--scaled--compiled.log) |

## clifford_resnet2d

Default flags: `--model clifford_resnet2d`. Scaled additions: `--grid 48 --hidden-channels 8 --blocks 3`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 4.671 | 3.530 | 1.323 | — | [log](logs/cuda/clifford_resnet2d--default--eager.log) |
| default | eager | forward+backward | 13.734 | 10.637 | 1.291 | — | [log](logs/cuda/clifford_resnet2d--default--eager.log) |
| default | compiled | forward | — | — | — | — | [log](logs/cuda/clifford_resnet2d--default--compiled.log) |
| default | compiled | forward+backward | — | — | — | — | [log](logs/cuda/clifford_resnet2d--default--compiled.log) |
| scaled | eager | forward | 6.406 | 4.895 | 1.309 | — | [log](logs/cuda/clifford_resnet2d--scaled--eager.log) |
| scaled | eager | forward+backward | 20.125 | 15.540 | 1.295 | — | [log](logs/cuda/clifford_resnet2d--scaled--eager.log) |
| scaled | compiled | forward | — | — | — | — | [log](logs/cuda/clifford_resnet2d--scaled--compiled.log) |
| scaled | compiled | forward+backward | — | — | — | — | [log](logs/cuda/clifford_resnet2d--scaled--compiled.log) |

Validation and compiler notes:
- default forward reference: AssertionError: Tensor-likes are not close!
- default forward clifra: AssertionError: Tensor-likes are not close!
- default forward+backward reference: AssertionError: Tensor-likes are not close!
- default forward+backward clifra: AssertionError: Tensor-likes are not close!
- scaled forward reference: AssertionError: Tensor-likes are not close!
- scaled forward clifra: AssertionError: Tensor-likes are not close!
- scaled forward+backward reference: AssertionError: Tensor-likes are not close!
- scaled forward+backward clifra: AssertionError: Tensor-likes are not close!

## clifford_fno2d

Default flags: `--model clifford_fno2d`. Scaled additions: `--grid 48 --hidden-channels 8 --blocks 3 --modes 16`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 3.217 | 2.298 | 1.400 | — | [log](logs/cuda/clifford_fno2d--default--eager.log) |
| default | eager | forward+backward | 12.193 | 9.296 | 1.312 | — | [log](logs/cuda/clifford_fno2d--default--eager.log) |
| default | compiled | forward | 1.333 | 1.415 | 0.942 | reference 1g/0b; clifra 1g/0b | [log](logs/cuda/clifford_fno2d--default--compiled.log) |
| default | compiled | forward+backward | 4.023 | 3.877 | 1.038 | reference 1g/0b; clifra 1g/0b | [log](logs/cuda/clifford_fno2d--default--compiled.log) |
| scaled | eager | forward | 4.481 | 3.264 | 1.373 | — | [log](logs/cuda/clifford_fno2d--scaled--eager.log) |
| scaled | eager | forward+backward | 17.157 | 13.181 | 1.302 | — | [log](logs/cuda/clifford_fno2d--scaled--eager.log) |
| scaled | compiled | forward | 1.877 | 1.982 | 0.947 | reference 1g/0b; clifra 1g/0b | [log](logs/cuda/clifford_fno2d--scaled--compiled.log) |
| scaled | compiled | forward+backward | 5.194 | 5.084 | 1.022 | reference 1g/0b; clifra 1g/0b | [log](logs/cuda/clifford_fno2d--scaled--compiled.log) |

## clifford_fno3d

Default flags: `--model clifford_fno3d`. Scaled additions: `--grid 20 --hidden-channels 4 --blocks 3 --modes 8`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 12.442 | 3.435 | 3.622 | — | [log](logs/cuda/clifford_fno3d--default--eager.log) |
| default | eager | forward+backward | 39.383 | 14.203 | 2.773 | — | [log](logs/cuda/clifford_fno3d--default--eager.log) |
| default | compiled | forward | — | 2.448 | — | clifra 1g/0b | [log](logs/cuda/clifford_fno3d--default--compiled.log) |
| default | compiled | forward+backward | — | 5.285 | — | clifra 1g/0b | [log](logs/cuda/clifford_fno3d--default--compiled.log) |
| scaled | eager | forward | 16.870 | 4.729 | 3.567 | — | [log](logs/cuda/clifford_fno3d--scaled--eager.log) |
| scaled | eager | forward+backward | 55.345 | 20.358 | 2.719 | — | [log](logs/cuda/clifford_fno3d--scaled--eager.log) |
| scaled | compiled | forward | — | 2.869 | — | clifra 1g/0b | [log](logs/cuda/clifford_fno3d--scaled--compiled.log) |
| scaled | compiled | forward+backward | — | 6.971 | — | clifra 1g/0b | [log](logs/cuda/clifford_fno3d--scaled--compiled.log) |

Validation and compiler notes:
- default forward reference: RuntimeError: CUDA driver error: invalid argument
- default forward+backward reference: RuntimeError: CUDA driver error: invalid argument
- scaled forward reference: RuntimeError: CUDA driver error: invalid argument
- scaled forward+backward reference: RuntimeError: CUDA driver error: invalid argument

## cgenn/o3

Default flags: `--model cgenn --variant o3`. Scaled additions: `--batch 4 --hidden-features 32 --layers 4`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 7.729 | 3.325 | 2.325 | — | [log](logs/cuda/cgenn_o3--default--eager.log) |
| default | eager | forward+backward | 18.432 | 10.810 | 1.705 | — | [log](logs/cuda/cgenn_o3--default--eager.log) |
| default | compiled | forward | 3.506 | 2.937 | 1.194 | reference 17g/17b; clifra 15g/7b | [log](logs/cuda/cgenn_o3--default--compiled.log) |
| default | compiled | forward+backward | 8.675 | 9.383 | 0.924 | reference 15g/14b; clifra 13g/6b | [log](logs/cuda/cgenn_o3--default--compiled.log) |
| scaled | eager | forward | 8.967 | 3.861 | 2.323 | — | [log](logs/cuda/cgenn_o3--scaled--eager.log) |
| scaled | eager | forward+backward | 21.389 | 12.745 | 1.678 | — | [log](logs/cuda/cgenn_o3--scaled--eager.log) |
| scaled | compiled | forward | 3.596 | 3.436 | 1.047 | reference 17g/17b; clifra 15g/7b | [log](logs/cuda/cgenn_o3--scaled--compiled.log) |
| scaled | compiled | forward+backward | 9.021 | 11.112 | 0.812 | reference 15g/14b; clifra 13g/6b | [log](logs/cuda/cgenn_o3--scaled--compiled.log) |

## cgenn/o5_regression

Default flags: `--model cgenn --variant o5_regression`. Scaled additions: `--batch 4`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 4.033 | 1.705 | 2.366 | — | [log](logs/cuda/cgenn_o5_regression--default--eager.log) |
| default | eager | forward+backward | 10.027 | 5.576 | 1.798 | — | [log](logs/cuda/cgenn_o5_regression--default--eager.log) |
| default | compiled | forward | 1.880 | 1.366 | 1.376 | reference 8g/7b; clifra 7g/3b | [log](logs/cuda/cgenn_o5_regression--default--compiled.log) |
| default | compiled | forward+backward | 4.552 | 4.451 | 1.023 | reference 8g/7b; clifra 7g/3b | [log](logs/cuda/cgenn_o5_regression--default--compiled.log) |
| scaled | eager | forward | 3.953 | 1.658 | 2.384 | — | [log](logs/cuda/cgenn_o5_regression--scaled--eager.log) |
| scaled | eager | forward+backward | 9.981 | 5.549 | 1.798 | — | [log](logs/cuda/cgenn_o5_regression--scaled--eager.log) |
| scaled | compiled | forward | 1.852 | 1.296 | 1.429 | reference 8g/7b; clifra 7g/3b | [log](logs/cuda/cgenn_o5_regression--scaled--compiled.log) |
| scaled | compiled | forward+backward | 4.729 | 4.517 | 1.047 | reference 8g/7b; clifra 7g/3b | [log](logs/cuda/cgenn_o5_regression--scaled--compiled.log) |

## cgenn/hulls

Default flags: `--model cgenn --variant hulls`. Scaled additions: `--batch 4 --hidden-features 16 --layers 4`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 14.692 | 5.647 | 2.602 | — | [log](logs/cuda/cgenn_hulls--default--eager.log) |
| default | eager | forward+backward | 35.519 | 17.975 | 1.976 | — | [log](logs/cuda/cgenn_hulls--default--eager.log) |
| default | compiled | forward | 6.192 | 4.201 | 1.474 | reference 11g/9b; clifra 9g/4b | [log](logs/cuda/cgenn_hulls--default--compiled.log) |
| default | compiled | forward+backward | 14.386 | 13.833 | 1.040 | reference 11g/8b; clifra 9g/4b | [log](logs/cuda/cgenn_hulls--default--compiled.log) |
| scaled | eager | forward | 14.585 | 5.631 | 2.590 | — | [log](logs/cuda/cgenn_hulls--scaled--eager.log) |
| scaled | eager | forward+backward | 35.551 | 17.947 | 1.981 | — | [log](logs/cuda/cgenn_hulls--scaled--eager.log) |
| scaled | compiled | forward | 6.598 | 4.416 | 1.494 | reference 11g/9b; clifra 9g/4b | [log](logs/cuda/cgenn_hulls--scaled--compiled.log) |
| scaled | compiled | forward+backward | 14.678 | 13.932 | 1.054 | reference 11g/8b; clifra 9g/4b | [log](logs/cuda/cgenn_hulls--scaled--compiled.log) |

## cgenn/nbody

Default flags: `--model cgenn --variant nbody`. Scaled additions: `--batch 3 --nodes 6 --hidden-features 16 --layers 3`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 38.123 | 16.450 | 2.317 | — | [log](logs/cuda/cgenn_nbody--default--eager.log) |
| default | eager | forward+backward | 91.445 | 52.805 | 1.732 | — | [log](logs/cuda/cgenn_nbody--default--eager.log) |
| default | compiled | forward | 18.009 | 14.404 | 1.250 | reference 26g/18b; clifra 23g/7b | [log](logs/cuda/cgenn_nbody--default--compiled.log) |
| default | compiled | forward+backward | 45.951 | 44.751 | 1.027 | reference 15g/13b ⚠; clifra 15g/5b ⚠ | [log](logs/cuda/cgenn_nbody--default--compiled.log) |
| scaled | eager | forward | 57.596 | 24.918 | 2.311 | — | [log](logs/cuda/cgenn_nbody--scaled--eager.log) |
| scaled | eager | forward+backward | 138.365 | 79.536 | 1.740 | — | [log](logs/cuda/cgenn_nbody--scaled--eager.log) |
| scaled | compiled | forward | 27.101 | 22.228 | 1.219 | reference 26g/18b; clifra 23g/7b | [log](logs/cuda/cgenn_nbody--scaled--compiled.log) |
| scaled | compiled | forward+backward | 67.342 | 66.330 | 1.015 | reference 15g/13b ⚠; clifra 15g/5b ⚠ | [log](logs/cuda/cgenn_nbody--scaled--compiled.log) |

## cgenn/lorentz

Default flags: `--model cgenn --variant lorentz`. Scaled additions: `--batch 3 --nodes 6 --hidden-features 16 --hidden-scalar-features 8 --decoder-features 16 --layers 3`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 15.998 | 7.895 | 2.026 | — | [log](logs/cuda/cgenn_lorentz--default--eager.log) |
| default | eager | forward+backward | 40.901 | 26.520 | 1.542 | — | [log](logs/cuda/cgenn_lorentz--default--eager.log) |
| default | compiled | forward | 7.831 | 5.655 | 1.385 | reference 20g/17b; clifra 14g/7b | [log](logs/cuda/cgenn_lorentz--default--compiled.log) |
| default | compiled | forward+backward | 20.328 | 18.568 | 1.095 | reference 17g/13b ⚠; clifra 11g/6b | [log](logs/cuda/cgenn_lorentz--default--compiled.log) |
| scaled | eager | forward | 23.936 | 11.919 | 2.008 | — | [log](logs/cuda/cgenn_lorentz--scaled--eager.log) |
| scaled | eager | forward+backward | 62.283 | 39.657 | 1.571 | — | [log](logs/cuda/cgenn_lorentz--scaled--eager.log) |
| scaled | compiled | forward | 11.470 | 8.125 | 1.412 | reference 20g/17b; clifra 14g/7b | [log](logs/cuda/cgenn_lorentz--scaled--compiled.log) |
| scaled | compiled | forward+backward | 30.007 | 26.290 | 1.141 | reference 17g/13b ⚠; clifra 11g/6b | [log](logs/cuda/cgenn_lorentz--scaled--compiled.log) |

## gatr

Default flags: `--model gatr`. Scaled additions: `--items 16 --hidden-mv-channels 8 --hidden-s-channels 32 --blocks 3 --heads 4`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 9.968 | 12.013 | 0.830 | — | [log](logs/cuda/gatr--default--eager.log) |
| default | eager | forward+backward | 39.820 | 35.725 | 1.115 | — | [log](logs/cuda/gatr--default--eager.log) |
| default | compiled | forward | 4.501 | 2.829 | 1.591 | reference 10g/5b; clifra 1g/0b | [log](logs/cuda/gatr--default--compiled.log) |
| default | compiled | forward+backward | 15.534 | 9.854 | 1.576 | reference 10g/5b; clifra 1g/0b | [log](logs/cuda/gatr--default--compiled.log) |
| scaled | eager | forward | 15.515 | 18.975 | 0.818 | — | [log](logs/cuda/gatr--scaled--eager.log) |
| scaled | eager | forward+backward | 62.376 | 56.799 | 1.098 | — | [log](logs/cuda/gatr--scaled--eager.log) |
| scaled | compiled | forward | 7.337 | 4.728 | 1.552 | reference 10g/5b; clifra 1g/0b | [log](logs/cuda/gatr--scaled--compiled.log) |
| scaled | compiled | forward+backward | 24.365 | 15.892 | 1.533 | reference 10g/5b; clifra 1g/0b | [log](logs/cuda/gatr--scaled--compiled.log) |
