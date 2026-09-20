# CPU paired-model sweep

Run started: 2026-09-20T10:25:59+00:00.
Source commit: `459c6c09f30625a6a214ee610e9bcfc4cd84543e`. Host: `macOS-26.6.2-arm64-arm-64bit`.
Device: `cpu`; float32; seed 0; CPU intra-op/inter-op threads: 4/4.
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
| default | eager | forward | 55.834 | 1.740 | 32.094 | — | [log](logs/cpu/gca_mlp--default--eager.log) |
| default | eager | forward+backward | 697.770 | 5.317 | 131.238 | — | [log](logs/cpu/gca_mlp--default--eager.log) |
| default | compiled | forward | 54.197 | 1.636 | 33.130 | clifra 1g/0b; reference 1g/0b | [log](logs/cpu/gca_mlp--default--compiled.log) |
| default | compiled | forward+backward | 736.295 | 4.485 | 164.165 | clifra 1g/0b; reference 1g/0b | [log](logs/cpu/gca_mlp--default--compiled.log) |
| scaled | eager | forward | 213.692 | 6.533 | 32.711 | — | [log](logs/cpu/gca_mlp--scaled--eager.log) |
| scaled | eager | forward+backward | 2642.048 | 17.041 | 155.039 | — | [log](logs/cpu/gca_mlp--scaled--eager.log) |
| scaled | compiled | forward | 183.605 | 4.818 | 38.111 | clifra 1g/0b; reference 1g/0b | [log](logs/cpu/gca_mlp--scaled--compiled.log) |
| scaled | compiled | forward+backward | 2410.546 | 13.908 | 173.318 | clifra 1g/0b; reference 1g/0b | [log](logs/cpu/gca_mlp--scaled--compiled.log) |

## gca_gnn

Default flags: `--model gca_gnn`. Scaled additions: `--graphs 2 --nodes 32 --edges 128 --node-channels 16 --message-channels 16 --mlp-hidden-channels 32`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 32.580 | 2.562 | 12.715 | — | [log](logs/cpu/gca_gnn--default--eager.log) |
| default | eager | forward+backward | 225.126 | 6.865 | 32.794 | — | [log](logs/cpu/gca_gnn--default--eager.log) |
| default | compiled | forward | 29.422 | 2.640 | 11.145 | clifra 1g/0b; reference 1g/0b | [log](logs/cpu/gca_gnn--default--compiled.log) |
| default | compiled | forward+backward | 229.669 | 6.443 | 35.646 | clifra 1g/0b; reference 1g/0b | [log](logs/cpu/gca_gnn--default--compiled.log) |
| scaled | eager | forward | 104.996 | 5.259 | 19.964 | — | [log](logs/cpu/gca_gnn--scaled--eager.log) |
| scaled | eager | forward+backward | 909.135 | 16.000 | 56.822 | — | [log](logs/cpu/gca_gnn--scaled--eager.log) |
| scaled | compiled | forward | 99.752 | 5.241 | 19.034 | clifra 1g/0b; reference 1g/0b | [log](logs/cpu/gca_gnn--scaled--compiled.log) |
| scaled | compiled | forward+backward | 971.552 | 13.563 | 71.632 | clifra 1g/0b; reference 1g/0b | [log](logs/cpu/gca_gnn--scaled--compiled.log) |

## clifford_resnet2d

Default flags: `--model clifford_resnet2d`. Scaled additions: `--grid 48 --hidden-channels 8 --blocks 3`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 1.548 | 1.367 | 1.132 | — | [log](logs/cpu/clifford_resnet2d--default--eager.log) |
| default | eager | forward+backward | 4.571 | 3.762 | 1.215 | — | [log](logs/cpu/clifford_resnet2d--default--eager.log) |
| default | compiled | forward | — | 1.850 | — | clifra 1g/0b | [log](logs/cpu/clifford_resnet2d--default--compiled.log) |
| default | compiled | forward+backward | 4.368 | 4.329 | 1.009 | reference 1g/0b; clifra 1g/0b | [log](logs/cpu/clifford_resnet2d--default--compiled.log) |
| scaled | eager | forward | 5.906 | 4.778 | 1.236 | — | [log](logs/cpu/clifford_resnet2d--scaled--eager.log) |
| scaled | eager | forward+backward | 17.261 | 13.398 | 1.288 | — | [log](logs/cpu/clifford_resnet2d--scaled--eager.log) |
| scaled | compiled | forward | 6.182 | — | — | reference 1g/0b | [log](logs/cpu/clifford_resnet2d--scaled--compiled.log) |
| scaled | compiled | forward+backward | — | — | — | — | [log](logs/cpu/clifford_resnet2d--scaled--compiled.log) |

Validation and compiler notes:
- default forward reference: AssertionError: Tensor-likes are not close!
- scaled forward clifra: AssertionError: Tensor-likes are not close!
- scaled forward+backward reference: AssertionError: Tensor-likes are not close!
- scaled forward+backward clifra: AssertionError: Tensor-likes are not close!

## clifford_fno2d

Default flags: `--model clifford_fno2d`. Scaled additions: `--grid 48 --hidden-channels 8 --blocks 3 --modes 16`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 1.195 | 1.020 | 1.172 | — | [log](logs/cpu/clifford_fno2d--default--eager.log) |
| default | eager | forward+backward | 4.697 | 3.077 | 1.526 | — | [log](logs/cpu/clifford_fno2d--default--eager.log) |
| default | compiled | forward | 1.754 | 1.296 | 1.353 | reference 1g/0b; clifra 1g/0b | [log](logs/cpu/clifford_fno2d--default--compiled.log) |
| default | compiled | forward+backward | 3.151 | 2.918 | 1.080 | reference 1g/0b; clifra 1g/0b | [log](logs/cpu/clifford_fno2d--default--compiled.log) |
| scaled | eager | forward | 9.782 | 9.051 | 1.081 | — | [log](logs/cpu/clifford_fno2d--scaled--eager.log) |
| scaled | eager | forward+backward | 31.312 | 27.226 | 1.150 | — | [log](logs/cpu/clifford_fno2d--scaled--eager.log) |
| scaled | compiled | forward | 3.974 | 3.507 | 1.133 | reference 1g/0b; clifra 1g/0b | [log](logs/cpu/clifford_fno2d--scaled--compiled.log) |
| scaled | compiled | forward+backward | 9.024 | 8.712 | 1.036 | reference 1g/0b; clifra 1g/0b | [log](logs/cpu/clifford_fno2d--scaled--compiled.log) |

## clifford_fno3d

Default flags: `--model clifford_fno3d`. Scaled additions: `--grid 20 --hidden-channels 4 --blocks 3 --modes 8`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 4.581 | 4.308 | 1.063 | — | [log](logs/cpu/clifford_fno3d--default--eager.log) |
| default | eager | forward+backward | 26.002 | 15.030 | 1.730 | — | [log](logs/cpu/clifford_fno3d--default--eager.log) |
| default | compiled | forward | 4.237 | 2.396 | 1.769 | reference 1g/0b; clifra 1g/0b | [log](logs/cpu/clifford_fno3d--default--compiled.log) |
| default | compiled | forward+backward | 9.253 | 7.371 | 1.255 | reference 1g/0b; clifra 1g/0b | [log](logs/cpu/clifford_fno3d--default--compiled.log) |
| scaled | eager | forward | 54.069 | 51.673 | 1.046 | — | [log](logs/cpu/clifford_fno3d--scaled--eager.log) |
| scaled | eager | forward+backward | 176.260 | 130.273 | 1.353 | — | [log](logs/cpu/clifford_fno3d--scaled--eager.log) |
| scaled | compiled | forward | 16.277 | 10.043 | 1.621 | reference 1g/0b; clifra 1g/0b | [log](logs/cpu/clifford_fno3d--scaled--compiled.log) |
| scaled | compiled | forward+backward | 37.492 | 33.486 | 1.120 | reference 1g/0b; clifra 1g/0b | [log](logs/cpu/clifford_fno3d--scaled--compiled.log) |

## cgenn/o3

Default flags: `--model cgenn --variant o3`. Scaled additions: `--batch 4 --hidden-features 32 --layers 4`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 1.109 | 0.262 | 4.234 | — | [log](logs/cpu/cgenn_o3--default--eager.log) |
| default | eager | forward+backward | 2.170 | 0.790 | 2.749 | — | [log](logs/cpu/cgenn_o3--default--eager.log) |
| default | compiled | forward | 0.674 | 0.444 | 1.518 | reference 17g/17b; clifra 15g/7b | [log](logs/cpu/cgenn_o3--default--compiled.log) |
| default | compiled | forward+backward | 1.223 | 1.011 | 1.210 | reference 15g/14b; clifra 13g/6b | [log](logs/cpu/cgenn_o3--default--compiled.log) |
| scaled | eager | forward | 1.362 | 0.308 | 4.424 | — | [log](logs/cpu/cgenn_o3--scaled--eager.log) |
| scaled | eager | forward+backward | 2.360 | 0.883 | 2.672 | — | [log](logs/cpu/cgenn_o3--scaled--eager.log) |
| scaled | compiled | forward | 0.731 | 0.448 | 1.633 | reference 17g/17b; clifra 15g/7b | [log](logs/cpu/cgenn_o3--scaled--compiled.log) |
| scaled | compiled | forward+backward | 1.620 | 1.273 | 1.272 | reference 15g/14b; clifra 13g/6b | [log](logs/cpu/cgenn_o3--scaled--compiled.log) |

## cgenn/o5_regression

Default flags: `--model cgenn --variant o5_regression`. Scaled additions: `--batch 4`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 0.775 | 0.348 | 2.228 | — | [log](logs/cpu/cgenn_o5_regression--default--eager.log) |
| default | eager | forward+backward | 1.497 | 1.047 | 1.430 | — | [log](logs/cpu/cgenn_o5_regression--default--eager.log) |
| default | compiled | forward | 0.556 | 0.406 | 1.368 | reference 8g/7b; clifra 7g/3b | [log](logs/cpu/cgenn_o5_regression--default--compiled.log) |
| default | compiled | forward+backward | 1.232 | 1.350 | 0.912 | reference 8g/7b; clifra 7g/3b | [log](logs/cpu/cgenn_o5_regression--default--compiled.log) |
| scaled | eager | forward | 0.823 | 0.395 | 2.083 | — | [log](logs/cpu/cgenn_o5_regression--scaled--eager.log) |
| scaled | eager | forward+backward | 1.588 | 1.102 | 1.442 | — | [log](logs/cpu/cgenn_o5_regression--scaled--eager.log) |
| scaled | compiled | forward | 0.652 | 0.479 | 1.363 | reference 8g/7b; clifra 7g/3b | [log](logs/cpu/cgenn_o5_regression--scaled--compiled.log) |
| scaled | compiled | forward+backward | 1.518 | 1.530 | 0.992 | reference 8g/7b; clifra 7g/3b | [log](logs/cpu/cgenn_o5_regression--scaled--compiled.log) |

## cgenn/hulls

Default flags: `--model cgenn --variant hulls`. Scaled additions: `--batch 4 --hidden-features 16 --layers 4`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 3.015 | 1.202 | 2.508 | — | [log](logs/cpu/cgenn_hulls--default--eager.log) |
| default | eager | forward+backward | 5.799 | 3.506 | 1.654 | — | [log](logs/cpu/cgenn_hulls--default--eager.log) |
| default | compiled | forward | 1.959 | 1.316 | 1.489 | reference 11g/9b; clifra 9g/4b | [log](logs/cpu/cgenn_hulls--default--compiled.log) |
| default | compiled | forward+backward | 4.853 | 4.849 | 1.001 | reference 11g/8b; clifra 9g/4b | [log](logs/cpu/cgenn_hulls--default--compiled.log) |
| scaled | eager | forward | 4.110 | 2.108 | 1.950 | — | [log](logs/cpu/cgenn_hulls--scaled--eager.log) |
| scaled | eager | forward+backward | 8.526 | 5.672 | 1.503 | — | [log](logs/cpu/cgenn_hulls--scaled--eager.log) |
| scaled | compiled | forward | 2.933 | 2.100 | 1.397 | reference 11g/9b; clifra 9g/4b | [log](logs/cpu/cgenn_hulls--scaled--compiled.log) |
| scaled | compiled | forward+backward | 7.631 | 7.447 | 1.025 | reference 11g/8b; clifra 9g/4b | [log](logs/cpu/cgenn_hulls--scaled--compiled.log) |

## cgenn/nbody

Default flags: `--model cgenn --variant nbody`. Scaled additions: `--batch 3 --nodes 6 --hidden-features 16 --layers 3`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 5.520 | 1.832 | 3.014 | — | [log](logs/cpu/cgenn_nbody--default--eager.log) |
| default | eager | forward+backward | 9.785 | 4.671 | 2.095 | — | [log](logs/cpu/cgenn_nbody--default--eager.log) |
| default | compiled | forward | 3.378 | 2.114 | 1.598 | reference 26g/18b; clifra 23g/7b | [log](logs/cpu/cgenn_nbody--default--compiled.log) |
| default | compiled | forward+backward | 6.840 | 5.291 | 1.293 | reference 15g/13b ⚠; clifra 15g/5b ⚠ | [log](logs/cpu/cgenn_nbody--default--compiled.log) |
| scaled | eager | forward | 10.890 | 5.231 | 2.082 | — | [log](logs/cpu/cgenn_nbody--scaled--eager.log) |
| scaled | eager | forward+backward | 20.428 | 12.366 | 1.652 | — | [log](logs/cpu/cgenn_nbody--scaled--eager.log) |
| scaled | compiled | forward | 9.954 | 8.278 | 1.202 | reference 26g/18b; clifra 23g/7b | [log](logs/cpu/cgenn_nbody--scaled--compiled.log) |
| scaled | compiled | forward+backward | 22.563 | 21.603 | 1.044 | reference 15g/13b ⚠; clifra 15g/5b ⚠ | [log](logs/cpu/cgenn_nbody--scaled--compiled.log) |

## cgenn/lorentz

Default flags: `--model cgenn --variant lorentz`. Scaled additions: `--batch 3 --nodes 6 --hidden-features 16 --hidden-scalar-features 8 --decoder-features 16 --layers 3`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 4.157 | 1.974 | 2.106 | — | [log](logs/cpu/cgenn_lorentz--default--eager.log) |
| default | eager | forward+backward | 6.968 | 3.831 | 1.819 | — | [log](logs/cpu/cgenn_lorentz--default--eager.log) |
| default | compiled | forward | 2.777 | 1.505 | 1.846 | reference 20g/17b; clifra 14g/7b | [log](logs/cpu/cgenn_lorentz--default--compiled.log) |
| default | compiled | forward+backward | 6.866 | 4.872 | 1.409 | reference 17g/13b ⚠; clifra 11g/6b | [log](logs/cpu/cgenn_lorentz--default--compiled.log) |
| scaled | eager | forward | 25.835 | 8.883 | 2.909 | — | [log](logs/cpu/cgenn_lorentz--scaled--eager.log) |
| scaled | eager | forward+backward | 50.233 | 20.681 | 2.429 | — | [log](logs/cpu/cgenn_lorentz--scaled--eager.log) |
| scaled | compiled | forward | 32.710 | 11.707 | 2.794 | reference 20g/17b; clifra 14g/7b | [log](logs/cpu/cgenn_lorentz--scaled--compiled.log) |
| scaled | compiled | forward+backward | 46.003 | 21.137 | 2.176 | reference 17g/13b ⚠; clifra 11g/6b | [log](logs/cpu/cgenn_lorentz--scaled--compiled.log) |

## gatr

Default flags: `--model gatr`. Scaled additions: `--items 16 --hidden-mv-channels 8 --hidden-s-channels 32 --blocks 3 --heads 4`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 1.192 | 1.937 | 0.615 | — | [log](logs/cpu/gatr--default--eager.log) |
| default | eager | forward+backward | 4.217 | 4.619 | 0.913 | — | [log](logs/cpu/gatr--default--eager.log) |
| default | compiled | forward | 1.854 | 1.213 | 1.528 | reference 10g/5b; clifra 1g/0b | [log](logs/cpu/gatr--default--compiled.log) |
| default | compiled | forward+backward | 4.234 | 2.538 | 1.668 | reference 10g/5b; clifra 1g/0b | [log](logs/cpu/gatr--default--compiled.log) |
| scaled | eager | forward | 2.550 | 3.713 | 0.687 | — | [log](logs/cpu/gatr--scaled--eager.log) |
| scaled | eager | forward+backward | 8.163 | 8.985 | 0.908 | — | [log](logs/cpu/gatr--scaled--eager.log) |
| scaled | compiled | forward | 4.152 | 3.294 | 1.260 | reference 10g/5b; clifra 1g/0b | [log](logs/cpu/gatr--scaled--compiled.log) |
| scaled | compiled | forward+backward | 11.195 | 8.035 | 1.393 | reference 10g/5b; clifra 1g/0b | [log](logs/cpu/gatr--scaled--compiled.log) |

