# MPS paired-model sweep

Run started: 2026-09-20T10:54:11+00:00.
Source commit: `459c6c09f30625a6a214ee610e9bcfc4cd84543e`. Host: `macOS-26.6.2-arm64-arm-64bit`.
Device: `mps`; float32; seed 0; CPU intra-op/inter-op threads: 4/4.
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
| default | eager | forward | 6.845 | 0.824 | 8.304 | — | [log](logs/mps/gca_mlp--default--eager.log) |
| default | eager | forward+backward | 15.650 | 7.622 | 2.053 | — | [log](logs/mps/gca_mlp--default--eager.log) |
| default | compiled | forward | 1.306 | 0.567 | 2.304 | clifra 1g/0b; reference 1g/0b | [log](logs/mps/gca_mlp--default--compiled.log) |
| default | compiled | forward+backward | 3.063 | 1.378 | 2.223 | clifra 1g/0b; reference 1g/0b | [log](logs/mps/gca_mlp--default--compiled.log) |
| scaled | eager | forward | 12.245 | 2.386 | 5.131 | — | [log](logs/mps/gca_mlp--scaled--eager.log) |
| scaled | eager | forward+backward | 24.450 | 9.851 | 2.482 | — | [log](logs/mps/gca_mlp--scaled--eager.log) |
| scaled | compiled | forward | 4.226 | 1.619 | 2.610 | clifra 1g/0b; reference 1g/0b | [log](logs/mps/gca_mlp--scaled--compiled.log) |
| scaled | compiled | forward+backward | 11.225 | 4.636 | 2.421 | clifra 1g/0b; reference 1g/0b | [log](logs/mps/gca_mlp--scaled--compiled.log) |

## gca_gnn

Default flags: `--model gca_gnn`. Scaled additions: `--graphs 2 --nodes 32 --edges 128 --node-channels 16 --message-channels 16 --mlp-hidden-channels 32`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 34.800 | 9.177 | 3.792 | — | [log](logs/mps/gca_gnn--default--eager.log) |
| default | eager | forward+backward | 65.760 | 49.366 | 1.332 | — | [log](logs/mps/gca_gnn--default--eager.log) |
| default | compiled | forward | 2.302 | 1.438 | 1.601 | clifra 1g/0b; reference 1g/0b | [log](logs/mps/gca_gnn--default--compiled.log) |
| default | compiled | forward+backward | 5.464 | 3.091 | 1.768 | clifra 1g/0b; reference 1g/0b | [log](logs/mps/gca_gnn--default--compiled.log) |
| scaled | eager | forward | 40.361 | 7.738 | 5.216 | — | [log](logs/mps/gca_gnn--scaled--eager.log) |
| scaled | eager | forward+backward | 86.974 | 56.096 | 1.550 | — | [log](logs/mps/gca_gnn--scaled--eager.log) |
| scaled | compiled | forward | 2.613 | 1.222 | 2.139 | clifra 1g/0b; reference 1g/0b | [log](logs/mps/gca_gnn--scaled--compiled.log) |
| scaled | compiled | forward+backward | 7.552 | 3.842 | 1.966 | clifra 1g/0b; reference 1g/0b | [log](logs/mps/gca_gnn--scaled--compiled.log) |

## clifford_resnet2d

Default flags: `--model clifford_resnet2d`. Scaled additions: `--grid 48 --hidden-channels 8 --blocks 3`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 2.297 | 2.100 | 1.094 | — | [log](logs/mps/clifford_resnet2d--default--eager.log) |
| default | eager | forward+backward | 6.043 | 4.975 | 1.215 | — | [log](logs/mps/clifford_resnet2d--default--eager.log) |
| default | compiled | forward | — | — | — | — | [log](logs/mps/clifford_resnet2d--default--compiled.log) |
| default | compiled | forward+backward | — | — | — | — | [log](logs/mps/clifford_resnet2d--default--compiled.log) |
| scaled | eager | forward | 4.365 | 3.903 | 1.118 | — | [log](logs/mps/clifford_resnet2d--scaled--eager.log) |
| scaled | eager | forward+backward | 7.886 | 7.416 | 1.063 | — | [log](logs/mps/clifford_resnet2d--scaled--eager.log) |
| scaled | compiled | forward | — | — | — | — | [log](logs/mps/clifford_resnet2d--scaled--compiled.log) |
| scaled | compiled | forward+backward | — | — | — | — | [log](logs/mps/clifford_resnet2d--scaled--compiled.log) |

Validation and compiler notes:
- default forward reference: AssertionError: expected size 4==4, stride 1681==1 at dim=1; expected size 1681==1681, stride 1==4 at dim=2
- default forward clifra: AssertionError: expected size 4==4, stride 1681==1 at dim=1; expected size 1681==1681, stride 1==4 at dim=2
- default forward+backward reference: AssertionError: expected size 4==4, stride 1681==1 at dim=1; expected size 1681==1681, stride 1==4 at dim=2
- default forward+backward clifra: AssertionError: expected size 4==4, stride 1681==1 at dim=1; expected size 1681==1681, stride 1==4 at dim=2
- scaled forward reference: AssertionError: expected size 4==4, stride 3249==1 at dim=1; expected size 3249==3249, stride 1==4 at dim=2
- scaled forward clifra: AssertionError: expected size 4==4, stride 3249==1 at dim=1; expected size 3249==3249, stride 1==4 at dim=2
- scaled forward+backward reference: AssertionError: expected size 4==4, stride 3249==1 at dim=1; expected size 3249==3249, stride 1==4 at dim=2
- scaled forward+backward clifra: AssertionError: expected size 4==4, stride 3249==1 at dim=1; expected size 3249==3249, stride 1==4 at dim=2

## clifford_fno2d

Default flags: `--model clifford_fno2d`. Scaled additions: `--grid 48 --hidden-channels 8 --blocks 3 --modes 16`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 2.124 | 1.574 | 1.349 | — | [log](logs/mps/clifford_fno2d--default--eager.log) |
| default | eager | forward+backward | 4.794 | 3.895 | 1.231 | — | [log](logs/mps/clifford_fno2d--default--eager.log) |
| default | compiled | forward | 1.633 | 1.395 | 1.171 | reference 1g/0b; clifra 1g/0b | [log](logs/mps/clifford_fno2d--default--compiled.log) |
| default | compiled | forward+backward | 3.794 | 3.956 | 0.959 | reference 1g/0b; clifra 1g/0b | [log](logs/mps/clifford_fno2d--default--compiled.log) |
| scaled | eager | forward | 2.444 | 2.298 | 1.064 | — | [log](logs/mps/clifford_fno2d--scaled--eager.log) |
| scaled | eager | forward+backward | 5.446 | 5.277 | 1.032 | — | [log](logs/mps/clifford_fno2d--scaled--eager.log) |
| scaled | compiled | forward | 1.821 | 2.005 | 0.909 | reference 1g/0b; clifra 1g/0b | [log](logs/mps/clifford_fno2d--scaled--compiled.log) |
| scaled | compiled | forward+backward | 4.071 | 4.391 | 0.927 | reference 1g/0b; clifra 1g/0b | [log](logs/mps/clifford_fno2d--scaled--compiled.log) |

## clifford_fno3d

Default flags: `--model clifford_fno3d`. Scaled additions: `--grid 20 --hidden-channels 4 --blocks 3 --modes 8`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 3.461 | 1.751 | 1.976 | — | [log](logs/mps/clifford_fno3d--default--eager.log) |
| default | eager | forward+backward | 9.152 | 5.805 | 1.577 | — | [log](logs/mps/clifford_fno3d--default--eager.log) |
| default | compiled | forward | — | 1.378 | — | clifra 1g/0b | [log](logs/mps/clifford_fno3d--default--compiled.log) |
| default | compiled | forward+backward | — | 5.016 | — | clifra 1g/0b | [log](logs/mps/clifford_fno3d--default--compiled.log) |
| scaled | eager | forward | 6.014 | 5.459 | 1.102 | — | [log](logs/mps/clifford_fno3d--scaled--eager.log) |
| scaled | eager | forward+backward | 21.132 | 16.589 | 1.274 | — | [log](logs/mps/clifford_fno3d--scaled--eager.log) |
| scaled | compiled | forward | — | 4.171 | — | clifra 1g/0b | [log](logs/mps/clifford_fno3d--scaled--compiled.log) |
| scaled | compiled | forward+backward | — | 13.559 | — | clifra 1g/0b | [log](logs/mps/clifford_fno3d--scaled--compiled.log) |

Validation and compiler notes:
- default forward reference: RuntimeError: Passed CPU tensor to MPS op
- default forward+backward reference: RuntimeError: Passed CPU tensor to MPS op
- scaled forward reference: RuntimeError: Passed CPU tensor to MPS op
- scaled forward+backward reference: RuntimeError: Passed CPU tensor to MPS op

## cgenn/o3

Default flags: `--model cgenn --variant o3`. Scaled additions: `--batch 4 --hidden-features 32 --layers 4`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 8.242 | 4.084 | 2.018 | — | [log](logs/mps/cgenn_o3--default--eager.log) |
| default | eager | forward+backward | 34.297 | 28.527 | 1.202 | — | [log](logs/mps/cgenn_o3--default--eager.log) |
| default | compiled | forward | 6.475 | 3.932 | 1.647 | reference 17g/17b; clifra 15g/7b | [log](logs/mps/cgenn_o3--default--compiled.log) |
| default | compiled | forward+backward | 8.562 | 8.318 | 1.029 | reference 15g/14b; clifra 13g/6b | [log](logs/mps/cgenn_o3--default--compiled.log) |
| scaled | eager | forward | 8.745 | 4.810 | 1.818 | — | [log](logs/mps/cgenn_o3--scaled--eager.log) |
| scaled | eager | forward+backward | 36.518 | 28.872 | 1.265 | — | [log](logs/mps/cgenn_o3--scaled--eager.log) |
| scaled | compiled | forward | 6.667 | 4.465 | 1.493 | reference 17g/17b; clifra 15g/7b | [log](logs/mps/cgenn_o3--scaled--compiled.log) |
| scaled | compiled | forward+backward | 9.274 | 9.678 | 0.958 | reference 15g/14b; clifra 13g/6b | [log](logs/mps/cgenn_o3--scaled--compiled.log) |

## cgenn/o5_regression

Default flags: `--model cgenn --variant o5_regression`. Scaled additions: `--batch 4`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 3.897 | 1.834 | 2.124 | — | [log](logs/mps/cgenn_o5_regression--default--eager.log) |
| default | eager | forward+backward | 23.648 | 6.562 | 3.604 | — | [log](logs/mps/cgenn_o5_regression--default--eager.log) |
| default | compiled | forward | 3.108 | 1.700 | 1.829 | reference 8g/7b; clifra 7g/3b | [log](logs/mps/cgenn_o5_regression--default--compiled.log) |
| default | compiled | forward+backward | 4.538 | 3.798 | 1.195 | reference 8g/7b; clifra 7g/3b | [log](logs/mps/cgenn_o5_regression--default--compiled.log) |
| scaled | eager | forward | 4.103 | 1.946 | 2.109 | — | [log](logs/mps/cgenn_o5_regression--scaled--eager.log) |
| scaled | eager | forward+backward | 24.547 | 7.431 | 3.303 | — | [log](logs/mps/cgenn_o5_regression--scaled--eager.log) |
| scaled | compiled | forward | 3.488 | 1.858 | 1.877 | reference 8g/7b; clifra 7g/3b | [log](logs/mps/cgenn_o5_regression--scaled--compiled.log) |
| scaled | compiled | forward+backward | 4.264 | 3.643 | 1.171 | reference 8g/7b; clifra 7g/3b | [log](logs/mps/cgenn_o5_regression--scaled--compiled.log) |

## cgenn/hulls

Default flags: `--model cgenn --variant hulls`. Scaled additions: `--batch 4 --hidden-features 16 --layers 4`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 14.353 | 6.487 | 2.213 | — | [log](logs/mps/cgenn_hulls--default--eager.log) |
| default | eager | forward+backward | 47.975 | 18.437 | 2.602 | — | [log](logs/mps/cgenn_hulls--default--eager.log) |
| default | compiled | forward | 11.480 | 6.000 | 1.913 | reference 11g/9b; clifra 9g/4b | [log](logs/mps/cgenn_hulls--default--compiled.log) |
| default | compiled | forward+backward | 15.516 | 13.621 | 1.139 | reference 11g/8b; clifra 9g/4b | [log](logs/mps/cgenn_hulls--default--compiled.log) |
| scaled | eager | forward | 14.731 | 6.745 | 2.184 | — | [log](logs/mps/cgenn_hulls--scaled--eager.log) |
| scaled | eager | forward+backward | 31.973 | 14.871 | 2.150 | — | [log](logs/mps/cgenn_hulls--scaled--eager.log) |
| scaled | compiled | forward | 11.470 | 6.021 | 1.905 | reference 11g/9b; clifra 9g/4b | [log](logs/mps/cgenn_hulls--scaled--compiled.log) |
| scaled | compiled | forward+backward | 15.641 | 13.114 | 1.193 | reference 11g/8b; clifra 9g/4b | [log](logs/mps/cgenn_hulls--scaled--compiled.log) |

## cgenn/nbody

Default flags: `--model cgenn --variant nbody`. Scaled additions: `--batch 3 --nodes 6 --hidden-features 16 --layers 3`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 72.865 | 18.527 | 3.933 | — | [log](logs/mps/cgenn_nbody--default--eager.log) |
| default | eager | forward+backward | 126.473 | 70.292 | 1.799 | — | [log](logs/mps/cgenn_nbody--default--eager.log) |
| default | compiled | forward | 27.853 | 14.137 | 1.970 | reference 26g/18b; clifra 23g/7b | [log](logs/mps/cgenn_nbody--default--compiled.log) |
| default | compiled | forward+backward | 63.903 | 49.914 | 1.280 | reference 15g/13b ⚠; clifra 15g/5b ⚠ | [log](logs/mps/cgenn_nbody--default--compiled.log) |
| scaled | eager | forward | 71.169 | 25.985 | 2.739 | — | [log](logs/mps/cgenn_nbody--scaled--eager.log) |
| scaled | eager | forward+backward | 149.443 | 79.416 | 1.882 | — | [log](logs/mps/cgenn_nbody--scaled--eager.log) |
| scaled | compiled | forward | 41.653 | 21.314 | 1.954 | reference 26g/18b; clifra 23g/7b | [log](logs/mps/cgenn_nbody--scaled--compiled.log) |
| scaled | compiled | forward+backward | 99.615 | 72.235 | 1.379 | reference 15g/13b ⚠; clifra 15g/5b ⚠ | [log](logs/mps/cgenn_nbody--scaled--compiled.log) |

## cgenn/lorentz

Default flags: `--model cgenn --variant lorentz`. Scaled additions: `--batch 3 --nodes 6 --hidden-features 16 --hidden-scalar-features 8 --decoder-features 16 --layers 3`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 55.061 | 12.921 | 4.262 | — | [log](logs/mps/cgenn_lorentz--default--eager.log) |
| default | eager | forward+backward | 115.806 | 41.165 | 2.813 | — | [log](logs/mps/cgenn_lorentz--default--eager.log) |
| default | compiled | forward | 12.676 | 6.883 | 1.842 | reference 20g/17b; clifra 14g/7b | [log](logs/mps/cgenn_lorentz--default--compiled.log) |
| default | compiled | forward+backward | 32.218 | 16.276 | 1.979 | reference 17g/13b ⚠; clifra 11g/6b | [log](logs/mps/cgenn_lorentz--default--compiled.log) |
| scaled | eager | forward | 66.189 | 18.764 | 3.527 | — | [log](logs/mps/cgenn_lorentz--scaled--eager.log) |
| scaled | eager | forward+backward | 161.845 | 48.198 | 3.358 | — | [log](logs/mps/cgenn_lorentz--scaled--eager.log) |
| scaled | compiled | forward | 22.474 | 11.743 | 1.914 | reference 20g/17b; clifra 14g/7b | [log](logs/mps/cgenn_lorentz--scaled--compiled.log) |
| scaled | compiled | forward+backward | 55.156 | 27.954 | 1.973 | reference 17g/13b ⚠; clifra 11g/6b | [log](logs/mps/cgenn_lorentz--scaled--compiled.log) |

## gatr

Default flags: `--model gatr`. Scaled additions: `--items 16 --hidden-mv-channels 8 --hidden-s-channels 32 --blocks 3 --heads 4`.

| Workload | Execution | Mode | Reference ms | Clifra ms | Ratio | Capture | Raw log |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| default | eager | forward | 7.661 | 4.269 | 1.795 | — | [log](logs/mps/gatr--default--eager.log) |
| default | eager | forward+backward | 49.777 | 24.892 | 2.000 | — | [log](logs/mps/gatr--default--eager.log) |
| default | compiled | forward | 4.107 | 2.299 | 1.786 | reference 10g/5b; clifra 1g/0b | [log](logs/mps/gatr--default--compiled.log) |
| default | compiled | forward+backward | — | 5.140 | — | clifra 1g/0b | [log](logs/mps/gatr--default--compiled.log) |
| scaled | eager | forward | 24.759 | 10.629 | 2.329 | — | [log](logs/mps/gatr--scaled--eager.log) |
| scaled | eager | forward+backward | 95.973 | 59.980 | 1.600 | — | [log](logs/mps/gatr--scaled--eager.log) |
| scaled | compiled | forward | 6.551 | 3.260 | 2.009 | reference 10g/5b; clifra 1g/0b | [log](logs/mps/gatr--scaled--compiled.log) |
| scaled | compiled | forward+backward | — | 7.795 | — | clifra 1g/0b | [log](logs/mps/gatr--scaled--compiled.log) |

Validation and compiler notes:
- default forward+backward reference: AssertionError: Tensor-likes are not close!
- scaled forward+backward reference: AssertionError: Tensor-likes are not close!

