"""Fixed CGENN variants, convention transfer, and correctness checks."""

from __future__ import annotations

import itertools

import torch
from clifra import make_algebra

from bench.basis import SignedBasisTransform
from bench.core import PreparedBenchmark, PreparedCall, parameter_count

from . import clifra as native
from . import reference as established

SOURCE_COMMIT = "08f2f2cbf5f7a723ea8539daf9f20a24cde73b1b"
VARIANTS = ("o3", "o5_regression", "hulls", "nbody", "lorentz")
PAPER_SCALE = {
    "o3": "batch=128, hidden=96, layers=4",
    "o5_regression": "batch=32, fixed 2→8 GP and 8→580→580→1 head",
    "hulls": "batch=128, hidden=32, layers=4",
    "nbody": "batch=100, hidden=28, layers=3",
    "lorentz": "batch=32, hidden_h=72, hidden_x=8, layers=4, decoder=64",
}


def add_arguments(parser) -> None:
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default="o3",
        help="default widths are small correctness workloads",
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--hidden-features", type=int)
    parser.add_argument("--hidden-scalar-features", type=int)
    parser.add_argument("--decoder-features", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--nodes", type=int, default=4)
    parser.epilog = "Official README experiment scales: " + "; ".join(
        f"{variant}: {scale}" for variant, scale in PAPER_SCALE.items()
    )


def check_arguments(parser, args) -> None:
    for name in ("batch", "nodes"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    for name in ("hidden_features", "hidden_scalar_features", "decoder_features", "layers"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.variant == "o5_regression" and (
        args.hidden_features is not None or args.layers is not None
    ):
        parser.error("o5_regression uses the official fixed network width and depth")
    if args.variant != "lorentz" and (
        args.hidden_scalar_features is not None or args.decoder_features is not None
    ):
        parser.error("--hidden-scalar-features and --decoder-features apply only to lorentz")


def signature(variant: str) -> tuple[int, int]:
    if variant in ("o3", "nbody"):
        return 3, 0
    if variant in ("o5_regression", "hulls"):
        return 5, 0
    return 1, 3


def basis_transform(
    reference_algebra: established.CliffordAlgebra, algebra
) -> SignedBasisTransform:
    canonical = tuple(int(i) for i in algebra.layout().indices_tensor())
    position = {blade: lane for lane, blade in enumerate(canonical)}
    target = tuple(position[int(blade)] for blade in reference_algebra.index_to_bitmap)
    return SignedBasisTransform(target, (1,) * len(target))


def _source_paths(layer: established._ProductLayer):
    # The source mask is indexed (left, right, output), then expanded against a
    # Cayley tensor indexed (left, output, right). Its learned lane therefore acts
    # on the semantic path (left, mask-third, mask-second).
    return {
        (int(left), int(output), int(right)): lane
        for lane, (left, right, output) in enumerate(layer.product_paths.nonzero().tolist())
    }


def parameter_pairs(reference, clifra):
    source = dict(reference.named_parameters())
    target = dict(clifra.named_parameters())
    if source.keys() != target.keys():
        raise AssertionError(f"parameter names differ: {source.keys() ^ target.keys()}")
    path_permutations = {}
    for name, module in reference.named_modules():
        if isinstance(module, established._ProductLayer):
            counterpart = clifra.get_submodule(name) if name else clifra
            source_paths = _source_paths(module)
            if set(source_paths) != set(counterpart.path_triples):
                raise AssertionError(f"geometric-product paths differ in {name}")
            path_permutations[f"{name}.weight" if name else "weight"] = tuple(
                source_paths[triple] for triple in counterpart.path_triples
            )
    return [(name, source[name], target[name], path_permutations.get(name)) for name in source]


def transfer(reference, clifra) -> None:
    with torch.no_grad():
        for name, source, target, permutation in parameter_pairs(reference, clifra):
            values = source if permutation is None else source[..., list(permutation)]
            if values.shape != target.shape:
                raise AssertionError(f"parameter shape differs: {name}")
            target.copy_(values)
        for name, module in reference.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                counterpart = clifra.get_submodule(name)
                counterpart.running_mean.copy_(module.running_mean)
                counterpart.running_var.copy_(module.running_var)
                counterpart.num_batches_tracked.copy_(module.num_batches_tracked)


def compare(
    reference,
    clifra,
    source_args: tuple,
    mapping: SignedBasisTransform,
    geometric_args: tuple[int, ...],
    *,
    geometric_output: bool,
    rtol=2e-3,
    atol=2e-4,
):
    pairs = parameter_pairs(reference, clifra)
    source_inputs = []
    target_inputs = []
    for index, value in enumerate(source_args):
        if not isinstance(value, torch.Tensor):
            source_inputs.append(value)
            target_inputs.append(value)
            continue
        differentiable = value.dtype.is_floating_point and index not in (5, 6)
        source_value = value.detach().clone().requires_grad_(differentiable)
        mapped = mapping.apply(value) if index in geometric_args else value
        target_value = mapped.detach().clone().requires_grad_(differentiable)
        source_inputs.append(source_value)
        target_inputs.append(target_value)
    reference.zero_grad(set_to_none=True)
    clifra.zero_grad(set_to_none=True)
    source_output = reference(*source_inputs)
    target_output = clifra(*target_inputs)
    aligned = mapping.inverse().apply(target_output) if geometric_output else target_output
    torch.testing.assert_close(aligned, source_output, rtol=rtol, atol=atol)
    source_output.square().mean().backward()
    target_output.square().mean().backward()
    for index, (left, right) in enumerate(zip(source_inputs, target_inputs)):
        if isinstance(left, torch.Tensor) and left.requires_grad:
            if left.grad is None or right.grad is None:
                raise AssertionError(f"missing input gradient {index}")
            aligned_grad = (
                mapping.inverse().apply(right.grad) if index in geometric_args else right.grad
            )
            torch.testing.assert_close(aligned_grad, left.grad, rtol=rtol, atol=atol)
    for name, source, target, permutation in pairs:
        if source.grad is None or target.grad is None:
            raise AssertionError(f"missing parameter gradient {name}")
        if not torch.isfinite(source.grad).all() or not torch.isfinite(target.grad).all():
            raise AssertionError(f"nonfinite parameter gradient {name}")
        expected = source.grad if permutation is None else source.grad[..., list(permutation)]
        torch.testing.assert_close(target.grad, expected, rtol=rtol, atol=atol, msg=name)
    reference.zero_grad(set_to_none=True)
    clifra.zero_grad(set_to_none=True)
    return len(pairs)


def check_primitives(p: int, q: int, mapping: SignedBasisTransform, device: torch.device) -> None:
    source_algebra = established.CliffordAlgebra((1.0,) * p + (-1.0,) * q).to(device)
    structure = native.CGStructure(make_algebra(p, q))
    generator = torch.Generator().manual_seed(887 + p)
    values = torch.randn(2, 3, 1 << (p + q), generator=generator).to(device)
    samples = (
        ("MVLinear", established.MVLinear(source_algebra, 3, 2), native.MVLinear(structure, 3, 2)),
        (
            "MVLinear shared",
            established.MVLinear(source_algebra, 3, 2, subspaces=False),
            native.MVLinear(structure, 3, 2, subspaces=False),
        ),
        (
            "NormalizationLayer",
            established.NormalizationLayer(source_algebra, 3),
            native.NormalizationLayer(structure, 3),
        ),
        ("MVSiLU", established.MVSiLU(source_algebra, 3), native.MVSiLU(structure, 3)),
        (
            "MVLayerNorm",
            established.MVLayerNorm(source_algebra, 3),
            native.MVLayerNorm(structure, 3),
        ),
        (
            "SteerableGP",
            established.SteerableGeometricProductLayer(source_algebra, 3),
            native.SteerableGeometricProductLayer(structure, 3),
        ),
        (
            "FullyConnectedGP",
            established.FullyConnectedSteerableGeometricProductLayer(source_algebra, 3, 2),
            native.FullyConnectedSteerableGeometricProductLayer(structure, 3, 2),
        ),
    )
    for label, reference, clifra in samples:
        transfer(reference, clifra)
        reference, clifra = reference.to(device), clifra.to(device)
        compare(reference, clifra, (values,), mapping, (0,), geometric_output=True)
        for trial in range(2):
            direction = torch.randn(
                p + q, generator=torch.Generator().manual_seed(2331 + 17 * p + trial)
            ).to(device)
            if q:
                direction[trial] += 3
            normal = source_algebra.embed_grade(direction, 1)
            if source_algebra.q(normal).abs().item() < 0.2:
                raise AssertionError("deterministic reflection normal was nearly null")
            moved = source_algebra.rho(normal, values)
            with torch.no_grad():
                expected = source_algebra.rho(normal, reference(values))
                actual = mapping.inverse().apply(clifra(mapping.apply(moved)))
                torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-4)
                torch.testing.assert_close(reference(moved), expected, rtol=3e-3, atol=3e-4)
        print(f"{label}: output/input/all parameter gradients and reflections PASS")


def _graph_edges(graphs: int, nodes: int) -> torch.Tensor:
    rows, cols = [], []
    for graph in range(graphs):
        offset = graph * nodes
        for i, j in itertools.product(range(nodes), repeat=2):
            if i != j:
                rows.append(offset + i)
                cols.append(offset + j)
    return torch.tensor((rows, cols), dtype=torch.long)


def _make_model(variant, args):
    hidden = args.hidden_features
    layers = args.layers
    if variant == "o3":
        options = dict(hidden_features=hidden or 16, num_layers=layers or 3)
        return established.O3CGMLP(**options), native.O3CGMLP(**options), options
    if variant == "o5_regression":
        return established.O5CGMLP(), native.O5CGMLP(), {}
    if variant == "hulls":
        options = dict(hidden_features=hidden or 8, num_layers=layers or 4)
        return established.ConvexHullCGMLP(**options), native.ConvexHullCGMLP(**options), options
    if variant == "nbody":
        options = dict(hidden_features=hidden or 8, n_layers=layers or 2)
        return established.NBodyCGGNN(**options), native.NBodyCGGNN(**options), options
    options = dict(
        hidden_features_x=hidden or 4,
        hidden_features_h=args.hidden_scalar_features or 8,
        n_layers=layers or 2,
        decoder_features=args.decoder_features or 8,
    )
    return established.LorentzCGGNN(**options), native.LorentzCGGNN(**options), options


def _inputs(variant, args, device):
    generator = torch.Generator().manual_seed(args.seed + 9817)
    p, q = signature(variant)
    algebra = established.CliffordAlgebra((1.0,) * p + (-1.0,) * q)
    if variant in ("o3", "o5_regression", "hulls"):
        channels = {"o3": 3, "o5_regression": 2, "hulls": 16}[variant]
        points = torch.randn(args.batch, channels, p + q, generator=generator).to(device)
        return (
            (algebra.embed_grade(points, 1),),
            (0,),
            False,
            {
                "samples": args.batch,
                "points": args.batch * channels,
            },
        )
    edges = _graph_edges(args.batch, args.nodes).to(device)
    total_nodes = args.batch * args.nodes
    vectors = torch.randn(total_nodes, p + q, generator=generator).to(device)
    if variant == "nbody":
        charges = torch.randn(total_nodes, 1, generator=generator).to(device)
        velocity = torch.randn(total_nodes, p + q, generator=generator).to(device)
        h = torch.stack(
            (
                algebra.embed(charges, (0,)),
                algebra.embed_grade(vectors, 1),
                algebra.embed_grade(velocity, 1),
            ),
            dim=1,
        )
        edge_scalar = torch.randn(edges.shape[1], 1, generator=generator).to(device)
        edge_attr = algebra.embed(edge_scalar, (0,)).unsqueeze(1)
        return (
            (h, edges, edge_attr),
            (0, 2),
            True,
            {
                "graphs": args.batch,
                "nodes": total_nodes,
                "edges": edges.shape[1],
            },
        )
    h = torch.randn(total_nodes, 2, generator=generator).to(device)
    x = algebra.embed_grade(vectors[:, None], 1)
    edge_x = torch.stack(
        (vectors[edges[0]] - vectors[edges[1]], vectors[edges[0]], vectors[edges[1]]), dim=1
    )
    edge_attr_x = algebra.embed_grade(edge_x, 1)
    node_attr_x = x.clone()
    node_attr_h = h.clone()
    node_mask = torch.ones(total_nodes, 1, device=device)
    return (
        (h, x, edge_attr_x, node_attr_x, node_attr_h, edges, node_mask, args.nodes),
        (1, 2, 3),
        False,
        {
            "graphs": args.batch,
            "nodes": total_nodes,
            "edges": edges.shape[1],
        },
    )


def _reflect(algebra, normal, values):
    return algebra.rho(normal, values)


def check_equivariance(
    variant, reference, clifra, args, mapping, geometric_args, source_args, geometric_output
):
    source_algebra = reference.algebra
    generator = torch.Generator().manual_seed(1301 + source_algebra.dim)
    first_normal = None
    for trial in range(3):
        direction = torch.randn(source_algebra.dim, generator=generator).to(
            source_algebra.cayley.device
        )
        if variant == "lorentz":
            direction[0] += 3 if trial % 2 == 0 else 0
            direction[1] += 3 if trial % 2 else 0
        normal = source_algebra.embed_grade(direction, 1)
        if source_algebra.q(normal).abs().item() < 0.2:
            raise AssertionError("deterministic reflection normal was nearly null")
        if trial == 0:
            first_normal = normal
        versor = source_algebra.geometric_product(first_normal, normal) if trial == 2 else normal
        moved = tuple(
            _reflect(source_algebra, versor, value) if index in geometric_args else value
            for index, value in enumerate(source_args)
        )
        native_args = tuple(
            mapping.apply(value) if index in geometric_args else value
            for index, value in enumerate(source_args)
        )
        moved_native = tuple(
            mapping.apply(value) if index in geometric_args else value
            for index, value in enumerate(moved)
        )
        with torch.no_grad():
            source_before, source_after = reference(*source_args), reference(*moved)
            native_before, native_after = clifra(*native_args), clifra(*moved_native)
            if geometric_output:
                expected = _reflect(source_algebra, versor, source_before)
            elif variant == "o3":
                expected = source_before if trial == 2 else -source_before
            else:
                expected = source_before
            torch.testing.assert_close(source_after, expected, rtol=3e-3, atol=3e-4)
            native_after = (
                mapping.inverse().apply(native_after) if geometric_output else native_after
            )
            native_before = (
                mapping.inverse().apply(native_before) if geometric_output else native_before
            )
            native_expected = (
                _reflect(source_algebra, versor, native_before)
                if geometric_output
                else (-native_before if variant == "o3" and trial != 2 else native_before)
            )
            torch.testing.assert_close(native_after, native_expected, rtol=3e-3, atol=3e-4)
    print("two reflections and one composed Clifford-group action: PASS")


def prepare(args, device: torch.device) -> PreparedBenchmark:
    variant = args.variant
    p, q = signature(variant)
    reference, clifra, options = _make_model(variant, args)
    mapping = basis_transform(reference.algebra, make_algebra(p, q))
    transfer(reference, clifra)
    reference, clifra = reference.to(device), clifra.to(device)
    reference.eval()
    clifra.eval()
    if parameter_count(reference) != parameter_count(clifra):
        raise AssertionError("parameter counts differ")
    source_args, geometric_args, geometric_output, work_units = _inputs(variant, args, device)
    target_args = tuple(
        mapping.apply(value) if index in geometric_args else value
        for index, value in enumerate(source_args)
    )
    source_args = tuple(
        value.detach().clone().requires_grad_(value.dtype.is_floating_point and index not in (5, 6))
        if isinstance(value, torch.Tensor)
        else value
        for index, value in enumerate(source_args)
    )
    target_args = tuple(
        value.detach().clone().requires_grad_(value.dtype.is_floating_point and index not in (5, 6))
        if isinstance(value, torch.Tensor)
        else value
        for index, value in enumerate(target_args)
    )
    calls = {
        "reference": PreparedCall(
            "reference", reference, source_args, loss=lambda out: out.square().mean()
        ),
        "clifra": PreparedCall("clifra", clifra, target_args, loss=lambda out: out.square().mean()),
    }
    selected = ["reference", "clifra"] if args.backend == "both" else [args.backend]

    def validate(outputs):
        probe = torch.randn(2, 1 << (p + q), generator=torch.Generator().manual_seed(1911)).to(
            device
        )
        torch.testing.assert_close(
            mapping.inverse().apply(mapping.apply(probe)), probe, rtol=0, atol=0
        )
        for grade in range(p + q + 1):
            source_indices = reference.algebra.grade_to_index[grade]
            target_indices = make_algebra(p, q).layout((grade,)).indices_tensor()
            if sorted(mapping.target_for_source[int(i)] for i in source_indices) != sorted(
                int(i) for i in target_indices
            ):
                raise AssertionError(f"grade {grade} convention differs")
        print("basis roundtrip and per-grade layouts: PASS")
        print(
            f"parameter correspondence: PASS ({parameter_count(reference):,} scalars; {len(parameter_pairs(reference, clifra))} tensors)"
        )
        if len(outputs) == 2:
            aligned = (
                mapping.inverse().apply(outputs["clifra"])
                if geometric_output
                else outputs["clifra"]
            )
            torch.testing.assert_close(aligned, outputs["reference"], rtol=2e-3, atol=2e-4)
            check_primitives(p, q, mapping, device)
            count = compare(
                reference,
                clifra,
                source_args,
                mapping,
                geometric_args,
                geometric_output=geometric_output,
            )
            print(f"whole-model output, input and {count} parameter tensor gradients: PASS")
            check_equivariance(
                variant,
                reference,
                clifra,
                args,
                mapping,
                geometric_args,
                source_args,
                geometric_output,
            )

    return PreparedBenchmark(
        calls=[calls[name] for name in selected],
        work_units=work_units,
        validate=validate,
        description=f"cgenn variant={variant}, signature=({p},{q}), batch={args.batch}, nodes/graph={args.nodes if variant in ('nbody', 'lorentz') else 'n/a'}, options={options}, eval=True",
        reference_source=f"Ruhe, Brandstetter, Forré, Clifford Group Equivariant Neural Networks; DavidRuhe/clifford-group-equivariant-neural-networks {SOURCE_COMMIT} (execution-only port; upstream b() indentation repaired)",
    )
