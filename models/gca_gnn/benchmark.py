"""Prepared GCA-GNN graph calls, alignment, and graph-specific checks."""

from __future__ import annotations

import torch

from bench.core import PreparedBenchmark, PreparedCall
from models.gca_mlp.benchmark import convention, transfer

from .clifra import ClifraGCAGNN
from .reference import ReferenceGCAGNN


def add_arguments(parser) -> None:
    parser.add_argument("--graphs", type=int, default=2)
    parser.add_argument("--nodes", type=int, default=16, help="nodes per graph")
    parser.add_argument(
        "--edges",
        type=int,
        default=64,
        help="unique directed edges per graph; nodes*(nodes-1) is the official complete graph",
    )
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--out-channels", type=int, default=4)
    parser.add_argument("--node-channels", type=int, default=8)
    parser.add_argument("--message-channels", type=int, default=8)
    parser.add_argument("--mlp-hidden-channels", type=int, default=16)
    parser.add_argument("--mlp-hidden-layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3, help="message-passing layers")


def check_arguments(parser, args) -> None:
    for name in (
        "graphs",
        "nodes",
        "edges",
        "in_channels",
        "out_channels",
        "node_channels",
        "message_channels",
        "mlp_hidden_channels",
        "mlp_hidden_layers",
        "steps",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.nodes < 2 or args.edges < args.nodes:
        parser.error("GCA-GNN needs at least two nodes and at least one ring edge per node")
    if args.edges > args.nodes * (args.nodes - 1):
        parser.error("--edges cannot exceed the loop-free directed complete graph")
    if args.steps < 2:
        parser.error(
            "this benchmark uses at least two steps to keep initial and final stages distinct"
        )


def _graph(graphs: int, nodes: int, edges: int, seed: int) -> torch.Tensor:
    """Disjoint directed graphs: a ring plus unique loop-free sampled edges."""
    generator = torch.Generator(device="cpu").manual_seed(seed + 1729)
    ring_source = torch.arange(nodes)
    ring_target = (ring_source + 1) % nodes
    candidates = torch.cartesian_prod(torch.arange(nodes), torch.arange(nodes))
    candidates = candidates[
        (candidates[:, 0] != candidates[:, 1])
        & (candidates[:, 1] != (candidates[:, 0] + 1) % nodes)
    ]
    chosen = candidates[torch.randperm(len(candidates), generator=generator)[: edges - nodes]]
    source = torch.cat((ring_source, chosen[:, 0]))
    target = torch.cat((ring_target, chosen[:, 1]))
    graph_edges = torch.stack((source, target))
    return torch.cat([graph_edges + graph * nodes for graph in range(graphs)], dim=1)


def _mlps(model):
    for layer in model.layers:
        yield layer.message_mlp
        yield layer.update_mlp


def _loss(output: torch.Tensor) -> torch.Tensor:
    return output.square().mean()


def _check_mlps(reference, clifra, transform, device) -> None:
    generator = torch.Generator(device="cpu").manual_seed(1730)
    max_error = 0.0
    for source, target in zip(_mlps(reference), _mlps(clifra), strict=True):
        values = torch.randn(3, source.in_channels, 16, generator=generator).to(device)
        expected = transform.apply(source(values))
        actual = target(transform.apply(values))
        max_error = max(max_error, float((expected - actual).abs().max().detach().cpu()))
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
    print(f"transferred message/update MLPs: PASS (max={max_error:.3e})")


def _check_layers(reference, clifra, transform, edge_index, node_count, device) -> None:
    generator = torch.Generator(device="cpu").manual_seed(1731)
    max_error = 0.0
    for source, target in zip(reference.layers, clifra.layers, strict=True):
        values = torch.randn(
            node_count, source.message_mlp.in_channels // 2, 16, generator=generator
        ).to(device)
        expected = transform.apply(source(values, edge_index))
        actual = target(transform.apply(values), edge_index)
        max_error = max(max_error, float((expected - actual).abs().max().detach().cpu()))
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
    print(f"transferred GCA-GNN layers: PASS (max={max_error:.3e})")


def _check_mlp_gradients(reference, clifra, transform) -> None:
    checked_reference: set[int] = set()
    checked_clifra: set[int] = set()

    def mark(source_parameter, target_parameter) -> None:
        checked_reference.add(id(source_parameter))
        checked_clifra.add(id(target_parameter))

    for source, target in zip(_mlps(reference), _mlps(clifra), strict=True):
        for source_linear, target_linear in zip(source.linears, target.linears, strict=True):
            torch.testing.assert_close(
                target_linear.weight.grad, source_linear.weight.grad, rtol=1e-3, atol=1e-5
            )
            mark(source_linear.weight, target_linear.weight)
            torch.testing.assert_close(
                target_linear.embed_e0.grad, source_linear.embed_e0.grad, rtol=1e-3, atol=1e-5
            )
            mark(source_linear.embed_e0, target_linear.embed_e0)
            target_positions = {
                blade: pos for pos, blade in enumerate(target_linear.action_basis_indices)
            }
            for source_pos, reference_lane in enumerate(source_linear.action_blades):
                target_pos = target_positions[transform.target_for_source[reference_lane]]
                sign = transform.sign_for_source[reference_lane]
                torch.testing.assert_close(
                    sign * target_linear._action.grad[..., target_pos],
                    source_linear._action.grad[..., source_pos],
                    rtol=1e-3,
                    atol=1e-5,
                )
            mark(source_linear._action, target_linear._action)
        for source_act, target_act in zip(source.activations, target.activations, strict=True):
            torch.testing.assert_close(
                transform.inverse().apply(target_act.gate_weight.grad),
                source_act.conv.weight.grad.squeeze(1),
                rtol=1e-3,
                atol=1e-5,
            )
            mark(source_act.conv.weight, target_act.gate_weight)
            torch.testing.assert_close(
                target_act.gate_bias.grad, source_act.conv.bias.grad, rtol=1e-3, atol=1e-5
            )
            mark(source_act.conv.bias, target_act.gate_bias)
    all_reference = {
        id(parameter) for parameter in reference.parameters() if parameter.requires_grad
    }
    all_clifra = {id(parameter) for parameter in clifra.parameters() if parameter.requires_grad}
    if checked_reference != all_reference or checked_clifra != all_clifra:
        raise AssertionError("parameter-gradient correspondence omitted a trainable parameter")


def _check_gradients(
    reference, clifra, reference_input, clifra_input, edge_index, transform
) -> None:
    reference.zero_grad(set_to_none=True)
    clifra.zero_grad(set_to_none=True)
    reference_input.grad = None
    clifra_input.grad = None
    reference_output = reference(reference_input, edge_index)
    clifra_output = clifra(clifra_input, edge_index)
    generator = torch.Generator(device="cpu").manual_seed(1733)
    probe = torch.randn(reference_output.shape, generator=generator).to(reference_output.device)
    (reference_output * probe).sum().backward()
    (clifra_output * transform.apply(probe)).sum().backward()
    torch.testing.assert_close(
        transform.inverse().apply(clifra_input.grad), reference_input.grad, rtol=1e-3, atol=1e-5
    )
    _check_mlp_gradients(reference, clifra, transform)
    reference.zero_grad(set_to_none=True)
    clifra.zero_grad(set_to_none=True)
    reference_input.grad = None
    clifra_input.grad = None
    print("corresponding input and parameter gradients: PASS")


def prepare(args, device: torch.device) -> PreparedBenchmark:
    configuration = dict(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        node_channels=args.node_channels,
        message_channels=args.message_channels,
        mlp_hidden_channels=args.mlp_hidden_channels,
        mlp_hidden_layers=args.mlp_hidden_layers,
        message_passing_steps=args.steps,
    )
    reference = ReferenceGCAGNN(**configuration)
    clifra = ClifraGCAGNN(**configuration)
    transform = convention(reference.layers[0].message_mlp, clifra.layers[0].message_mlp)
    for source, target in zip(_mlps(reference), _mlps(clifra), strict=True):
        transfer(source, target, transform)
    reference = reference.to(device)
    clifra = clifra.to(device)
    edge_index = _graph(args.graphs, args.nodes, args.edges, args.seed).to(device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1732)
    reference_values = torch.randn(
        args.graphs * args.nodes, args.in_channels, 16, generator=generator
    ).to(device)
    clifra_values = transform.apply(reference_values)
    reference_input = reference_values.detach().requires_grad_(True)
    clifra_input = clifra_values.detach().requires_grad_(True)
    calls = {
        "reference": PreparedCall("reference", reference, (reference_input, edge_index), _loss),
        "clifra": PreparedCall("clifra", clifra, (clifra_input, edge_index), _loss),
    }

    def validate(outputs: dict[str, torch.Tensor]) -> None:
        torch.testing.assert_close(
            transform.inverse().apply(transform.apply(reference_values)),
            reference_values,
            rtol=0,
            atol=0,
        )
        print("convention roundtrip: PASS")
        expected_shape = (args.graphs * args.nodes, args.out_channels, 16)
        for name, output in outputs.items():
            if output.shape != expected_shape:
                raise AssertionError(
                    f"{name}: got shape {tuple(output.shape)}, expected {expected_shape}"
                )
        if len(outputs) == 2:
            converted = transform.inverse().apply(outputs["clifra"])
            error = float((outputs["reference"] - converted).abs().max().detach().cpu())
            torch.testing.assert_close(converted, outputs["reference"], rtol=1e-3, atol=1e-4)
            print(f"transferred whole GCA-GNN: PASS (max={error:.3e})")
            _check_mlps(reference, clifra, transform, device)
            _check_layers(
                reference, clifra, transform, edge_index, args.graphs * args.nodes, device
            )
            _check_gradients(
                reference, clifra, reference_input, clifra_input, edge_index, transform
            )
        print(
            "equivariance: not asserted; GCA-GNN is not constructed as an E(3)-equivariant architecture"
        )

    selected = ["clifra", "reference"] if args.backend == "both" else [args.backend]
    return PreparedBenchmark(
        calls=[calls[name] for name in selected],
        work_units={
            "graphs": args.graphs,
            "nodes": args.graphs * args.nodes,
            "edges": args.graphs * args.edges,
            "output MV": args.graphs * args.nodes * args.out_channels,
        },
        validate=validate,
        description=(
            f"gca_gnn graphs={args.graphs}, nodes/graph={args.nodes}, edges/graph={args.edges}, "
            f"in={args.in_channels}, out={args.out_channels}, node={args.node_channels}, "
            f"message={args.message_channels}, mlp_hidden={args.mlp_hidden_channels}, "
            f"mlp_hidden_layers={args.mlp_hidden_layers}, steps={args.steps}"
        ),
        reference_source=(
            "Qualcomm GATr 6afc26f26b8fcf51136ae8c1d264a36e14b6e497; "
            "faithful GCA wrapper over cliffordlayers 74799cf4588a065916305bfcf2f030d84918f0ad primitives"
        ),
        provenance_packages=("cliffordlayers", "torch-geometric"),
    )
