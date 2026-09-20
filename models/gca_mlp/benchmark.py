"""GCA-MLP comparison: native calls, parameter alignment, and correctness."""

from __future__ import annotations

import torch

from bench.basis import SignedBasisTransform
from bench.core import PreparedBenchmark, PreparedCall

from .clifra import ClifraGCAMLP
from .reference import ReferenceGCAMLP


def add_arguments(parser) -> None:
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--items", type=int, default=64)
    parser.add_argument("--in-channels", type=int, default=8)
    parser.add_argument("--out-channels", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--act-agg", choices=("linear", "sum", "mean"), default="linear")
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="flatten items and MV channels as in the official n-body GCA-MLP config",
    )


def check_arguments(parser, args) -> None:
    for name in (
        "batch",
        "items",
        "in_channels",
        "out_channels",
        "hidden_channels",
        "hidden_layers",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")


def convention(reference: ReferenceGCAMLP, clifra: ClifraGCAMLP) -> SignedBasisTransform:
    """Map cliffordlayers' null-first PGA blades to Clifra's null-last blades."""
    reference_bitmaps = [int(value) for value in reference.pga.bbo.index_to_bitmap]
    target_position = {
        blade: position for position, blade in enumerate(clifra.algebra.layout().basis_indices)
    }
    targets, signs = [], []
    for bitmap in reference_bitmaps:
        # Reference generators (e0,e1,e2,e3) correspond to Clifra (e4,e1,e2,e3).
        bits = [3 if bit == 0 else bit - 1 for bit in range(4) if bitmap & (1 << bit)]
        inversions = sum(
            bits[left] > bits[right]
            for left in range(len(bits))
            for right in range(left + 1, len(bits))
        )
        targets.append(target_position[sum(1 << bit for bit in bits)])
        signs.append(-1.0 if inversions % 2 else 1.0)
    return SignedBasisTransform(tuple(targets), tuple(signs))


def transfer(
    reference: ReferenceGCAMLP, clifra: ClifraGCAMLP, transform: SignedBasisTransform
) -> None:
    """Transfer each trainable parameter through the coefficient isomorphism."""
    with torch.no_grad():
        for source, target in zip(reference.linears, clifra.linears):
            target.weight.copy_(source.weight)
            target.embed_e0.copy_(source.embed_e0)
            target._action.zero_()
            target_positions = {
                blade: position for position, blade in enumerate(target.action_basis_indices)
            }
            for source_position, reference_lane in enumerate(source.action_blades):
                clifra_lane = transform.target_for_source[reference_lane]
                sign = transform.sign_for_source[reference_lane]
                target._action[..., target_positions[clifra_lane]].copy_(
                    sign * source._action[..., source_position]
                )
        for source, target in zip(reference.activations, clifra.activations):
            if hasattr(source, "conv"):
                target.gate_weight.copy_(transform.apply(source.conv.weight.squeeze(1)))
                target.gate_bias.copy_(source.conv.bias)


def _loss(output: torch.Tensor) -> torch.Tensor:
    return output.square().mean()


def check_gradients(reference, clifra, transform, reference_input, clifra_input) -> None:
    """Compare every learned lane under the signed PGA basis isomorphism."""
    reference.zero_grad(set_to_none=True)
    clifra.zero_grad(set_to_none=True)
    reference_input.grad = None
    clifra_input.grad = None
    source_output = reference(reference_input)
    target_output = clifra(clifra_input)
    probe = torch.randn(source_output.shape, generator=torch.Generator().manual_seed(1733)).to(
        source_output.device
    )
    (source_output * probe).sum().backward()
    (target_output * transform.apply(probe)).sum().backward()
    torch.testing.assert_close(
        transform.inverse().apply(clifra_input.grad), reference_input.grad, rtol=1e-3, atol=1e-5
    )
    covered_source, covered_target = set(), set()

    def checked(source, target, actual):
        torch.testing.assert_close(actual, source.grad, rtol=1e-3, atol=1e-5)
        covered_source.add(id(source))
        covered_target.add(id(target))

    for source, target in zip(reference.linears, clifra.linears, strict=True):
        checked(source.weight, target.weight, target.weight.grad)
        checked(source.embed_e0, target.embed_e0, target.embed_e0.grad)
        positions = {blade: index for index, blade in enumerate(target.action_basis_indices)}
        for source_index, source_lane in enumerate(source.action_blades):
            target_index = positions[transform.target_for_source[source_lane]]
            torch.testing.assert_close(
                transform.sign_for_source[source_lane] * target._action.grad[..., target_index],
                source._action.grad[..., source_index],
                rtol=1e-3,
                atol=1e-5,
            )
        covered_source.add(id(source._action))
        covered_target.add(id(target._action))
    for source, target in zip(reference.activations, clifra.activations, strict=True):
        if hasattr(source, "conv"):
            checked(
                source.conv.weight,
                target.gate_weight,
                transform.inverse().apply(target.gate_weight.grad).unsqueeze(1),
            )
            checked(source.conv.bias, target.gate_bias, target.gate_bias.grad)
    if covered_source != {id(p) for p in reference.parameters() if p.requires_grad}:
        raise AssertionError("reference trainable parameter omitted from gradient comparison")
    if covered_target != {id(p) for p in clifra.parameters() if p.requires_grad}:
        raise AssertionError("Clifra trainable parameter omitted from gradient comparison")
    reference.zero_grad(set_to_none=True)
    clifra.zero_grad(set_to_none=True)
    reference_input.grad = None
    clifra_input.grad = None


def prepare(args, device: torch.device) -> PreparedBenchmark:
    configuration = dict(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        hidden_channels=args.hidden_channels,
        hidden_layers=args.hidden_layers,
        act_agg=args.act_agg,
        flatten=args.flatten,
        items=args.items,
    )
    reference = ReferenceGCAMLP(**configuration)
    clifra = ClifraGCAMLP(**configuration)
    transform = convention(reference, clifra)
    transfer(reference, clifra, transform)
    reference = reference.to(device)
    clifra = clifra.to(device)
    reference_input = torch.randn(
        args.batch, args.items, args.in_channels, 16, device=device, dtype=torch.float32
    )
    clifra_input = transform.apply(reference_input)
    calls = {
        "reference": PreparedCall(
            "reference", reference, (reference_input.detach().requires_grad_(True),), loss=_loss
        ),
        "clifra": PreparedCall(
            "clifra", clifra, (clifra_input.detach().requires_grad_(True),), loss=_loss
        ),
    }
    selected = ["clifra", "reference"] if args.backend == "both" else [args.backend]

    def validate(outputs: dict[str, torch.Tensor]) -> None:
        roundtrip = transform.inverse().apply(transform.apply(reference_input))
        torch.testing.assert_close(roundtrip, reference_input, rtol=0.0, atol=0.0)
        print("convention roundtrip: PASS")
        expected_shape = (args.batch, args.items, args.out_channels, 16)
        for name, output in outputs.items():
            if output.shape != expected_shape:
                raise AssertionError(
                    f"{name}: got shape {tuple(output.shape)}, expected {expected_shape}"
                )
        if len(outputs) == 2:
            converted = transform.inverse().apply(outputs["clifra"])
            error = float((outputs["reference"] - converted).abs().max().detach().cpu())
            torch.testing.assert_close(converted, outputs["reference"], rtol=1e-3, atol=1e-4)
            print(f"transferred whole-model equivalence: PASS (max={error:.3e})")
            _check_layers(reference, clifra, transform, device)
            check_gradients(
                reference, clifra, transform, calls["reference"].args[0], calls["clifra"].args[0]
            )
            print("corresponding input and all parameter gradients: PASS")
        print(
            "equivariance: not asserted; upstream documents GCA-MLP as using PGA representations but not E(3)-equivariant"
        )

    return PreparedBenchmark(
        calls=[calls[name] for name in selected],
        work_units={"samples": args.batch, "MV": args.batch * args.items * args.out_channels},
        validate=validate,
        description=(
            f"gca_mlp in={args.in_channels}, out={args.out_channels}, hidden={args.hidden_channels}, "
            f"hidden_layers={args.hidden_layers}, act_agg={args.act_agg}, flatten={args.flatten}, batch={args.batch}, items={args.items}"
        ),
        reference_source=(
            "Qualcomm GATr 6afc26f26b8fcf51136ae8c1d264a36e14b6e497; "
            "faithful GCA wrapper over cliffordlayers 74799cf4588a065916305bfcf2f030d84918f0ad primitives"
        ),
        provenance_packages=("cliffordlayers",),
    )


def _check_layers(reference, clifra, transform, device) -> None:
    generator = torch.Generator(device="cpu").manual_seed(1729)
    max_linear_error = 0.0
    max_activation_error = 0.0
    for reference_layer, clifra_layer in zip(reference.linears, clifra.linears):
        source = torch.randn(3, reference_layer.in_features, 16, generator=generator).to(device)
        expected = transform.apply(reference_layer(source))
        actual = clifra_layer(transform.apply(source))
        max_linear_error = max(
            max_linear_error, float((expected - actual).abs().max().detach().cpu())
        )
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-5)
    for reference_activation, clifra_activation in zip(reference.activations, clifra.activations):
        source = torch.randn(3, reference.linears[0].out_features, 16, generator=generator).to(
            device
        )
        expected = transform.apply(reference_activation(source))
        actual = clifra_activation(transform.apply(source))
        max_activation_error = max(
            max_activation_error, float((expected - actual).abs().max().detach().cpu())
        )
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-5)
    print(
        f"transferred layer equivalence: PASS (linear max={max_linear_error:.3e}, activation max={max_activation_error:.3e})"
    )
