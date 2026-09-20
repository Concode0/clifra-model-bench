"""Fixed CliffordLayers/Clifra 2D fluid ResNet comparison."""

from __future__ import annotations

import torch
from cliffordlayers.nn.modules.cliffordconv import CliffordConv2d as ReferenceConv2d
from cliffordlayers.nn.modules.groupnorm import CliffordGroupNorm2d as ReferenceNorm2d
from clifra import make_algebra

from bench.core import PreparedBenchmark, PreparedCall, parameter_count

from .clifra import ClifraResNet2d
from .reference import ReferenceResNet2d

SOURCE_COMMIT = "74799cf4588a065916305bfcf2f030d84918f0ad"


def check_convention(field: torch.Tensor) -> None:
    scalar_vector = make_algebra(2, 0).layout((0, 1))
    torch.testing.assert_close(
        scalar_vector.compact(scalar_vector.full(field)), field, rtol=0, atol=0
    )
    print("basis/convention roundtrip: PASS (identical [1,e1,e2,e12] order)")


def add_arguments(parser) -> None:
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--in-channels", "--channels", type=int, default=1)
    parser.add_argument(
        "--out-channels", type=int, default=None, help="default: match input channels"
    )
    parser.add_argument("--hidden-channels", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=2)


def check_arguments(parser, args) -> None:
    if args.out_channels is None:
        args.out_channels = args.in_channels
    for name in ("batch", "grid", "in_channels", "out_channels", "hidden_channels", "blocks"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")


def parameter_pairs(reference, clifra, extra_pairs=()):
    """All learned convolution lanes and normalization affine parameters."""
    pairs = []
    for name, source in reference.named_modules():
        if not name:
            continue
        if isinstance(source, ReferenceConv2d):
            target = clifra.get_submodule(name)
            pairs.extend(
                (f"{name}.weight.{blade}", weight, target.weight, blade)
                for blade, weight in enumerate(source.weight)
            )
            if source.bias is not None:
                pairs.append((f"{name}.bias", source.bias, target.bias, None))
        elif isinstance(source, ReferenceNorm2d):
            target = clifra.get_submodule(name)
            pairs.extend(
                (
                    (f"{name}.weight", source.weight, target.weight, None),
                    (f"{name}.bias", source.bias, target.bias, None),
                )
            )
    pairs.extend(extra_pairs)
    source_parameters = {
        id(parameter) for parameter in reference.parameters() if parameter.requires_grad
    }
    target_parameters = {
        id(parameter) for parameter in clifra.parameters() if parameter.requires_grad
    }
    if {id(source) for _, source, _, _ in pairs} != source_parameters:
        raise AssertionError("reference parameter correspondence is incomplete")
    if {id(target) for _, _, target, _ in pairs} != target_parameters:
        raise AssertionError("Clifra parameter correspondence is incomplete")
    return pairs


def transfer(reference, clifra, extra_pairs=()) -> None:
    with torch.no_grad():
        for _, source, target, blade in parameter_pairs(reference, clifra, extra_pairs):
            (target if blade is None else target[blade]).copy_(source)


def assert_corresponding_gradients(
    reference, clifra, x: torch.Tensor, *, extra_pairs=(), rtol=2e-3, atol=2e-4
):
    pairs = parameter_pairs(reference, clifra, extra_pairs)
    source_input = x.detach().clone().requires_grad_(True)
    target_input = x.detach().clone().requires_grad_(True)
    reference.zero_grad(set_to_none=True)
    clifra.zero_grad(set_to_none=True)
    source_output = reference(source_input)
    target_output = clifra(target_input)
    torch.testing.assert_close(target_output, source_output, rtol=rtol, atol=atol)
    source_output.square().mean().backward()
    target_output.square().mean().backward()
    torch.testing.assert_close(target_input.grad, source_input.grad, rtol=rtol, atol=atol)
    for name, source, target, blade in pairs:
        if source.grad is None or target.grad is None:
            raise AssertionError(f"missing gradient: {name}")
        if not torch.isfinite(source.grad).all() or not torch.isfinite(target.grad).all():
            raise AssertionError(f"nonfinite gradient: {name}")
        torch.testing.assert_close(
            target.grad if blade is None else target.grad[blade],
            source.grad,
            rtol=rtol,
            atol=atol,
            msg=name,
        )
    reference.zero_grad(set_to_none=True)
    clifra.zero_grad(set_to_none=True)
    return len(pairs)


def check_layers(reference, clifra, device: torch.device) -> None:
    generator = torch.Generator().manual_seed(1741)

    def values(channels, blades):
        return torch.randn(1, channels, 7, 7, blades, generator=generator).to(device)

    for label, source, target, field in (
        ("encoder", reference.encoder, clifra.encoder, values(reference.encoder.in_channels, 3)),
        ("decoder", reference.decoder, clifra.decoder, values(reference.decoder.in_channels, 4)),
        (
            "conv2d",
            reference.layers[0][0].conv1,
            clifra.layers[0][0].conv1,
            values(reference.layers[0][0].conv1.in_channels, 4),
        ),
        (
            "basic block",
            reference.layers[0][0],
            clifra.layers[0][0],
            values(reference.layers[0][0].conv1.in_channels, 4),
        ),
    ):
        torch.testing.assert_close(target(field), source(field), rtol=2e-3, atol=2e-4, msg=label)
        print(f"{label} equivalence: PASS")


def prepare(args, device: torch.device) -> PreparedBenchmark:
    options = dict(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        hidden_channels=args.hidden_channels,
        blocks=args.blocks,
    )
    reference = ReferenceResNet2d(**options, mps_norm_patch=device.type == "mps")
    clifra = ClifraResNet2d(**options)
    transfer(reference, clifra)
    reference, clifra = reference.to(device), clifra.to(device)
    if parameter_count(reference) != parameter_count(clifra):
        raise AssertionError("parameter counts differ")
    generator = torch.Generator().manual_seed(args.seed + 2027)
    field = torch.randn(
        args.batch, args.in_channels, args.grid, args.grid, 3, generator=generator
    ).to(device)
    calls = {
        "reference": PreparedCall(
            "reference",
            reference,
            (field.detach().clone().requires_grad_(True),),
            loss=lambda out: out.square().mean(),
        ),
        "clifra": PreparedCall(
            "clifra",
            clifra,
            (field.detach().clone().requires_grad_(True),),
            loss=lambda out: out.square().mean(),
        ),
    }
    selected = ["reference", "clifra"] if args.backend == "both" else [args.backend]

    def validate(outputs):
        check_convention(field)
        print(f"parameter correspondence: PASS ({parameter_count(reference):,} scalars)")
        for name, output in outputs.items():
            if output.shape != (args.batch, args.out_channels, args.grid, args.grid, 3):
                raise AssertionError(f"{name}: unexpected output shape {tuple(output.shape)}")
        if len(outputs) == 2:
            check_layers(reference, clifra, device)
            count = assert_corresponding_gradients(reference, clifra, field)
            print(f"whole-model output, input and {count} parameter gradients: PASS")

    return PreparedBenchmark(
        calls=[calls[name] for name in selected],
        work_units={"samples": args.batch, "grid points": args.batch * args.grid**2},
        validate=validate,
        description=(
            f"clifford_resnet2d g=(2,0), batch={args.batch}, grid={args.grid}x{args.grid}, "
            f"channels={args.in_channels}/{args.out_channels}, hidden={args.hidden_channels}, blocks={args.blocks}, "
            "norm=True, rotation=False, padding=9"
        ),
        reference_source=(
            f"Brandstetter et al., Clifford Neural Layers for PDE Modeling; Microsoft cliffordlayers {SOURCE_COMMIT}"
            + (" (MPS triangular-solve layout patch)" if device.type == "mps" else "")
        ),
        provenance_packages=("cliffordlayers",),
    )
