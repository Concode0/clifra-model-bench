"""Fixed CliffordLayers/Clifra 2D Clifford Fourier fluid comparison."""

from __future__ import annotations

import copy

import torch
from cliffordlayers.nn.modules.cliffordfourier import CliffordSpectralConv2d as ReferenceSpectral

from bench.core import PreparedBenchmark, PreparedCall, parameter_count
from models.clifford_resnet2d.benchmark import (
    SOURCE_COMMIT,
    assert_corresponding_gradients,
    check_convention,
    parameter_pairs,
    transfer,
)

from .clifra import ClifraFNO2d, pair_fourier, unpair_fourier
from .reference import ReferenceFNO2d


def add_arguments(parser) -> None:
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--in-channels", "--channels", type=int, default=1)
    parser.add_argument(
        "--out-channels", type=int, default=None, help="default: match input channels"
    )
    parser.add_argument("--hidden-channels", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--modes", type=int, default=16, help="retained Fourier modes per axis")


def check_arguments(parser, args) -> None:
    if args.out_channels is None:
        args.out_channels = args.in_channels
    for name in (
        "batch",
        "grid",
        "in_channels",
        "out_channels",
        "hidden_channels",
        "blocks",
        "modes",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.grid + 9 < 2 * args.modes:
        parser.error("--grid + 9 must be at least twice --modes")


def spectral_pairs(reference, clifra):
    return [
        (f"{name}.weights", source.weights, clifra.get_submodule(name).weights, None)
        for name, source in reference.named_modules()
        if isinstance(source, ReferenceSpectral)
    ]


def check_layers(reference, clifra, device):
    generator = torch.Generator().manual_seed(2311)
    boundary = torch.randn(1, reference.encoder.in_channels, 8, 8, 3, generator=generator).to(
        device
    )
    size = max(41, 2 * reference.layers[0][0].fourier.modes1 + 1)
    internal = torch.randn(
        1, reference.encoder.out_channels, size, size, 4, generator=generator
    ).to(device)
    first, second = pair_fourier(internal)
    torch.testing.assert_close(unpair_fourier(first, second), internal, rtol=0, atol=0)
    print("Fourier dual-pair roundtrip: PASS")
    modes = reference.layers[0][0].fourier.modes1
    crop_only = ReferenceSpectral(
        [1, 1], internal.shape[1], internal.shape[1], modes, modes, multiply=False
    ).to(device)
    identity_product = copy.deepcopy(clifra.layers[0][0].fourier)
    with torch.no_grad():
        identity_product.weights.zero_()
        for channel in range(internal.shape[1]):
            identity_product.weights[0, channel, channel].fill_(1)
    torch.testing.assert_close(
        identity_product(internal), crop_only(internal), rtol=2e-3, atol=2e-4
    )
    print("spectral mode selection: PASS")
    for label, source, target, values in (
        ("encoder", reference.encoder, clifra.encoder, boundary),
        ("decoder", reference.decoder, clifra.decoder, internal),
        ("local CliffordConv2d", reference.layers[0][0].conv, clifra.layers[0][0].conv, internal),
        (
            "spectral convolution",
            reference.layers[0][0].fourier,
            clifra.layers[0][0].fourier,
            internal,
        ),
        ("Fourier block", reference.layers[0][0], clifra.layers[0][0], internal),
    ):
        torch.testing.assert_close(target(values), source(values), rtol=2e-3, atol=2e-4, msg=label)
        print(f"{label} equivalence: PASS")
    source_spectral = reference.layers[0][0].fourier
    target_spectral = clifra.layers[0][0].fourier
    count = assert_corresponding_gradients(
        source_spectral,
        target_spectral,
        internal,
        extra_pairs=(("weights", source_spectral.weights, target_spectral.weights, None),),
    )
    print(f"spectral input and {count} parameter gradient: PASS")


def prepare(args, device: torch.device) -> PreparedBenchmark:
    options = dict(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        hidden_channels=args.hidden_channels,
        blocks=args.blocks,
        modes=args.modes,
    )
    reference = ReferenceFNO2d(**options)
    clifra = ClifraFNO2d(**options)
    transfer(reference, clifra, spectral_pairs(reference, clifra))
    reference, clifra = reference.to(device), clifra.to(device)
    if parameter_count(reference) != parameter_count(clifra):
        raise AssertionError("parameter counts differ")
    generator = torch.Generator().manual_seed(args.seed + 3031)
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
            pairs = spectral_pairs(reference, clifra)
            parameter_pairs(reference, clifra, pairs)
            count = assert_corresponding_gradients(reference, clifra, field, extra_pairs=pairs)
            print(f"whole-model output, input and {count} parameter gradients: PASS")

    return PreparedBenchmark(
        calls=[calls[name] for name in selected],
        work_units={"samples": args.batch, "grid points": args.batch * args.grid**2},
        validate=validate,
        description=(
            f"clifford_fno2d g=(2,0), batch={args.batch}, grid={args.grid}x{args.grid}, "
            f"channels={args.in_channels}/{args.out_channels}, hidden={args.hidden_channels}, blocks={args.blocks}, "
            f"modes=({args.modes},{args.modes}), norm=False, rotation=False, padding=9"
        ),
        reference_source=f"Brandstetter et al., Clifford Neural Layers for PDE Modeling; Microsoft cliffordlayers {SOURCE_COMMIT}",
        provenance_packages=("cliffordlayers",),
    )
