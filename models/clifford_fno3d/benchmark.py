"""Fixed CliffordLayers/Clifra 3D Maxwell Fourier comparison."""

from __future__ import annotations

import copy
import itertools

import torch
from cliffordlayers.cliffordkernels import get_3d_clifford_kernel
from cliffordlayers.nn.modules.cliffordconv import CliffordConv3d as ReferenceConv3d
from cliffordlayers.nn.modules.cliffordfourier import CliffordSpectralConv3d as ReferenceSpectral
from clifra import make_algebra

from bench.basis import SignedBasisTransform
from bench.core import PreparedBenchmark, PreparedCall, parameter_count

from .clifra import ClifraFNO3d, pair_fourier, product_basis_3d, unpair_fourier
from .reference import ReferenceFNO3d

SOURCE_COMMIT = "74799cf4588a065916305bfcf2f030d84918f0ad"


def _blade_map(grades: tuple[int, ...]) -> SignedBasisTransform:
    """The source orders blades by grade; both libraries orient e1,e2,e3 alike."""
    source_blades = tuple(
        sum(1 << generator for generator in generators)
        for grade in grades
        for generators in itertools.combinations(range(3), grade)
    )
    target_blades = tuple(
        int(index) for index in make_algebra(3, 0).layout(grades).indices_tensor()
    )
    position = {blade: index for index, blade in enumerate(target_blades)}
    return SignedBasisTransform(
        tuple(position[blade] for blade in source_blades), (1,) * len(source_blades)
    )


FULL_MAP = _blade_map((0, 1, 2, 3))
BOUNDARY_MAP = _blade_map((1, 2))


def add_arguments(parser) -> None:
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--in-channels", "--channels", type=int, default=1)
    parser.add_argument(
        "--out-channels", type=int, default=None, help="default: match input channels"
    )
    parser.add_argument("--hidden-channels", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--modes", type=int, default=8, help="retained Fourier modes per axis")


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
    if args.grid + 2 < 2 * args.modes:
        parser.error("--grid + 2 must be at least twice --modes")


def check_convention(field: torch.Tensor) -> None:
    algebra = make_algebra(3, 0)
    boundary = algebra.layout((1, 2))
    clifra_field = BOUNDARY_MAP.apply(field)
    torch.testing.assert_close(BOUNDARY_MAP.inverse().apply(clifra_field), field, rtol=0, atol=0)
    torch.testing.assert_close(
        boundary.compact(boundary.full(clifra_field)), clifra_field, rtol=0, atol=0
    )
    full = torch.arange(8, dtype=field.dtype, device=field.device)
    torch.testing.assert_close(FULL_MAP.inverse().apply(FULL_MAP.apply(full)), full, rtol=0, atol=0)
    print("full and vector+bivector convention roundtrips: PASS")


def parameter_pairs(reference, clifra):
    pairs = []
    for name, source in reference.named_modules():
        if isinstance(source, ReferenceConv3d):
            target = clifra.get_submodule(name) if name else clifra
            for blade, weight in enumerate(source.weight):
                pairs.append(
                    (
                        f"{name}.weight.{blade}",
                        weight,
                        target.weight,
                        FULL_MAP.target_for_source[blade],
                        None,
                    )
                )
            if source.bias is not None:
                mapping = BOUNDARY_MAP if name == "decoder" else FULL_MAP
                pairs.append((f"{name}.bias", source.bias, target.bias, None, mapping))
        elif isinstance(source, ReferenceSpectral):
            target = clifra.get_submodule(name) if name else clifra
            pairs.append((f"{name}.weights", source.weights, target.weights, None, FULL_MAP))
    source_ids = {id(p) for p in reference.parameters() if p.requires_grad}
    target_ids = {id(p) for p in clifra.parameters() if p.requires_grad}
    if {id(source) for _, source, _, _, _ in pairs} != source_ids:
        raise AssertionError("reference parameter correspondence is incomplete")
    if {id(target) for _, _, target, _, _ in pairs} != target_ids:
        raise AssertionError("Clifra parameter correspondence is incomplete")
    return pairs


def mapped_parameter(value, blade, mapping):
    if blade is not None:
        return value
    if mapping is None:
        return value
    # Weight/bias coefficient lane is the leading axis for these reference modules.
    return mapping.apply(value.movedim(0, -1)).movedim(-1, 0)


def transfer(reference, clifra) -> None:
    with torch.no_grad():
        for _, source, target, blade, mapping in parameter_pairs(reference, clifra):
            data = mapped_parameter(source, blade, mapping)
            (target[blade] if blade is not None else target).copy_(data)


def assert_corresponding_gradients(
    reference,
    clifra,
    field: torch.Tensor,
    input_map: SignedBasisTransform,
    output_map: SignedBasisTransform,
    *,
    rtol=2e-3,
    atol=2e-4,
):
    pairs = parameter_pairs(reference, clifra)
    source_input = field.detach().clone().requires_grad_(True)
    target_input = input_map.apply(field).detach().clone().requires_grad_(True)
    reference.zero_grad(set_to_none=True)
    clifra.zero_grad(set_to_none=True)
    source_output = reference(source_input)
    target_output = clifra(target_input)
    torch.testing.assert_close(
        output_map.inverse().apply(target_output), source_output, rtol=rtol, atol=atol
    )
    source_output.square().mean().backward()
    target_output.square().mean().backward()
    torch.testing.assert_close(
        input_map.inverse().apply(target_input.grad), source_input.grad, rtol=rtol, atol=atol
    )
    for name, source, target, blade, mapping in pairs:
        if source.grad is None or target.grad is None:
            raise AssertionError(f"missing gradient: {name}")
        if not torch.isfinite(source.grad).all() or not torch.isfinite(target.grad).all():
            raise AssertionError(f"nonfinite gradient: {name}")
        expected = mapped_parameter(source.grad, blade, mapping)
        actual = target.grad[blade] if blade is not None else target.grad
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=name)
    reference.zero_grad(set_to_none=True)
    clifra.zero_grad(set_to_none=True)
    return len(pairs)


def check_layers(reference, clifra, device: torch.device) -> None:
    generator = torch.Generator().manual_seed(3307)
    boundary = torch.randn(1, reference.encoder.in_channels, 5, 5, 5, 6, generator=generator).to(
        device
    )
    modes = reference.layers[0][0].fourier.modes1
    size = max(18, 2 * modes + 2)
    internal = torch.randn(
        1, reference.encoder.out_channels, size, size, size, 8, generator=generator
    ).to(device)
    torch.testing.assert_close(
        unpair_fourier(pair_fourier(FULL_MAP.apply(internal))),
        FULL_MAP.apply(internal),
        rtol=0,
        atol=0,
    )
    print("3D Fourier pair/unpair: PASS")
    source_spectral = reference.layers[0][0].fourier
    target_spectral = clifra.layers[0][0].fourier
    crop_only = ReferenceSpectral(
        [1, 1, 1], internal.shape[1], internal.shape[1], modes, modes, modes, multiply=False
    ).to(device)
    identity = copy.deepcopy(target_spectral)
    with torch.no_grad():
        identity.weights.zero_()
        for channel in range(internal.shape[1]):
            identity.weights[0, channel, channel].fill_(1)
    torch.testing.assert_close(
        FULL_MAP.inverse().apply(identity(FULL_MAP.apply(internal))),
        crop_only(internal),
        rtol=2e-3,
        atol=2e-4,
    )
    print("3D spectral mode selection: PASS")
    _, source_kernel = get_3d_clifford_kernel(source_spectral.weights, source_spectral.g)
    basis = product_basis_3d((0, 1, 2, 3), (0, 1, 2, 3)).to(device)
    target_kernel = torch.einsum("koidhw,bka->aobidhw", target_spectral.weights, basis)
    target_kernel = target_kernel.reshape(
        8 * target_spectral.out_channels,
        8 * target_spectral.in_channels,
        2 * modes,
        2 * modes,
        2 * modes,
    )
    permutation = FULL_MAP.target_for_source
    source_kernel = source_kernel.reshape(
        8,
        source_spectral.out_channels,
        8,
        source_spectral.in_channels,
        2 * modes,
        2 * modes,
        2 * modes,
    )
    source_kernel = source_kernel[list(permutation), :, :, :, :, :, :][:, :, list(permutation)]
    # The permutation is self-inverse for this 3D basis order.
    torch.testing.assert_close(
        target_kernel, source_kernel.reshape_as(target_kernel), rtol=0, atol=0
    )
    print("Fourier-space geometric-product kernel: PASS")
    for label, source, target, values, input_map, output_map in (
        ("Maxwell encoder", reference.encoder, clifra.encoder, boundary, BOUNDARY_MAP, FULL_MAP),
        ("Maxwell decoder", reference.decoder, clifra.decoder, internal, FULL_MAP, BOUNDARY_MAP),
        (
            "CliffordConv3d",
            reference.layers[0][0].conv,
            clifra.layers[0][0].conv,
            internal,
            FULL_MAP,
            FULL_MAP,
        ),
        ("spectral convolution", source_spectral, target_spectral, internal, FULL_MAP, FULL_MAP),
        (
            "Fourier block",
            reference.layers[0][0],
            clifra.layers[0][0],
            internal,
            FULL_MAP,
            FULL_MAP,
        ),
    ):
        torch.testing.assert_close(
            output_map.inverse().apply(target(input_map.apply(values))),
            source(values),
            rtol=2e-3,
            atol=2e-4,
            msg=label,
        )
        print(f"{label} equivalence: PASS")
    count = assert_corresponding_gradients(
        source_spectral, target_spectral, internal, FULL_MAP, FULL_MAP
    )
    print(f"spectral input and {count} parameter gradients: PASS")


def prepare(args, device: torch.device) -> PreparedBenchmark:
    options = dict(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        hidden_channels=args.hidden_channels,
        blocks=args.blocks,
        modes=args.modes,
    )
    reference = ReferenceFNO3d(**options)
    clifra = ClifraFNO3d(**options)
    transfer(reference, clifra)
    reference, clifra = reference.to(device), clifra.to(device)
    if parameter_count(reference) != parameter_count(clifra):
        raise AssertionError("parameter counts differ")
    generator = torch.Generator().manual_seed(args.seed + 3301)
    field = torch.randn(
        args.batch, args.in_channels, args.grid, args.grid, args.grid, 6, generator=generator
    ).to(device)
    clifra_field = BOUNDARY_MAP.apply(field)
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
            (clifra_field.detach().clone().requires_grad_(True),),
            loss=lambda out: out.square().mean(),
        ),
    }
    selected = ["reference", "clifra"] if args.backend == "both" else [args.backend]

    def validate(outputs):
        check_convention(field)
        print(f"parameter correspondence: PASS ({parameter_count(reference):,} scalars)")
        for name, output in outputs.items():
            if output.shape != (args.batch, args.out_channels, args.grid, args.grid, args.grid, 6):
                raise AssertionError(f"{name}: unexpected output shape {tuple(output.shape)}")
        if len(outputs) == 2:
            torch.testing.assert_close(
                BOUNDARY_MAP.inverse().apply(outputs["clifra"]),
                outputs["reference"],
                rtol=2e-3,
                atol=2e-4,
            )
            check_layers(reference, clifra, device)
            count = assert_corresponding_gradients(
                reference, clifra, field, BOUNDARY_MAP, BOUNDARY_MAP
            )
            print(f"whole-model output, input and {count} parameter gradients: PASS")

    return PreparedBenchmark(
        calls=[calls[name] for name in selected],
        work_units={"samples": args.batch, "grid points": args.batch * args.grid**3},
        validate=validate,
        description=(
            f"clifford_fno3d g=(3,0), batch={args.batch}, grid={args.grid}^3, "
            f"channels={args.in_channels}/{args.out_channels}, hidden={args.hidden_channels}, blocks={args.blocks}, "
            f"modes=({args.modes},{args.modes},{args.modes}), norm=False, padding=2, boundary=vector+bivector"
        ),
        reference_source=f"Brandstetter et al., Clifford Neural Layers for PDE Modeling; Microsoft cliffordlayers {SOURCE_COMMIT}",
        provenance_packages=("cliffordlayers",),
    )
