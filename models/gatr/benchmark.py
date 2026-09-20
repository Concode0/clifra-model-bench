"""Native input preparation and correctness checks for full GATr."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

from bench.basis import SignedBasisTransform
from bench.core import PreparedBenchmark, PreparedCall

from .clifra import ClifraGATr
from .reference import make_reference


@dataclass(frozen=True)
class Configuration:
    in_mv_channels: int
    out_mv_channels: int
    hidden_mv_channels: int
    in_s_channels: int
    out_s_channels: int
    hidden_s_channels: int
    blocks: int
    heads: int


def add_arguments(parser) -> None:
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--items", type=int, default=8)
    parser.add_argument("--in-mv-channels", type=int, default=2)
    parser.add_argument("--out-mv-channels", type=int, default=2)
    parser.add_argument("--hidden-mv-channels", type=int, default=4)
    parser.add_argument("--in-s-channels", type=int, default=2)
    parser.add_argument("--out-s-channels", type=int, default=2)
    parser.add_argument("--hidden-s-channels", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)


def check_arguments(parser, args) -> None:
    for name in (
        "batch",
        "items",
        "in_mv_channels",
        "out_mv_channels",
        "hidden_mv_channels",
        "in_s_channels",
        "out_s_channels",
        "hidden_s_channels",
        "blocks",
        "heads",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")


def convention() -> SignedBasisTransform:
    """Official null-first shortlex PGA to Clifra's Euclidean-first binary basis."""
    targets, signs = [], []
    for grade in range(5):
        for source_generators in itertools.combinations(range(4), grade):
            target_generators = [
                3 if generator == 0 else generator - 1 for generator in source_generators
            ]
            inversions = sum(
                left > right
                for i, left in enumerate(target_generators)
                for right in target_generators[i + 1 :]
            )
            targets.append(sum(1 << generator for generator in target_generators))
            signs.append(-1.0 if inversions % 2 else 1.0)
    return SignedBasisTransform(tuple(targets), tuple(signs))


def transfer(reference, clifra) -> dict[str, tuple[torch.nn.Parameter, torch.nn.Parameter]]:
    left = dict(reference.named_parameters())
    right = dict(clifra.named_parameters())
    if left.keys() != right.keys():
        raise AssertionError(f"parameter names differ: {left.keys() ^ right.keys()}")
    for name in left:
        if left[name].shape != right[name].shape:
            raise AssertionError(f"{name}: parameter shapes differ")
    with torch.no_grad():
        for name in left:
            right[name].copy_(left[name])
    return {name: (left[name], right[name]) for name in left}


def _loss(outputs):
    mv, scalars = outputs
    return mv.square().mean() + scalars.square().mean()


def _close(actual, expected, label, *, rtol=1e-3, atol=1e-4):
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as error:
        raise AssertionError(f"{label}: {error}") from error


def _check_primitives(reference, clifra, transform, device):
    """Compare the algebra and the first block's learned components in isolation."""
    from gatr.primitives import equivariant_join, geometric_product, outer_product
    from gatr.primitives.dual import dual
    from gatr.primitives.linear import _compute_pin_equi_linear_basis

    st = clifra.structure
    basis = _compute_pin_equi_linear_basis(device, torch.float32, True)
    identity = torch.eye(16, device=device)
    # Basis-lane correspondence includes all nine learned linear coefficients.
    for lane in range(9):
        source_action = torch.einsum("ij,bj->bi", basis[lane], transform.inverse().apply(identity))
        target_action = torch.einsum("ij,bj->bi", st.linear_basis[lane], identity)
        _close(
            transform.apply(source_action),
            target_action,
            f"EquiLinear basis {lane}",
            rtol=0,
            atol=0,
        )

    generator = torch.Generator(device="cpu").manual_seed(1928)
    x = torch.randn(2, 3, reference.linear_in._in_mv_channels, 16, generator=generator).to(device)
    y = torch.randn(2, 3, reference.linear_in._in_mv_channels, 16, generator=generator).to(device)
    reference_mv = torch.randn(2, 1, 1, 16, generator=generator).to(device)
    cx, cy, cr = map(transform.apply, (x, y, reference_mv))
    _close(st.gp(cx, cy), transform.apply(geometric_product(x, y)), "geometric product")
    _close(st.wedge(cx, cy), transform.apply(outer_product(x, y)), "outer product")
    _close(st.dual(cx), transform.apply(dual(x)), "PGA dual", rtol=0, atol=0)
    _close(
        st.join(cx, cy, cr),
        transform.apply(equivariant_join(x, y, reference_mv)),
        "equivariant join",
    )

    rs = torch.randn(2, 3, reference.linear_in.s2mvs.in_features, generator=generator).to(device)
    rmv, rs_out = reference.linear_in(x, rs)
    cmv, cs_out = clifra.linear_in(cx, rs)
    _close(cmv, transform.apply(rmv), "EquiLinear MV")
    _close(cs_out, rs_out, "EquiLinear scalar")

    block_r, block_c = reference.blocks[0], clifra.blocks[0]
    hidden_r, hidden_s_r = reference.linear_in(x, rs)
    hidden_c, hidden_s_c = clifra.linear_in(cx, rs)
    norm_r = block_r.norm(hidden_r, hidden_s_r)
    norm_c = block_c.norm(hidden_c, hidden_s_c)
    _close(norm_c[0], transform.apply(norm_r[0]), "EquiLayerNorm MV")
    _close(norm_c[1], norm_r[1], "EquiLayerNorm scalar")

    qkv_r = block_r.attention.qkv_module(norm_r[0], norm_r[1])
    qkv_c = block_c.attention.qkv_module(*norm_c)
    for lane, (left, right) in enumerate(zip(qkv_r, qkv_c)):
        _close(right, transform.apply(left) if lane < 3 else left, f"QKV {lane}")

    # Observe the official Q/K/V tensors at the common attention-kernel boundary.
    import gatr.primitives.attention as official_attention

    captured = []
    original = official_attention.scaled_dot_product_attention

    def observe(q, k, v, attn_mask=None):
        captured.extend((q.detach(), k.detach(), v.detach()))
        return original(q, k, v, attn_mask=attn_mask)

    try:
        official_attention.scaled_dot_product_attention = observe
        attn_r = block_r.attention.attention(*qkv_r)
    finally:
        official_attention.scaled_dot_product_attention = original
    q, k, v = block_c.attention.attention.features(*qkv_c)
    _close(q, captured[0], "attention Q features")
    _close(k.expand_as(captured[1]), captured[1], "attention K features")
    # V is in native coefficient order; the scalar and padding regions are identical.
    v_reference = captured[2]
    mv_width = qkv_r[2].shape[-2] * 16
    v_mv = transform.apply(v_reference[..., :mv_width].reshape(*v_reference.shape[:-1], -1, 16))
    v_expected = torch.cat((v_mv.flatten(-2), v_reference[..., mv_width:]), dim=-1)
    _close(v.expand_as(v_expected), v_expected, "attention V features")
    attn_c = block_c.attention.attention(*qkv_c)
    _close(attn_c[0], transform.apply(attn_r[0]), "geometric attention MV")
    _close(attn_c[1], attn_r[1], "geometric attention scalar")

    ref = x.mean(dim=(1, 2), keepdim=True)
    cref = transform.apply(ref)
    bilinear_r = block_r.mlp.layers[0](hidden_r, reference_mv=ref, scalars=hidden_s_r)
    bilinear_c = block_c.mlp.layers[0](hidden_c, cref, hidden_s_c)
    _close(bilinear_c[0], transform.apply(bilinear_r[0]), "GeometricBilinear MV")
    _close(bilinear_c[1], bilinear_r[1], "GeometricBilinear scalar")
    nonlinear_r = block_r.mlp.layers[1](*bilinear_r)
    nonlinear_c = block_c.mlp.layers[1](*bilinear_c)
    _close(nonlinear_c[0], transform.apply(nonlinear_r[0]), "ScalarGated MV")
    _close(nonlinear_c[1], nonlinear_r[1], "ScalarGated scalar")
    mlp_r = block_r.mlp(hidden_r, hidden_s_r, ref)
    mlp_c = block_c.mlp(hidden_c, hidden_s_c, cref)
    _close(mlp_c[0], transform.apply(mlp_r[0]), "GeoMLP MV")
    _close(mlp_c[1], mlp_r[1], "GeoMLP scalar")
    block_out_r = block_r(hidden_r, hidden_s_r, ref)
    block_out_c = block_c(hidden_c, hidden_s_c, cref)
    _close(block_out_c[0], transform.apply(block_out_r[0]), "GATrBlock MV")
    _close(block_out_c[1], block_out_r[1], "GATrBlock scalar")
    print("primitive, Q/K/V, MLP, and one-block equivalence: PASS")


def _check_gradients(reference, clifra, transform, input_mv, input_s, mapping):
    mv_r = input_mv.detach().clone().requires_grad_(True)
    s_r = input_s.detach().clone().requires_grad_(True)
    mv_c = transform.apply(input_mv).detach().requires_grad_(True)
    s_c = input_s.detach().clone().requires_grad_(True)
    reference.zero_grad(set_to_none=True)
    clifra.zero_grad(set_to_none=True)
    _loss(reference(mv_r, s_r)).backward()
    _loss(clifra(mv_c, s_c)).backward()
    _close(mv_c.grad, transform.apply(mv_r.grad), "MV input gradient")
    _close(s_c.grad, s_r.grad, "scalar input gradient")
    for name, (left, right) in mapping.items():
        if left.grad is None or right.grad is None:
            raise AssertionError(f"{name}: missing mapped gradient")
        _close(right.grad, left.grad, f"parameter gradient {name}")
    print(f"MV/scalar input and all {len(mapping)} parameter gradients: PASS")
    reference.zero_grad(set_to_none=True)
    clifra.zero_grad(set_to_none=True)


def _check_equivariance(reference, clifra, transform, input_mv, input_s):
    from gatr.interface import embed_reflection, embed_rotation, embed_translation
    from gatr.primitives import geometric_product
    from gatr.primitives.linear import grade_involute, reverse

    # Valid even Pin elements: unit Euclidean rotors and unit translators.
    generator = torch.Generator(device="cpu").manual_seed(771)
    motors = []
    for _ in range(2):
        quaternion = torch.randn(4, generator=generator, device="cpu").to(input_mv.device)
        quaternion = quaternion / quaternion.norm()
        motors.append(("rotation", embed_rotation(quaternion), False))
        translation = torch.randn(3, generator=generator, device="cpu").to(input_mv.device) * 0.3
        motors.append(("translation", embed_translation(translation), False))
    normal = torch.randn(3, generator=generator).to(input_mv.device)
    normal = normal / normal.norm()
    position = torch.randn(3, generator=generator).to(input_mv.device) * 0.2
    motors.append(("reflection", embed_reflection(normal, position), True))

    def action(values, motor, odd):
        middle = grade_involute(values) if odd else values
        return geometric_product(geometric_product(motor, middle), reverse(motor))

    with torch.no_grad():
        for name, motor, odd in motors:
            transformed = action(input_mv, motor, odd)
            for label, model, values, converted in (
                ("reference", reference, input_mv, False),
                ("clifra", clifra, transform.apply(input_mv), True),
            ):
                before_mv, before_s = model(values, input_s)
                after_mv, after_s = model(
                    transform.apply(transformed) if converted else transformed, input_s
                )
                expected_mv = (
                    transform.apply(action(transform.inverse().apply(before_mv), motor, odd))
                    if converted
                    else action(before_mv, motor, odd)
                )
                _close(
                    after_mv, expected_mv, f"{label} {name} MV equivariance", rtol=3e-3, atol=3e-4
                )
                _close(after_s, before_s, f"{label} {name} scalar invariance", rtol=3e-3, atol=3e-4)
    print("rotations, translations, reflection: MV equivariance, scalar invariance PASS")


def prepare(args, device: torch.device) -> PreparedBenchmark:
    from gatr.interface import embed_point
    from gatr.utils.einsum import enable_cached_einsum

    # Official GATr documents this setting for torch.compile. It selects the
    # same tensor contraction with PyTorch's compiler-supported einsum path.
    if args.compile:
        enable_cached_einsum(False)
    config = Configuration(
        **{name: getattr(args, name) for name in Configuration.__dataclass_fields__}
    )
    reference = make_reference(config)
    clifra = ClifraGATr(config)
    transform = convention()
    mapping = transfer(reference, clifra)
    reference, clifra = reference.to(device), clifra.to(device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 2381)
    points = torch.randn(args.batch, args.items, args.in_mv_channels, 3, generator=generator).to(
        device
    )
    input_mv = embed_point(points)
    input_s = torch.randn(args.batch, args.items, args.in_s_channels, generator=generator).to(
        device
    )
    clifra_mv = transform.apply(input_mv)
    calls = {
        "reference": PreparedCall(
            "reference",
            reference,
            (input_mv.detach().requires_grad_(True), input_s.detach().requires_grad_(True)),
            loss=_loss,
        ),
        "clifra": PreparedCall(
            "clifra",
            clifra,
            (
                clifra_mv.detach().requires_grad_(True),
                input_s.detach().clone().requires_grad_(True),
            ),
            loss=_loss,
        ),
    }
    selected = ("reference", "clifra") if args.backend == "both" else (args.backend,)

    def validate(outputs):
        _close(
            transform.inverse().apply(transform.apply(input_mv)),
            input_mv,
            "convention roundtrip",
            rtol=0,
            atol=0,
        )
        print("convention roundtrip: PASS")
        for label, output in outputs.items():
            if output[0].shape != (args.batch, args.items, args.out_mv_channels, 16):
                raise AssertionError(f"{label}: wrong multivector output shape")
            if output[1].shape != (args.batch, args.items, args.out_s_channels):
                raise AssertionError(f"{label}: wrong scalar output shape")
        if len(outputs) == 2:
            _close(outputs["clifra"][0], transform.apply(outputs["reference"][0]), "full GATr MV")
            _close(outputs["clifra"][1], outputs["reference"][1], "full GATr scalar")
            print("full-model MV/scalar output equivalence: PASS")
            _check_primitives(reference, clifra, transform, device)
            _check_gradients(reference, clifra, transform, input_mv, input_s, mapping)
        _check_equivariance(reference, clifra, transform, input_mv, input_s)

    import gatr.primitives.attention as attention

    if attention.FORCE_XFORMERS:
        raise RuntimeError(
            "GATr has FORCE_XFORMERS enabled; this benchmark requires common PyTorch SDPA"
        )
    print("attention backend: PyTorch SDPA (reference and Clifra)")
    return PreparedBenchmark(
        calls=[calls[name] for name in selected],
        work_units={
            "samples": args.batch,
            "tokens": args.batch * args.items,
            "output MV": args.batch * args.items * args.out_mv_channels,
        },
        validate=validate,
        description=(
            f"gatr batch={args.batch}, items={args.items}, MV={args.in_mv_channels}/{args.hidden_mv_channels}/{args.out_mv_channels}, "
            f"scalar={args.in_s_channels}/{args.hidden_s_channels}/{args.out_s_channels}, blocks={args.blocks}, heads={args.heads}, attention=PyTorch SDPA"
        ),
        reference_source="Qualcomm GATr commit 6afc26f26b8fcf51136ae8c1d264a36e14b6e497 (installed official package; changelog 1.4.3, package __version__ 1.4.2)",
        provenance_packages=("gatr", "einops", "opt-einsum", "xformers"),
    )
