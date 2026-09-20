"""Small coefficient-basis isomorphism used during benchmark preparation."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SignedBasisTransform:
    """Map source lanes to target lanes with signs, preserving coefficient values."""

    target_for_source: tuple[int, ...]
    sign_for_source: tuple[float, ...]

    def __post_init__(self) -> None:
        size = len(self.target_for_source)
        if sorted(self.target_for_source) != list(range(size)):
            raise ValueError("basis mapping must be a permutation")
        if len(self.sign_for_source) != size or any(
            abs(sign) != 1 for sign in self.sign_for_source
        ):
            raise ValueError("basis signs must be +1 or -1 for every lane")

    def inverse(self) -> "SignedBasisTransform":
        source_for_target = [0] * len(self.target_for_source)
        signs = [0.0] * len(self.target_for_source)
        for source, target in enumerate(self.target_for_source):
            source_for_target[target] = source
            signs[target] = self.sign_for_source[source]
        return SignedBasisTransform(tuple(source_for_target), tuple(signs))

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        """Convert the trailing coefficient axis from source to target order."""
        if values.shape[-1] != len(self.target_for_source):
            raise ValueError("coefficient axis has the wrong size")
        inverse = self.inverse()
        indices = torch.tensor(inverse.target_for_source, device=values.device)
        signs = values.new_tensor(inverse.sign_for_source)
        return torch.index_select(values, -1, indices) * signs
