"""Prepared native calls and generic tensor-tree checks."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch


def tensors(value: Any):
    """Visit tensors in nested call arguments or outputs."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from tensors(child)


def assert_close_tree(actual: Any, expected: Any, *, rtol: float, atol: float) -> None:
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    elif isinstance(actual, Mapping) and isinstance(expected, Mapping):
        if actual.keys() != expected.keys():
            raise AssertionError("output mapping keys differ")
        for key in actual:
            assert_close_tree(actual[key], expected[key], rtol=rtol, atol=atol)
    elif isinstance(actual, (tuple, list)) and type(actual) is type(expected):
        if len(actual) != len(expected):
            raise AssertionError("output sequence lengths differ")
        for left, right in zip(actual, expected):
            assert_close_tree(left, right, rtol=rtol, atol=atol)
    elif actual != expected:
        raise AssertionError("non-tensor outputs differ")


def detach_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, Mapping):
        return {key: detach_tree(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_tree(child) for child in value)
    if isinstance(value, list):
        return [detach_tree(child) for child in value]
    return value


@dataclass
class PreparedCall:
    name: str
    model: torch.nn.Module
    args: tuple[Any, ...]
    loss: Callable[[Any], torch.Tensor]
    kwargs: dict[str, Any] = field(default_factory=dict)

    def forward(self) -> Any:
        return self.model(*self.args, **self.kwargs)

    def reset_grad(self) -> None:
        self.model.zero_grad(set_to_none=True)
        for tensor in tensors((self.args, self.kwargs)):
            if tensor.grad is not None:
                tensor.grad = None


@dataclass
class PreparedBenchmark:
    calls: list[PreparedCall]
    work_units: dict[str, float]
    validate: Callable[[dict[str, Any]], None]
    description: str
    reference_source: str = ""
    provenance_packages: tuple[str, ...] = ()


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def check_finite(call: PreparedCall) -> Any:
    call.reset_grad()
    output = call.forward()
    output_tensors = list(tensors(output))
    if not output_tensors or any(not torch.isfinite(value).all() for value in output_tensors):
        raise AssertionError(f"{call.name}: missing or non-finite forward tensors")
    loss = call.loss(output)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise AssertionError(f"{call.name}: loss must be a finite scalar")
    loss.backward()
    for value in tensors((call.args, call.kwargs)):
        if value.requires_grad and (value.grad is None or not torch.isfinite(value.grad).all()):
            raise AssertionError(f"{call.name}: missing or non-finite input gradient")
    for name, parameter in call.model.named_parameters():
        if parameter.requires_grad and (
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
        ):
            raise AssertionError(f"{call.name}: missing or non-finite gradient for {name}")
    call.reset_grad()
    return detach_tree(output)
