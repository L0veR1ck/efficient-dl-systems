from typing import Optional

import torch
from prefetching import WeightStore


class _OffloadedLinearOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        param_name: str,
        bias: Optional[torch.Tensor],
        weight_store: WeightStore,
    ) -> torch.Tensor:
        with torch.no_grad():
            weight = weight_store.get(param_name, device=str(input.device))

        ctx._param_name = param_name
        ctx._weight_store = weight_store
        ctx._has_bias = bias is not None
        ctx._device = input.device

        return torch.nn.functional.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        with torch.no_grad():
            weight = ctx._weight_store.get(ctx._param_name, device=str(ctx._device))

        weight = weight.to(grad_output.dtype)
        grad_input = torch.nn.functional.linear(grad_output, weight.t())
        grad_bias = grad_output.flatten(0, -2).sum(0) if ctx._has_bias else None

        return grad_input, None, grad_bias, None


class MyOffloadedLinear(torch.nn.Linear):
    def __init__(
        self,
        param_name: str,
        bias: Optional[torch.Tensor],
        weight_store: WeightStore,
    ) -> None:
        torch.nn.Module.__init__(self)

        self.param_name = param_name
        self.weight_store = weight_store

        w = weight_store.get(param_name)
        self.in_features = w.shape[1]
        self.out_features = w.shape[0]

        self.register_buffer("weight", torch.empty(0, dtype=w.dtype), persistent=False)
        self.bias = torch.nn.Parameter(bias) if bias is not None else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _OffloadedLinearOp.apply(
            input,
            self.param_name,
            self.bias,
            self.weight_store,
        )
