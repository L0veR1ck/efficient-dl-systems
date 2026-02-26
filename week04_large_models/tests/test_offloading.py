import torch
from offloading import MyOffloadedLinear
from prefetching import WeightStore
from safetensors.torch import save_file


def test_offloaded_linear(tmp_path):
    torch.manual_seed(0)
    linear = torch.nn.Linear(64, 32)

    param_name = "test.weight"
    weight_path = f"{tmp_path}/weights.safetensors"
    save_file({param_name: linear.weight.detach()}, weight_path)

    store = WeightStore({param_name: weight_path}, weights_dir=str(tmp_path))
    bias = linear.bias.detach().clone().requires_grad_(True)

    offloaded = MyOffloadedLinear(
        param_name=param_name,
        bias=bias,
        weight_store=store,
    )

    x = torch.randn(4, 64, requires_grad=True)
    x_clone = x.detach().clone().requires_grad_(True)

    expected_out = linear(x)
    actual_out = offloaded(x_clone)

    assert torch.allclose(expected_out, actual_out)

    expected_out.sum().backward()
    actual_out.sum().backward()

    assert torch.allclose(x.grad, x_clone.grad)
    assert torch.allclose(linear.bias.grad, offloaded.bias.grad)
    assert torch.sum(x.grad ** 2) > 0
    assert torch.sum(linear.bias.grad ** 2) > 0
