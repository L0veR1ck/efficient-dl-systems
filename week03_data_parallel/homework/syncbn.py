import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, running_mean, running_std, eps: float, momentum: float, training: bool):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`

        n, c, l = input.shape
        cnt = n * l

        if training:
            local_sum = input.sum(axis=(0, 2))
            local_sum_sq = (input ** 2).sum(axis=(0, 2))
            local_count = torch.tensor([cnt], device=input.device, dtype=input.dtype)

            stats = torch.cat([local_sum, local_sum_sq, local_count], dim=0)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(stats)

            global_sum = stats[:c]
            global_sum_sq = stats[c : 2 * c]
            global_count = stats[-1]

            mean = global_sum / global_count
            var = global_sum_sq / global_count - mean * mean

            if global_count > 1:
                unbiased_var = var * global_count / (global_count - 1)
            else:
                unbiased_var = var

            running_mean.mul_(1 - momentum).add_(mean, alpha=momentum)
            running_std.mul_(1 - momentum).add_(unbiased_var, alpha=momentum)
            count_for_backward = local_count
        else:
            mean = running_mean
            var = running_std
            count_for_backward = torch.tensor([cnt], device=input.device, dtype=input.dtype)

        forward_centered_inputs = input - mean[None, :, None]
        forward_inverse_std = 1 / torch.sqrt(eps + var[None, :, None])
        forward_normalized_inputs = forward_centered_inputs * forward_inverse_std

        ctx.save_for_backward(forward_normalized_inputs, forward_inverse_std, count_for_backward)
        ctx.training = training
        
        return forward_normalized_inputs

    @staticmethod
    def backward(ctx, grad_output):
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!
        forward_normalized_inputs, forward_inverse_std, local_count = ctx.saved_tensors

        if not ctx.training:
            grad_input = grad_output * forward_inverse_std
            return grad_input, *(None,) * 5

        c = grad_output.shape[1]
        local_grad_sum = grad_output.sum(axis=(0, 2))
        local_grad_norm_sum = (grad_output * forward_normalized_inputs).sum(axis=(0, 2))
        stats = torch.cat([local_grad_sum, local_grad_norm_sum, local_count], dim=0)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats)

        grad_mean = stats[:c] / stats[-1]
        grad_norm_mean = stats[c : 2 * c] / stats[-1]

        grad_input = forward_inverse_std * (
            grad_output
            - grad_mean[None, :, None]
            - forward_normalized_inputs * grad_norm_mean[None, :, None]
        )

        return grad_input, *(None,) * 5


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        # your code here

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # You will probably need to use `sync_batch_norm` from above
        return sync_batch_norm.apply(
            input, 
            self.running_mean, 
            self.running_var, 
            self.eps, 
            self.momentum,
            self.training,
        )
