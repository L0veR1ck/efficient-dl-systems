import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard


dist.init_process_group(backend="gloo")
rank = dist.get_rank()
world_size = dist.get_world_size()

mesh = init_device_mesh("cpu", (world_size,))

param = torch.arange(8, dtype=torch.float32).reshape(4, 2)  # shape (4, 2)

shard_size = param.size(0) // world_size
sharded_param = param[rank * shard_size : (rank + 1) * shard_size]  # каждый берёт свой кусок

print(f"rank={rank}, my shard: {sharded_param}")

dt = DTensor.from_local(
    sharded_param,
    mesh,
    [Shard(0)],
    shape=param.shape,     
    stride=param.stride(),
)

full = dt.full_tensor()
print(f"rank={rank}, full_tensor: {full}")
