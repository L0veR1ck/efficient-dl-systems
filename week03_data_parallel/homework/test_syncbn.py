import os

import pytest
import torch
import torch.distributed as dist

from syncbn import SyncBatchNorm


def worker_process(rank, world_size, hid_dim, batch_size, input_data_list, output_queue):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    try:
        dist.init_process_group(
            backend="gloo", 
            rank=rank, 
            world_size=world_size,
        )
        
        input_data = input_data_list[rank].detach().clone()
        input_data.requires_grad = True
        
        syncbn = SyncBatchNorm(num_features=hid_dim)
        output = syncbn(input_data)
        
        per_worker_batch = batch_size // world_size
        samples_to_use = max(0, min(per_worker_batch, batch_size // 2 - rank * per_worker_batch))
        
        loss = output[:samples_to_use].sum()
        loss.backward()
        
        output_queue.put({
            "rank": rank,
            "output": output.detach().clone(),
            "grad_input": input_data.grad.clone() if input_data.grad is not None else torch.zeros_like(input_data)
        })
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("num_workers", [1, 4])
@pytest.mark.parametrize("hid_dim", [128, 256, 512, 1024])
@pytest.mark.parametrize("batch_size", [32, 64])
def test_batchnorm(num_workers, hid_dim, batch_size):
    ctx = torch.multiprocessing.get_context("spawn")
    
    torch.manual_seed(42)
    full_input = torch.randn(batch_size, hid_dim, 1, dtype=torch.float32)
    full_input.requires_grad = True
    
    bn = torch.nn.BatchNorm1d(hid_dim, affine=False, track_running_stats=True)
    expected_output = bn(full_input)
    
    loss = expected_output[:batch_size // 2].sum()
    loss.backward()
    expected_grad = full_input.grad.clone()
    
    output_queue = ctx.Manager().Queue()
    
    per_worker_batch = batch_size // num_workers
    input_data_list = [
        full_input[i*per_worker_batch:(i+1)*per_worker_batch].detach()
        for i in range(num_workers)
    ]
    
    processes = []
    for rank in range(num_workers):
        p = ctx.Process(
            target=worker_process,
            args=(rank, num_workers, hid_dim, batch_size, input_data_list, output_queue),
        )
        p.start()
        processes.append(p)
    
    results = [output_queue.get() for _ in range(num_workers)]
    results.sort(key=lambda x: x["rank"])

    actual_output = torch.cat([r["output"] for r in results], dim=0)
    actual_grad = torch.cat([r["grad_input"] for r in results], dim=0)
    
    assert torch.allclose(actual_output, expected_output, atol=1e-3, rtol=0)
    assert torch.allclose(actual_grad, expected_grad, atol=1e-3, rtol=0)
