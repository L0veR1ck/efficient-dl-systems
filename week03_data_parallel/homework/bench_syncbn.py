import argparse
import time
import torch
import torch.distributed as dist

from syncbn import SyncBatchNorm


def benchmark_syncbn(rank, world_size, hid_dim, batch_size, num_iterations=100, warmup=10):
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    
    per_worker_batch = batch_size // world_size
    input_data = torch.randn(per_worker_batch, hid_dim, 1, dtype=torch.float32, device=device)
    input_data.requires_grad = True
    
    syncbn = SyncBatchNorm(num_features=hid_dim).to(device)
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    for _ in range(warmup):
        input_data_copy = input_data.clone().detach().requires_grad_(True)
        output = syncbn(input_data_copy)
        loss = output[:batch_size // (2 * world_size)].sum()
        loss.backward()
        torch.cuda.synchronize(device)
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start_mem = torch.cuda.memory_allocated(device)
    
    start_time = time.time()
    
    for _ in range(num_iterations):
        input_data_copy = input_data.clone().detach().requires_grad_(True)
        output = syncbn(input_data_copy)
        loss = output[:batch_size // (2 * world_size)].sum()
        loss.backward()
        torch.cuda.synchronize(device)
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations * 1000
    
    peak_mem = torch.cuda.max_memory_allocated(device)
    mem_usage = (peak_mem - start_mem) / (1024 ** 2)
    
    return avg_time, mem_usage


def benchmark_torch_syncbn(rank, world_size, hid_dim, batch_size, num_iterations=100, warmup=10):
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    
    per_worker_batch = batch_size // world_size
    input_data = torch.randn(per_worker_batch, hid_dim, 1, dtype=torch.float32, device=device)
    input_data.requires_grad = True
    
    syncbn = torch.nn.SyncBatchNorm(num_features=hid_dim, affine=False, track_running_stats=True).to(device)
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    
    for _ in range(warmup):
        input_data_copy = input_data.clone().detach().requires_grad_(True)
        output = syncbn(input_data_copy)
        loss = output[:batch_size // (2 * world_size)].sum()
        loss.backward()
        torch.cuda.synchronize(device)
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start_mem = torch.cuda.memory_allocated(device)
    
    start_time = time.time()
    
    for _ in range(num_iterations):
        input_data_copy = input_data.clone().detach().requires_grad_(True)
        output = syncbn(input_data_copy)
        loss = output[:batch_size // (2 * world_size)].sum()
        loss.backward()
        torch.cuda.synchronize(device)
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations * 1000
    
    peak_mem = torch.cuda.max_memory_allocated(device)
    mem_usage = (peak_mem - start_mem) / (1024 ** 2)
    
    return avg_time, mem_usage


def main():
    assert torch.cuda.is_available(), "no gpu found"

    parser = argparse.ArgumentParser()
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=250)
    args = parser.parse_args()
    
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    custom_time, custom_mem = benchmark_syncbn(
        rank, 
        world_size, 
        args.hid_dim, 
        args.batch_size, 
        args.num_iterations, 
        args.warmup,
    )
    
    torch_time, torch_mem = benchmark_torch_syncbn(
        rank, 
        world_size, 
        args.hid_dim, 
        args.batch_size,
        args.num_iterations, 
        args.warmup,
    )
    
    if rank == 0:
        print(f"Custom SyncBatchNorm: {custom_time:.3f} ms | {custom_mem:.2f} MB")
        print(f"PyTorch SyncBatchNorm: {torch_time:.3f} ms | {torch_mem:.2f} MB")
        print(f"Speedup: {torch_time / custom_time:.2f}x")
        print(f"Memory reduction: {(torch_mem - custom_mem) / torch_mem * 100:.1f}%")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

