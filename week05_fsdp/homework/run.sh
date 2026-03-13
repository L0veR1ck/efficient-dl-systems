CUDA_VISIBLE_DEVICES=5,6 \
torchrun --nproc-per-node 2 train.py \
    --model 1b  \
    --fsdp effdl \
    --traces-dir artifacts/task3/effdl_traces \
    --snapshots-dir artifacts/task3/effdl_snapshots