import time

import torch
from constants import MODEL_DIR
from create_model import create_model


def main() -> None:
    device = "cuda"

    model = create_model(MODEL_DIR)
    model = model.to(device)
    model.eval()

    batch_size, seq_len = 128, 1024
    micro_batch_size = 8
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    print(f"forward: ({batch_size}, {seq_len}), {micro_batch_size=}")

    with torch.no_grad():
        model(input_ids[:1, :32])

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    logits_parts = []
    with torch.no_grad():
        for i in range(0, batch_size, micro_batch_size):
            micro = input_ids[i : i + micro_batch_size]
            out = model(micro).logits[:, -1, :]
            logits_parts.append(out.cpu())

    torch.cuda.synchronize()
    duration = time.time() - t0

    total_tokens = batch_size * seq_len

    print(f"{duration:.2f}s ({total_tokens / duration:.0f} tokens/s)")
    print(f"max memory allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f}GB")


if __name__ == "__main__":
    main()
