import time

import torch
import transformers
from constants import MODEL_DIR
from create_model import create_model
from datasets import load_dataset


MICRO_BATCH = 8
BATCH_SIZE = 16
SEQ_LEN = 1024
N_REPEATS = 3


def load_real_data(model_dir: str, batch_size: int = 128, seq_len: int = 1024) -> torch.Tensor:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    data = load_dataset("wikitext", "wikitext-2-v1")["train"]
    all_text = "\n".join(t for t in data["text"] if t.strip())
    tokens = tokenizer(all_text, return_tensors="pt", truncation=True, max_length=batch_size * seq_len)
    return tokens["input_ids"][:, : batch_size * seq_len].reshape(batch_size, seq_len)

def measure_forward_time(model, input_ids: torch.Tensor) -> float:
    torch.cuda.synchronize()
    
    t0 = time.time()
    with torch.no_grad():
        for _ in range(N_REPEATS):
            for i in range(0, input_ids.shape[0], MICRO_BATCH):
                model(input_ids[i : i + MICRO_BATCH])
    
    torch.cuda.synchronize()

    return (time.time() - t0) / N_REPEATS

def traced_forward(model, input_ids: torch.Tensor, trace_path: str) -> float:
    torch.cuda.synchronize()

    with torch.no_grad(), torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
    ) as prof:
        for i in range(0, input_ids.shape[0], MICRO_BATCH):
            model(input_ids[i : i + MICRO_BATCH])

    torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)

def main() -> None:
    device = "cuda"

    input_ids = load_real_data(MODEL_DIR, batch_size=BATCH_SIZE, seq_len=SEQ_LEN).to(device)

    print(f"benchmark: input ({BATCH_SIZE}, {SEQ_LEN}), {MICRO_BATCH=}, {N_REPEATS=}\n")

    model = create_model(MODEL_DIR)
    model = model.to(device).eval()
    store = next(
        m.weight_store for m in model.modules() if hasattr(m, "weight_store")
    )

    measure_forward_time(model, input_ids[:MICRO_BATCH])
    t_prefetch = measure_forward_time(model, input_ids)
    print(f"with prefetch: {round(t_prefetch, 2)}s")
    traced_forward(model, input_ids, "trace_prefetch.json")

    store.prefetch = False
    store.gpu_cache.clear()
    store.prefetched_layer = None
    store.layer_events.clear()

    measure_forward_time(model, input_ids[:MICRO_BATCH])
    t_naive = measure_forward_time(model, input_ids)
    print(f"without prefetch: {round(t_naive, 2)}s")
    traced_forward(model, input_ids, "trace_naive.json")


if __name__ == "__main__":
    main()
