import torch
import transformers
from constants import MODEL_DIR
from create_model import create_model
from datasets import load_dataset

MICRO_BATCH = 8


def load_real_data(model_dir: str, batch_size: int = 128, seq_len: int = 1024) -> torch.Tensor:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    data = load_dataset("wikitext", "wikitext-2-v1")["train"]
    all_text = "\n".join(t for t in data["text"] if t.strip())
    tokens = tokenizer(all_text, return_tensors="pt", truncation=True, max_length=batch_size * seq_len)
    return tokens["input_ids"][:, : batch_size * seq_len].reshape(batch_size, seq_len)


def test_big_batch() -> None:
    device = "cuda"

    model = create_model(MODEL_DIR).to(device)
    model.eval()

    input_ids = load_real_data(MODEL_DIR).to(device)
    print(f"forward: {input_ids.shape=}, {MICRO_BATCH=}")

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(0, input_ids.shape[0], MICRO_BATCH):
            model(input_ids[i : i + MICRO_BATCH])
    
    torch.cuda.synchronize()

    assert torch.cuda.max_memory_allocated() / 2**30 <= 10

