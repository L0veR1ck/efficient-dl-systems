import time

import torch
import transformers
from constants import MODEL_DIR
from create_model import create_model
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

BATCH_SIZE = 128
SEQ_LEN = 1024
MICRO_BATCH = 2
NUM_STEPS = 10
LR = 1e-4


def load_real_data(model_dir: str, batch_size: int = BATCH_SIZE, seq_len: int = SEQ_LEN) -> torch.Tensor:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    data = load_dataset("wikitext", "wikitext-2-v1")["train"]
    all_text = "\n".join(t for t in data["text"] if t.strip())
    tokens = tokenizer(all_text, return_tensors="pt", truncation=True, max_length=batch_size * seq_len)
    return tokens["input_ids"][:, : batch_size * seq_len].reshape(batch_size, seq_len)


def test_backwards() -> None:
    device = "cuda"

    model = create_model(MODEL_DIR).to(device)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model.print_trainable_parameters()

    input_ids = load_real_data(MODEL_DIR).to(device)
    print(f"training on {input_ids.shape=}, {NUM_STEPS=}, {MICRO_BATCH=}")

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=LR,
    )
    grad_accum_steps = BATCH_SIZE // MICRO_BATCH

    torch.cuda.reset_peak_memory_stats()
    for step in range(NUM_STEPS):
        t0 = time.time()
        optimizer.zero_grad()
        total_loss = 0.0

        for micro_i in range(0, BATCH_SIZE, MICRO_BATCH):
            micro_ids = input_ids[micro_i : micro_i + MICRO_BATCH]
            loss = model(micro_ids, labels=micro_ids).loss / grad_accum_steps
                
            loss.backward()
            total_loss += loss.item()

        optimizer.step()
        elapsed = time.time() - t0
        print(f"{step=}: {total_loss=}  ({elapsed=}s)")

    assert torch.cuda.max_memory_allocated() / 2**30 <= 10
