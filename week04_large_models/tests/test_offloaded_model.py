import torch
import transformers
from create_model import create_model

SMALL_MODEL_DIR = "./model-1.5b"
SMALL_MODEL_NAME = "Qwen/Qwen2.5-1.5B"


def test_offloaded_model() -> None:
    torch.manual_seed(0)
    device = "cuda"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL_DIR, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    offloaded_model = create_model(SMALL_MODEL_DIR).to(device).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(SMALL_MODEL_DIR)
    inputs = tokenizer("A cat sat on a mat", return_tensors="pt").to(device)

    with torch.no_grad():
        ref_logits = model(**inputs).logits
        off_logits = offloaded_model(**inputs).logits

    assert torch.allclose(ref_logits, off_logits, atol=1e-2)
