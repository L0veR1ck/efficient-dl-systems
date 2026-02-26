import torch
import transformers
from constants import MODEL_DIR
from create_model import create_model


def test_generate_text() -> None:
    device = "cuda"

    model = create_model(MODEL_DIR)
    model = model.to(device).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)

    prompt = """\
Tony is talking to Darth Vader. 
Tony is a german kid that likes ice cream and computer games. 
Darth Vader is a sith lord that likes killing jedi and breathing heavily.

Tony: \
"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    new_tokens = 50

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
        )

    texts = [
        str(tokenizer.decode(ids, skip_special_tokens=True))
        for ids in output_ids
    ]
    print()
    print("\n".join(texts))
    
    assert torch.cuda.max_memory_allocated() / 2**30 <= 10


if __name__ == "__main__":
    test_generate_text()