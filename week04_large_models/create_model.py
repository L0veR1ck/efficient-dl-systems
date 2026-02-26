import torch
import transformers
from tqdm import trange
from prefetching import get_weight_map, WeightStore
from constants import MODEL_DIR
from offloading import MyOffloadedLinear


def prepare_for_offloading(
    layer: torch.nn.Module,
    index: int,
    store: WeightStore,
) -> None:
    linears = [
        (name, child)
        for (name, child) in layer.named_modules()
        if isinstance(child, torch.nn.Linear)
    ]
    for name, _ in linears:
        offloaded = MyOffloadedLinear(
            param_name=f"model.layers.{index}.{name}.weight",
            bias=store.get(f"model.layers.{index}.{name}.bias"),
            weight_store=store,
        )
        parent_name, child_name = name.rsplit(".", 1)
        layer.get_submodule(parent_name)._modules[child_name] = offloaded


def load_non_offloaded_weights(
    model: torch.nn.Module,
    store: WeightStore,
) -> None:
    for name, param in model.named_parameters():
        tensor = store.get(name)
        if tensor is not None and param.numel() > 0:
            param.data.copy_(tensor)


def create_model(model_dir: str = MODEL_DIR) -> transformers.AutoModelForCausalLM:
    config = transformers.AutoConfig.from_pretrained(model_dir)
    config._attn_implementation = "sdpa"
    actual_hidden_layers = config.num_hidden_layers
    config.num_hidden_layers = 0
    model = transformers.AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
    )

    store = WeightStore(get_weight_map(model_dir), weights_dir=model_dir)

    for index in trange(actual_hidden_layers, desc="preparing model for offloading"):
        layer = transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer(
            config,
            index,
        ).to(torch.bfloat16)
        prepare_for_offloading(layer, index, store)
        model.model.layers.append(layer)

    config.num_hidden_layers = actual_hidden_layers

    load_non_offloaded_weights(model, store)
    store.drop_non_layer_weights()

    return model


def main() -> None:
    model = create_model()
    print(model)


if __name__ == "__main__":
    main()
