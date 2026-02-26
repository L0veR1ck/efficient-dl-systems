import json
import os
from typing import Optional

import torch
from safetensors import safe_open


def get_weight_map(model_dir: str) -> dict[str, str]:
    index_path = f"{model_dir}/model.safetensors.index.json"
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)["weight_map"]
        
    path = f"{model_dir}/model.safetensors"
    with safe_open(path, framework="pt", device="cpu") as f:
        return {name: "model.safetensors" for name in f.keys()}


def _parse_layer_idx(param_name: str) -> Optional[int]:
    parts = param_name.split(".")
    if (
        len(parts) >= 4
        and parts[0] == "model"
        and parts[1] == "layers"
        and parts[2].isdigit()
    ):
        return int(parts[2])
    
    return None


class WeightStore:
    def __init__(
        self,
        file_weight_map: dict[str, str],
        weights_dir: str,
        prefetch: bool = True,
    ):
        self.weights = {}
        self.layer_params = {}
        self.prefetch = prefetch
        self.load_all(file_weight_map, weights_dir)
        self.build_layer_index()

        self.copy_stream = None
        self.gpu_cache = {}
        self.prefetched_layer = None
        self.layer_events = {}

    def load_all(self, file_weight_map: dict[str, str], weights_dir: str) -> None:
        files_to_params: dict[str, list[str]] = {}
        for param_name, filename in file_weight_map.items():
            files_to_params.setdefault(filename, []).append(param_name)

        for filename, param_names in files_to_params.items():
            path = filename if os.path.isabs(filename) else f"{weights_dir}/{filename}"
            with safe_open(path, framework="pt", device="cpu") as f:
                for name in param_names:
                    self.weights[name] = f.get_tensor(name).pin_memory()

    def build_layer_index(self) -> None:
        for name in self.weights:
            if name.endswith(".weight") and len(name.split(".")) == 6:
                layer_idx = _parse_layer_idx(name)
                if layer_idx is not None:
                    self.layer_params.setdefault(layer_idx, []).append(name)

    def get_copy_stream(self) -> torch.cuda.Stream:
        if self.copy_stream is None:
            self.copy_stream = torch.cuda.Stream()

        return self.copy_stream

    def prefetch_layer_to_gpu(self, layer_idx: int, device: str = "cuda") -> None:
        if layer_idx not in self.layer_params or layer_idx == self.prefetched_layer:
            return

        stream = self.get_copy_stream()
        stream.synchronize()

        stale = [k for k in self.gpu_cache if _parse_layer_idx(k) != layer_idx - 1]
        for k in stale:
            del self.gpu_cache[k]

        with torch.cuda.stream(stream):
            for name in self.layer_params[layer_idx]:
                self.gpu_cache[name] = self.weights[name].to(
                    device, non_blocking=True
                )
        self.layer_events[layer_idx] = stream.record_event()
        self.prefetched_layer = layer_idx

    def prefetch_next_layer(self, param_name: str):
        if not param_name.endswith(".weight"):
            return
        
        layer_idx = _parse_layer_idx(param_name)
        if layer_idx is not None and (layer_idx + 1) in self.layer_params:
            self.prefetch_layer_to_gpu(layer_idx + 1)

    def drop_non_layer_weights(self):
        layer_names = {n for names in self.layer_params.values() for n in names}
        for name in [n for n in self.weights if n not in layer_names]:
            del self.weights[name]

    def get(self, param_name: str, device: Optional[str] = None) -> Optional[torch.Tensor]:
        if self.prefetch and device is not None and device.startswith("cuda"):
            if param_name in self.gpu_cache:
                layer_idx = _parse_layer_idx(param_name)
                if layer_idx is not None and layer_idx in self.layer_events:
                    torch.cuda.current_stream().wait_event(self.layer_events[layer_idx])

                tensor = self.gpu_cache.pop(param_name)
                self.prefetch_next_layer(param_name)
                
                return tensor

        tensor = self.weights.get(param_name)
        if tensor is None:
            return None
        
        if device is not None:
            result = tensor.to(device)
            if self.prefetch:
                self.prefetch_next_layer(param_name)
            
            return result
        
        return tensor
