"""Qwen3-VL-4B as text encoder for Intel Arc XPU.

Supports standard HuggingFace weights as well as ComfyUI INT8 quantized weights
(e.g., qwen3vl_4b_int8_convrot.safetensors) with automatic dequantization into bfloat16.
"""

import math
from pathlib import Path
import torch
from torch import Tensor
from safetensors.torch import load_file
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
MAX_LENGTH = 4096
PAD_MULTIPLE = 80


def find_int8_text_encoder(path: Path) -> Path | None:
    """Look for ComfyUI INT8 text encoder safetensors in path or nearby directories."""
    candidates = [
        path / "qwen3vl_4b_int8_convrot.safetensors",
        path.parent / "qwen3vl_4b_int8_convrot.safetensors",
        path.parent / "transformer" / "qwen3vl_4b_int8_convrot.safetensors",
        path / "text_encoder.safetensors",
    ]
    for c in candidates:
        if c.is_file():
            return c

    # Search for any qwen int8 safetensors
    for root in (path, path.parent):
        if root.is_dir():
            for f in root.glob("*qwen*int8*.safetensors"):
                return f
            for f in root.glob("*/*qwen*int8*.safetensors"):
                return f
    return None


class TextEncoder:
    def __init__(self, path: str, device: str = "xpu"):
        self.path = Path(path)
        self.device = device

        print(f"[TextEncoder_XPU] Loading processor from {self.path}...")
        self.processor = AutoProcessor.from_pretrained(str(self.path))
        self.processor.tokenizer.padding_side = "right"

        int8_file = find_int8_text_encoder(self.path)
        if int8_file:
            print(f"[TextEncoder_XPU] Found INT8 text encoder weights: {int8_file}")
            self.model = self._load_int8_model(int8_file, device)
        else:
            # Check if standard sharded weights exist
            shards = list(self.path.glob("model*.safetensors"))
            if not shards:
                raise FileNotFoundError(
                    f"\n[TextEncoder] No weights found in '{self.path}'.\n"
                    "If downloading 'qwen3vl_4b_int8_convrot.safetensors', please ensure it finishes downloading\n"
                    f"and place it in '{self.path}' or '{self.path.parent}'.\n"
                )
            print(f"[TextEncoder_XPU] Loading standard weights from {self.path} to {device}...")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(self.path), torch_dtype=torch.bfloat16
            ).to(device).eval()

    def _load_int8_model(self, file_path: Path, device: str) -> torch.nn.Module:
        print(f"[TextEncoder_XPU] Initializing Qwen3-VL-4B architecture from config...")
        config = AutoConfig.from_pretrained(str(self.path))
        with torch.device("meta"):
            model = Qwen3VLForConditionalGeneration(config)

        expected = model.state_dict()
        print(f"[TextEncoder_XPU] Loading INT8 safetensors from {file_path}...")
        try:
            raw_state = load_file(str(file_path), device="cpu")
        except Exception:
            raw_state = load_file(str(file_path))

        all_keys = set(raw_state.keys())
        state = {}

        print("[TextEncoder_XPU] Dequantizing INT8 weights into bfloat16 (un-rotating ConvRot Hadamard)...")
        for k in all_keys:
            if k.endswith(".comfy_quant") or k.endswith(".weight_scale") or k.endswith(".comfy_attention.config"):
                continue

            # Check if there is a weight_scale
            scale_key = k.replace(".weight", ".weight_scale") if k.endswith(".weight") else (k + "_scale")
            tensor = raw_state[k]

            if scale_key in all_keys and tensor.dtype in (torch.int8, torch.uint8):
                scale = raw_state[scale_key]
                if tensor.ndim >= 2 and tensor.shape[-1] % 256 == 0:
                    try:
                        from comfy_kitchen.backends.eager import dequantize_int8_convrot_weight
                        val = dequantize_int8_convrot_weight(tensor, scale, 256).to(torch.bfloat16)
                    except Exception as e:
                        print(f"[TextEncoder_XPU] Warning: fallback dequantization for {k}: {e}")
                        val = (tensor.float() * scale).to(torch.bfloat16)
                else:
                    val = (tensor.float() * scale).to(torch.bfloat16)
            else:
                val = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor

            # Handle possible key prefixes (e.g., comfy adds or removes 'model.' prefix)
            target_key = k
            if target_key not in expected:
                if f"model.{target_key}" in expected:
                    target_key = f"model.{target_key}"
                elif target_key.startswith("model.") and target_key[6:] in expected:
                    target_key = target_key[6:]

            state[target_key] = val

        # Realize model on device with assign=True
        model.to_empty(device=device)
        missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
        if missing:
            print(f"[TextEncoder_XPU] Note: {len(missing)} missing keys in INT8 checkpoint (e.g. {missing[:3]})")
        
        filtered_state = {k: v.to(device) for k, v in state.items() if k in expected}
        model.load_state_dict(filtered_state, strict=False, assign=True)
        return model.eval()

    def to(self, device: str):
        self.device = device
        self.model = self.model.to(device)
        return self

    @torch.inference_mode()
    def encode(self, prompt: str) -> tuple[Tensor, Tensor]:
        """prompt -> tokens (L, 20480) bf16 and ids (L, 4) = (0, 0, 0, position); L padded to a multiple of 80."""
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        tokenizer = self.processor.tokenizer
        real = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)["input_ids"].shape[1]
        length = min(math.ceil(real / PAD_MULTIPLE) * PAD_MULTIPLE, MAX_LENGTH)
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
        mask = inputs["attention_mask"].to(self.device)
        positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
        output = self.model(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=mask,
            position_ids=positions[None].expand(4, -1, -1),
            output_hidden_states=True,
            use_cache=False,
        )
        stacked = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS], dim=1)
        tokens = stacked.permute(0, 2, 1, 3).reshape(length, -1).to(torch.bfloat16)
        positions = torch.arange(length, device=self.device)
        ids = torch.stack([torch.zeros_like(positions)] * 3 + [positions], dim=-1)
        return tokens, ids
