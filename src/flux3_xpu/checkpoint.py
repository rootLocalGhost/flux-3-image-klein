"""Load a torch checkpoint or safetensors file (bf16, FP8, NVFP4, or ComfyUI INT8) into a module on XPU or CPU."""

from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from torch.distributed.checkpoint import FileSystemReader, load

from flux3_xpu import fp8, nvfp4
from flux3_xpu.int8 import XPUInt8Linear

FLOW_MAP_KEY = "ref_time_in.in_layer.weight"
QUANT_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")


def safetensors_file(path: str) -> Path | None:
    p = Path(path)
    if p.is_file() and p.suffix == ".safetensors":
        return p
    if (p / "model.safetensors").exists():
        return p / "model.safetensors"
    if (p / "transformer").is_dir():
        for f in (p / "transformer").glob("*.safetensors"):
            return f
    if p.is_dir():
        for f in p.glob("*.safetensors"):
            return f
        for f in p.glob("*/*.safetensors"):
            if "text_encoder" not in str(f):
                return f
    return None


def checkpoint_keys(path: str) -> set[str]:
    if file := safetensors_file(path):
        with safe_open(file, "pt") as f:
            return set(f.keys())
    if Path(path).is_dir() and (Path(path) / ".metadata").exists():
        return {k.removeprefix("model.") for k in FileSystemReader(path).read_metadata().state_dict_metadata}
    return set()


def is_flow_map(path: str) -> bool:
    keys = checkpoint_keys(path)
    return FLOW_MAP_KEY in keys or any("ref_time_in" in k for k in keys)


def swap_linears(module: nn.Module, keys) -> type[nn.Module] | None:
    for suffix, linear_type in ((nvfp4.SCALE_SUFFIX, nvfp4.NVFP4Linear), (fp8.SCALE_SUFFIX, fp8.FP8Linear)):
        names = [n for n, m in module.named_modules() if isinstance(m, nn.Linear) and n + suffix in keys]
        if names:
            for name in names:
                linear = module.get_submodule(name)
                parent, _, child = name.rpartition(".")
                setattr(
                    module.get_submodule(parent), child,
                    linear_type(linear.in_features, linear.out_features, linear.weight.device)
                )
            return linear_type
    return None


def load_checkpoint(module: nn.Module, path: str, device: str, dtype: torch.dtype) -> nn.Module:
    """Reads only the keys the module has, loading into XPU or CPU."""
    if file := safetensors_file(path):
        return _load_safetensors(module, file, device, dtype)
    if Path(path).is_dir() and (Path(path) / ".metadata").exists():
        state = {
            f"model.{name}": torch.empty(
                tensor.shape, device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype
            )
            for name, tensor in module.state_dict().items()
        }
        load(state, checkpoint_id=path)
        module.load_state_dict({name[len("model."):]: tensor for name, tensor in state.items()}, assign=True)
        return module.to(device).eval()

    raise FileNotFoundError(
        f"Could not find valid checkpoint weights in '{path}'. "
        "Expected a .safetensors file (e.g. model.safetensors or in transformer/) or a distributed checkpoint."
    )


def _load_safetensors(module: nn.Module, file: Path, device: str, dtype: torch.dtype) -> nn.Module:
    print(f"[Checkpoint_XPU] Loading weights from {file} to {device}...")
    
    with safe_open(str(file), framework="pt", device="cpu") as f:
        all_keys = set(f.keys())

    is_comfy_int8 = any(k.endswith(".comfy_quant") or ".qkv.weight" in k for k in all_keys)

    if is_comfy_int8:
        print(f"[Checkpoint_XPU] Converting linear layers to native XPUInt8Linear (saving 50% VRAM)...")
        # Swap Linear layers with XPUInt8Linear
        for name, m in list(module.named_modules()):
            if isinstance(m, nn.Linear):
                has_scale = f"{name}.weight" in all_keys and f"{name}.weight_scale" in all_keys
                is_qkv = any(sub in name for sub in (".q_proj", ".k_proj", ".v_proj"))
                if has_scale or is_qkv:
                    parent, _, child = name.rpartition(".")
                    int8_layer = XPUInt8Linear(m.in_features, m.out_features, device=device)
                    setattr(module.get_submodule(parent), child, int8_layer)

        module.to_empty(device=device)
        module.to(dtype)
        state_dict = module.state_dict()

        with safe_open(str(file), framework="pt", device="cpu") as f:
            for k in all_keys:
                if k.endswith(".comfy_quant") or k.endswith(".comfy_attention.config") or k.endswith(".weight_scale"):
                    continue
                if ".qkv.weight" in k:
                    w = f.get_tensor(k)
                    s = f.get_tensor(k.replace(".weight", ".weight_scale"))
                    q_w, k_w, v_w = w.chunk(3, dim=0)
                    q_s, k_s, v_s = s.chunk(3, dim=0)
                    state_dict[k.replace(".qkv.weight", ".q_proj.weight")].copy_(q_w)
                    state_dict[k.replace(".qkv.weight", ".q_proj.weight_scale")].copy_(q_s)
                    state_dict[k.replace(".qkv.weight", ".k_proj.weight")].copy_(k_w)
                    state_dict[k.replace(".qkv.weight", ".k_proj.weight_scale")].copy_(k_s)
                    state_dict[k.replace(".qkv.weight", ".v_proj.weight")].copy_(v_w)
                    state_dict[k.replace(".qkv.weight", ".v_proj.weight_scale")].copy_(v_s)
                else:
                    scale_key = k.replace(".weight", ".weight_scale")
                    if scale_key in all_keys:
                        w = f.get_tensor(k)
                        s = f.get_tensor(scale_key)
                        if k in state_dict:
                            state_dict[k].copy_(w)
                        if scale_key in state_dict:
                            state_dict[scale_key].copy_(s)
                    else:
                        t = f.get_tensor(k)
                        if k in state_dict:
                            state_dict[k].copy_(t.to(device).to(state_dict[k].dtype))
        return module.eval()

    # Standard path (FP8 / BF16)
    try:
        raw_state = load_file(str(file), device=str(device))
    except Exception:
        raw_state = load_file(str(file), device="cpu")
        raw_state = {k: v.to(device) for k, v in raw_state.items()}

    quantized = swap_linears(module, all_keys)
    keep = lambda k, v: k.endswith(QUANT_SUFFIXES) or v.dtype in (fp8.FP8, nvfp4.PACKED) or not v.is_floating_point()
    state = {k: v if keep(k, v) else v.to(dtype) for k, v in raw_state.items()}
    if quantized is nvfp4.NVFP4Linear and str(device).startswith("cuda"):
        nvfp4.swizzle_weight_scales(module)

    expected = module.state_dict()
    missing = sorted(set(expected) - set(state))
    if missing:
        raise KeyError(f"{file}: missing {len(missing)} keys, sample: {missing[:5]}")

    filtered_state = {k: v for k, v in state.items() if k in expected}
    module.load_state_dict(filtered_state, strict=True, assign=True)
    return module.to(device).eval()
