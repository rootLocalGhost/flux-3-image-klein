"""NVFP4 linears adapter for Intel Arc XPU environment.

NVFP4 is an NVIDIA Blackwell specific feature. If requested on Intel Arc XPU,
it provides informative diagnostics while allowing the checkpoint inspection to function.
"""

import torch
from torch import Tensor, nn

PACKED = torch.uint8
FP8 = torch.float8_e4m3fn
BLOCK = 16
SCALE_SUFFIX = ".weight_scale_2"


def _flashinfer():
    try:
        import flashinfer
        return flashinfer
    except ImportError as e:
        raise RuntimeError(
            "NVFP4 requires NVIDIA Blackwell architecture and FlashInfer. "
            "For Intel Arc A770, please run standard BF16 or FP8 checkpoints."
        ) from e


class NVFP4Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.empty(out_features, in_features // 2, dtype=PACKED, device=device))
        self.register_buffer("weight_scale", torch.empty(out_features, in_features // BLOCK, dtype=FP8, device=device))
        self.register_buffer("weight_scale_2", torch.empty((), dtype=torch.float32, device=device))
        self.register_buffer("input_scale", torch.empty((), dtype=torch.float32, device=device))

    def forward(self, x: Tensor) -> Tensor:
        if x.device.type == "xpu":
            raise RuntimeError(
                "NVFP4 inference is not supported on Intel Arc GPUs. "
                "Please run standard BF16 or FP8 on your Intel Arc A770."
            )
        flashinfer = _flashinfer()
        leading = tuple(x.shape[:-1])
        rows = math.prod(leading)
        if rows == 0:
            return x.new_zeros(*leading, self.out_features, dtype=torch.bfloat16)
        flat = x.to(torch.bfloat16).reshape(rows, self.in_features).contiguous()
        inv_a = self.input_scale.float().clamp(min=1e-8).reshape(1)
        x_q, a_sf = flashinfer.nvfp4_quantize(
            flat, (1.0 / inv_a).reshape(1), sfLayout=flashinfer.SfLayout.layout_128x4, do_shuffle=False
        )
        alpha = (self.input_scale * self.weight_scale_2).float().reshape(1)
        out = flashinfer.mm_fp4(
            x_q, self.weight.T, a_sf, self.weight_scale.T, alpha=alpha, out_dtype=torch.bfloat16, backend="cutlass"
        )
        return out.reshape(*leading, self.out_features)


def swizzle_weight_scales(model: nn.Module) -> None:
    flashinfer = _flashinfer()
    interleave = flashinfer.block_scale_interleave
    for module in model.modules():
        if isinstance(module, NVFP4Linear):
            rows, cols = module.weight_scale.shape
            swizzled = interleave(module.weight_scale.view(torch.uint8).contiguous())
            padded_rows = (rows + 127) // 128 * 128
            padded_cols = (cols + 3) // 4 * 4
            module.weight_scale = swizzled.reshape(padded_rows, padded_cols).view(FP8)

