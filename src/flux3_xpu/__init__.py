"""FLUX 3 Image Klein inference codebase adapted for Intel Arc GPUs using torch.xpu."""

import torch

def get_xpu_device(device_id: int = 0) -> str:
    """Return xpu:<id> if torch.xpu is available, else fallback with warning."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return f"xpu:{device_id}" if device_id >= 0 else "xpu"
    raise RuntimeError(
        "torch.xpu is not available. Please ensure you have PyTorch with XPU support installed "
        "(e.g., PyTorch >= 2.4/2.5+ with native SYCL support for Intel Arc) and your Intel Arc GPU drivers are up to date."
    )

def empty_xpu_cache() -> None:
    """Safely flush XPU memory cache."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()

