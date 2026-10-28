"""Diagnostic tool to verify Intel Arc A770 GPU and native torch.xpu environment.

Run with:
  python scripts_xpu/test_env_xpu.py
"""

import sys
import torch

def main():
    print("=" * 65)
    print("FLUX 3 Image Klein - Intel Arc A770 (torch.xpu) Diagnostic")
    print("=" * 65)
    print(f"Python version : {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check IPEX is NOT loaded
    ipex_loaded = "intel_extension_for_pytorch" in sys.modules
    print(f"IPEX loaded    : {ipex_loaded} (Should be False - using native torch.xpu)")

    # Check torch.xpu availability
    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    print(f"torch.xpu ready: {xpu_available}")

    if not xpu_available:
        print("\n[!] torch.xpu is NOT detected in your active Python environment.")
        print("    To install PyTorch with native Intel Arc (XPU/SYCL) support on Windows:")
        print("    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/xpu")
        print("    or install official stable PyTorch with XPU support.\n")
        return

    device_count = torch.xpu.device_count()
    print(f"XPU Device count: {device_count}")

    for i in range(device_count):
        props = torch.xpu.get_device_properties(i)
        total_mem_gb = props.total_memory / (1024**3)
        print(f"\nDevice {i}: {props.name}")
        print(f"  Total VRAM : {total_mem_gb:.2f} GB")
        if hasattr(props, "driver_version"):
            print(f"  Driver ver : {props.driver_version}")

    # Test tensor allocation and bfloat16 matmul on XPU
    print("\nRunning bfloat16 compute test on torch.xpu...")
    try:
        device = torch.device("xpu:0")
        a = torch.randn(1024, 1024, dtype=torch.bfloat16, device=device)
        b = torch.randn(1024, 1024, dtype=torch.bfloat16, device=device)
        c = torch.matmul(a, b)
        torch.xpu.synchronize()
        print(f"  [PASS] bfloat16 GEMM succeeded: result shape {c.shape}, dtype {c.dtype}")
    except Exception as e:
        print(f"  [FAIL] Compute test encountered an error: {e}")
        return

    # Test memory clearing
    torch.xpu.empty_cache()
    print("  [PASS] torch.xpu.empty_cache() executed cleanly.")

    print("\n" + "=" * 65)
    print("SUCCESS: Your Intel Arc A770 is fully configured for FLUX 3 Klein!")
    print("Run inference using: python scripts_xpu/cli_xpu.py \"Your prompt\"")
    print("=" * 65)

if __name__ == "__main__":
    main()

