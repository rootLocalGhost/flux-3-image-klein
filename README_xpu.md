# Running FLUX 3 Image Klein on Intel Arc A770 (16GB VRAM)

This guide explains how to run FLUX 3 Image Klein on your Intel Arc A770 GPU using native `torch.xpu` with sequential loading to prevent VRAM overflow.

---

## 1. Environment Setup

### Install PyTorch with native XPU support (No IPEX required)

Intel Arc GPUs are natively supported by upstream PyTorch (2.4+ / 2.5+ / 2.6+) via SYCL.

In your activated conda environment:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/xpu
```
*(Or install the latest official PyTorch XPU wheel for Windows)*

### Install Other Dependencies
```bash
pip install -r requirements_xpu.txt
```
*(Includes `transformers`, `safetensors`, `accelerate`, `gradio`, and `comfy-kitchen` for Intel Arc INT8 ConvRot XMX acceleration)*

---

## 2. Verify Your Environment

Run the diagnostic script to ensure your Intel Arc A770 is detected and bfloat16 operations succeed on `torch.xpu`:
```bash
python scripts_xpu/test_env_xpu.py
```

Expected output:
```
FLUX 3 Image Klein - Intel Arc A770 (torch.xpu) Diagnostic
Python version : 3.x
PyTorch version: 2.x
IPEX loaded    : False (Should be False - using native torch.xpu)
torch.xpu ready: True
Device 0: Intel(R) Arc(TM) A770 Graphics
  Total VRAM : 16.00 GB
  [PASS] bfloat16 GEMM succeeded: result shape torch.Size([1024, 1024]), dtype torch.bfloat16
  [PASS] torch.xpu.empty_cache() executed cleanly.
SUCCESS: Your Intel Arc A770 is fully configured for FLUX 3 Klein!
```

---

## 3. Sequential Loading (Why it fits into 16GB VRAM)

The model components in BF16 are:
- **DiT (5.6B)**: ~11.2 GB
- **Text Encoder (Qwen3-VL-4B)**: ~8.0 GB
- **AutoEncoder (VAE)**: ~0.5 GB

Loaded together, they require ~19.7 GB VRAM, which exceeds the 16GB capacity of the Intel Arc A770.

The `flux3_xpu` pipeline uses **Sequential Component Loading**:
1. **Stage 1**: TextEncoder loads onto XPU -> Encodes prompt -> Unloads and clears VRAM (`torch.xpu.empty_cache()`).
2. **Stage 2**: DiT (5.6B) loads onto XPU (~11.2 GB) -> Executes all 16 (distilled) or 50 (base) diffusion steps at full native GPU speed -> Unloads.
3. **Stage 3**: AutoEncoder loads onto XPU (~0.5 GB) -> Decodes latents to the final RGB image.

*Notice: This is NOT layer-wise CPU offloading; the DiT stays 100% on the Arc A770 during the entire denoising process with zero PCIe bottlenecks.*

---

## 4. Running Inference

### Text-to-Image (CLI)

Fastest inference using the distilled model (16 steps, no CFG):
```bash
python scripts_xpu/cli_xpu.py "A photorealistic red vintage sports car parked by the coast at sunset" --model distilled --out car.png
```

To run without prompt upsampler (fastest startup):
```bash
python scripts_xpu/cli_xpu.py "A photorealistic red vintage sports car parked by the coast at sunset" --model distilled --no-pup --out car.png
```

### Image-to-Image / Editing (CLI)
```bash
python scripts_xpu/cli_xpu.py "Put a warm yellow scarf around the neck" --image input.png --model distilled --out edited.png
```

### Web UI (Gradio)
```bash
python scripts_xpu/app_xpu.py
```
Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

