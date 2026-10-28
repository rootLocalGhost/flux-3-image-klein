"""Generate images on Intel Arc A770 GPU using native torch.xpu.

Example usage:
  python scripts_xpu/cli_xpu.py "A serene Japanese zen garden with maple leaves" --model distilled --out out_xpu.png
  python scripts_xpu/cli_xpu.py "Put a warm yellow scarf on the dog" --image input.png --model distilled --out edit_xpu.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to python path
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import torch
from PIL import Image

from flux3_xpu.generate import Models, Refused, generate
from flux3_xpu.pipeline import MAX_REFERENCES, MODELS, WEIGHTS, dit_path
from flux3_xpu.safety import exif


def check_xpu_device(device_name: str) -> None:
    if device_name.startswith("xpu"):
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            print("=" * 60)
            print("WARNING: torch.xpu is not available in the current environment.")
            print("Make sure you are using PyTorch with Intel XPU support (e.g., PyTorch >= 2.4/2.5+)")
            print("Falling back to CPU for execution.")
            print("=" * 60)
            return
        device_id = 0
        if ":" in device_name:
            device_id = int(device_name.split(":")[1])
        device_props = torch.xpu.get_device_properties(device_id)
        total_vram_gb = device_props.total_memory / (1024**3)
        print(f"[XPU] Detected Intel GPU: {device_props.name} ({total_vram_gb:.2f} GB VRAM)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Text prompt for generation or editing")
    parser.add_argument(
        "--model", choices=MODELS, default="distilled",
        help="distilled: 16 steps, no CFG (fastest); base: 50 steps, CFG 4"
    )
    parser.add_argument(
        "--image", action="append", default=[],
        help=f"Reference image for image-to-image/editing (max {MAX_REFERENCES})"
    )
    parser.add_argument("--boxes", help="JSON list of bounding boxes")
    parser.add_argument("--out", default="out_xpu.png", help="Output image file path")
    parser.add_argument("--width", type=int, help="Output width (must be multiple of 16)")
    parser.add_argument("--height", type=int, help="Output height (must be multiple of 16)")
    parser.add_argument("--steps", type=int, help="Sampling steps (defaults to 16 for distilled, 50 for base)")
    parser.add_argument("--guidance", type=float, help="CFG scale (base model only)")
    parser.add_argument("--seed", type=int, help="Random seed for generation")
    parser.add_argument(
        "--no-pup", action="store_true",
        help="Skip prompt upsampler (recommended on 16GB VRAM for fastest direct inference)"
    )
    parser.add_argument(
        "--sequential", action=argparse.BooleanOptionalAction, default=True,
        help="Stage models sequentially into XPU (default: enabled; required for 16GB VRAM)"
    )
    parser.add_argument(
        "--device", default="xpu",
        help="Compute device: 'xpu' or 'xpu:0' (default: 'xpu')"
    )
    quant = parser.add_mutually_exclusive_group()
    quant.add_argument("--fp8", action="store_true", help="Load FP8 quantized DiT checkpoint")
    parser.add_argument("--dit", help="Custom DiT checkpoint path")
    parser.add_argument("--safety", action="store_true", help="Enable safety filters")

    args = parser.parse_args()

    # Determine effective device
    effective_device = args.device
    if effective_device.startswith("xpu") and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        effective_device = "cpu"
    else:
        check_xpu_device(args.device)

    quant_mode = "fp8" if args.fp8 else ""
    dit = args.dit or str(dit_path(args.model, quant_mode))
    
    pup_path = None if args.no_pup else str(WEIGHTS / "pup")
    if pup_path and not Path(pup_path).exists():
        print(f"[PUP] Notice: '{pup_path}' not found on disk. Proceeding without prompt upsampler (--no-pup).")
        pup_path = None

    print(f"[Init] Loading FLUX 3 Klein for Intel Arc A770 (Sequential: {args.sequential})...")
    models = Models(
        dits={args.model: dit},
        pup=pup_path,
        device=effective_device,
        sequential=args.sequential,
        load_safety=args.safety,
    )

    boxes = json.loads(Path(args.boxes).read_text()) if args.boxes else None
    images = [Image.open(path) for path in args.image]

    try:
        image, record = generate(
            models=models,
            model=args.model,
            prompt=args.prompt,
            images=images,
            boxes=boxes,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance=args.guidance,
            seed=args.seed,
            pup=(pup_path is not None and not args.no_pup),
            safety=args.safety,
        )
    except Refused as error:
        raise SystemExit(f"Blocked by safety check: {error}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out, exif=exif(args.model))
    record |= {"references": args.image, "dit": dit}
    Path(args.out).with_suffix(".json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[Success] Generated image saved to: {args.out}")
    print(f"[Success] Generation details saved to: {Path(args.out).with_suffix('.json')}")


if __name__ == "__main__":
    main()

