"""Prompt + reference images -> image pipeline for Intel Arc A770 using torch.xpu.

Zero-RAM-Overhead Sequential Pipeline:
Does NOT keep models sitting in System RAM. Each component is loaded on-demand
from disk directly to XPU for its execution phase and immediately freed:
  1. TextEncoder loaded -> encode prompt & references -> immediately deleted.
  2. DiT loaded directly to XPU -> runs all 16/50 steps -> immediately deleted.
  3. AutoEncoder loaded -> decodes latents to PIL image -> immediately deleted.

Peak System RAM usage: ~4-5 GB during load, dropping back to ~500 MB.
Peak XPU VRAM usage: ~11 GB during DiT denoising.
"""

import gc
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor

from flux3_xpu.autoencoder import AutoEncoder, DOWNSAMPLING_RATIO, latent_ids, latent_shape
from flux3_xpu.checkpoint import is_flow_map, load_checkpoint
from flux3_xpu.dit import DiT, IN_CHANNELS, MODES, Packed
from flux3_xpu.sampling import denoise, denoise_flow_map, timesteps
from flux3_xpu.text_encoder import TextEncoder

WEIGHTS = Path(os.environ.get("WEIGHTS_DIR", "weights"))
MODELS = {
    "base": {"dit": "flux-3-image-klein-base", "steps": 50, "guidance": 4.0},
    "distilled": {"dit": "flux-3-image-klein", "steps": 16, "guidance": None},
}
QUANT = {"fp8": "fp8r", "nvfp4": "nvfp4"}
IMAGE_ROPE_SCALE = 0.5
MATCH_MAX_PIXELS = 4 * 1024 * 1024
MIN_SIDE, MAX_SIDE = 64, 4096
MAX_REFERENCES = 10
MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024


def dit_path(model: str, quant: str = "") -> Path:
    root = WEIGHTS / MODELS[model]["dit"]
    return root / "variants" / QUANT[quant] if quant else root


def find_autoencoder_path(shared_path: Path) -> Path | None:
    candidates = [
        shared_path / "ae.safetensors",
        shared_path / "vae" / "diffusion_pytorch_model.safetensors",
        shared_path / "vae" / "ae.safetensors",
        WEIGHTS / "ae.safetensors",
    ]
    for c in candidates:
        if c.is_file():
            return c
    if (shared_path / "vae").is_dir():
        for f in (shared_path / "vae").glob("*.safetensors"):
            return f
    return None


def empty_xpu_cache() -> None:
    gc.collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


class Pipeline:
    """Zero-RAM-overhead pipeline for Intel Arc A770.
    
    Loads each component from disk on demand and frees it immediately after use.
    """

    def __init__(
        self,
        dits: dict[str, str],
        text_encoder: str,
        autoencoder: str,
        device: str = "xpu",
        sequential: bool = True,
    ):
        self.device = device
        self.sequential = sequential
        self.dits_paths = dits
        self.text_encoder_path = text_encoder
        self.autoencoder_path = autoencoder
        print(f"[Pipeline_XPU] Configured Zero-RAM on-demand pipeline (device: {device}).")

    def generate(
        self,
        model: str,
        prompt: str,
        references: list[Image.Image],
        width: int,
        height: int,
        steps: int,
        guidance: float | None,
        seed: int,
        alpha: float = 16.0,
    ) -> Image.Image:
        if len(references) > MAX_REFERENCES:
            raise ValueError(f"at most {MAX_REFERENCES} references")

        h, w = latent_shape(height, width)

        # -------------------------------------------------------------
        # STAGE 1: Text Encoding & Reference Encoding
        # -------------------------------------------------------------
        print("\n[Stage 1/3] Loading TextEncoder from disk...")
        text_encoder = TextEncoder(self.text_encoder_path, device=self.device)

        cond = None
        if references:
            print("[Stage 1/3] Loading AutoEncoder for reference encoding...")
            autoencoder = load_checkpoint(AutoEncoder(), self.autoencoder_path, self.device, torch.float32)
            cond = self.encode_references(autoencoder, references)
            del autoencoder
            empty_xpu_cache()

        print("[Stage 1/3] Encoding prompt...")
        dit_path_str = self.dits_paths[model]
        flow_map = is_flow_map(dit_path_str)

        if flow_map:
            text_emb = text_encoder.encode(prompt)
            packed_samples = [[text_emb, cond, None]]
        else:
            text_uncond = text_encoder.encode("")
            text_cond = text_encoder.encode(prompt)
            packed_samples = [[text_uncond, cond, None], [text_cond, cond, None]]

        # Free TextEncoder completely from both RAM and XPU
        del text_encoder
        empty_xpu_cache()
        print("[Stage 1/3] TextEncoder unloaded. System RAM and XPU freed.")

        # -------------------------------------------------------------
        # STAGE 2: DiT Latents Setup & Denoising on XPU
        # -------------------------------------------------------------
        print(f"\n[Stage 2/3] Loading DiT '{model}' directly from disk to XPU...")
        with torch.device("meta"):
            dit_model = DiT(flow_map)
        dit_instance = load_checkpoint(dit_model, dit_path_str, self.device, torch.bfloat16)

        # Prepare initial noise latents on XPU
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn((h * w, IN_CHANNELS["image"]), generator=generator)
        noise = noise.to(torch.bfloat16).float().to(self.device)
        ids = latent_ids(h, w, self.device)
        image_tuple = (noise, scaled_ids(ids))

        for sample in packed_samples:
            sample[2] = image_tuple

        packed = pack(packed_samples)
        schedule = timesteps(steps, alpha)

        print(f"[Stage 2/3] Denoising {steps} steps on Intel Arc XPU...")
        if dit_instance.flow_map:
            tokens = denoise_flow_map(dit_instance, packed, schedule)
        else:
            tokens = denoise(dit_instance, packed, schedule, guidance)

        # Free DiT completely from both RAM and XPU
        del dit_instance
        del dit_model
        del packed
        empty_xpu_cache()
        print(f"[Stage 2/3] DiT '{model}' unloaded. System RAM and XPU freed.")

        # -------------------------------------------------------------
        # STAGE 3: AutoEncoder Latent Decoding
        # -------------------------------------------------------------
        print("\n[Stage 3/3] Loading AutoEncoder from disk to decode image...")
        autoencoder = load_checkpoint(AutoEncoder(), self.autoencoder_path, self.device, torch.float32)

        print("[Stage 3/3] Decoding latents to RGB image...")
        with torch.no_grad():
            decoded_image = autoencoder.decode(tokens.to(self.device), ids.to(self.device))

        del autoencoder
        empty_xpu_cache()
        print("[Stage 3/3] AutoEncoder unloaded. Generation complete.")

        return to_pil(decoded_image)

    def encode_references(self, autoencoder: AutoEncoder, references: list[Image.Image]) -> tuple[Tensor, Tensor] | None:
        if not references:
            return None
        tokens, ids = [], []
        for index, reference in enumerate(references):
            t, i = autoencoder.encode(to_tensor(reference).to(self.device))
            i = scaled_ids(i)
            i[:, 3] = -(1 + index)
            tokens.append(t.to(torch.bfloat16))
            ids.append(i)
        return torch.cat(tokens), torch.cat(ids)


def pack(samples: list[list[tuple[Tensor, Tensor] | None]]) -> Packed:
    data = {m: [] for m in MODES}
    ids = {m: [] for m in MODES}
    mode_indices = {m: [] for m in MODES}
    counts = dict.fromkeys(MODES, 0)
    for text, cond, image in samples:
        for mode, item in (("text", text), ("image_cond", cond), ("image", image)):
            start = counts[mode]
            if item is not None:
                data[mode].append(item[0])
                ids[mode].append(item[1])
                counts[mode] += item[0].shape[0]
            mode_indices[mode].append((start, counts[mode]))
    device = samples[0][0][0].device
    cat = lambda parts, width, dtype: torch.cat(parts) if parts else torch.zeros((0, width), device=device, dtype=dtype)
    data = {m: cat(data[m], IN_CHANNELS[m], torch.bfloat16) for m in MODES}
    ids = {m: cat(ids[m], 4, torch.float32) for m in MODES}
    return Packed(
        data=data,
        ids=ids,
        timesteps={m: torch.zeros(data[m].shape[0], device=device, dtype=torch.bfloat16) for m in MODES},
        mode_indices=mode_indices,
    )


def scaled_ids(ids: Tensor) -> Tensor:
    ids = ids.float()
    ids[:, 1:3] *= IMAGE_ROPE_SCALE
    return ids


def _ceil16(v: int) -> int:
    return -(-v // DOWNSAMPLING_RATIO) * DOWNSAMPLING_RATIO


def reference_canvas(width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    scale = min(1.0, math.sqrt(MATCH_MAX_PIXELS / (width * height)), MAX_SIDE / max(width, height))
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(min(MAX_SIDE, max(MIN_SIDE, _ceil16(v))) for v in fit)
        if canvas[0] * canvas[1] <= MATCH_MAX_PIXELS:
            return fit, canvas
        scale *= 0.995


def prepare_reference(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    fit, (w, h) = reference_canvas(image.width, image.height)
    if fit != image.size:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    if (w, h) == image.size:
        return image, None
    left, top = (w - image.width) // 2, (h - image.height) // 2
    padding = ((top, h - image.height - top), (left, w - image.width - left), (0, 0))
    mode = "reflect" if min(image.size) > DOWNSAMPLING_RATIO else "symmetric"
    return Image.fromarray(np.pad(np.asarray(image), padding, mode=mode)), (left, top, left + image.width, top + image.height)


def prepare_multi_reference(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width * height > MULTI_REFERENCE_MAX_PIXELS:
        scale = math.sqrt(MULTI_REFERENCE_MAX_PIXELS / (width * height))
        width, height = int(width * scale), int(height * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    w, h = width // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO, height // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO
    left, top = (width - w) // 2, (height - h) // 2
    return image.crop((left, top, left + w, top + h))


def default_size(references: list[Image.Image]) -> tuple[int, int]:
    return references[0].size if references else (1024, 1024)


def to_tensor(image: Image.Image) -> Tensor:
    return torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 127.5 - 1.0


def to_pil(image: Tensor) -> Image.Image:
    array = ((image.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)
