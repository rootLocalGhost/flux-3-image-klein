"""Prompt (+ references, + boxes) -> image for Intel Arc XPU with sequential loading."""

import hashlib
import json
from pathlib import Path

from PIL import Image

from flux3_xpu import bbox
from flux3_xpu.pipeline import (
    MAX_REFERENCES,
    MODELS,
    WEIGHTS,
    Pipeline,
    default_size,
    find_autoencoder_path,
    prepare_multi_reference,
    prepare_reference,
)
from flux3_xpu.safety import Safety, exif


class Refused(Exception):
    """Blocked by a safety check."""


class Models:
    """FLUX 3 Klein model bundle for Intel Arc A770.
    
    Loads models once. In sequential mode (default), components are staged into XPU
    one by one during generation to ensure peak VRAM stays well within 16GB.
    """

    def __init__(
        self,
        dits: dict[str, str],
        pup: str | None = None,
        device: str = "xpu",
        sequential: bool = True,
        load_safety: bool = False,
    ):
        self.device = device
        self.sequential = sequential

        from flux3_xpu.pup import Upsampler
        self.upsampler = Upsampler(pup, device=device) if pup else None

        shared = WEIGHTS / MODELS[next(iter(dits))]["dit"]
        ae_path = find_autoencoder_path(shared)
        if ae_path is None:
            raise FileNotFoundError(
                f"\n[Missing VAE] Could not locate 'ae.safetensors' in '{shared}' or '{shared / 'vae'}'.\n"
                "FLUX 3 requires the VAE AutoEncoder weights to decode image latents.\n"
                "Please download ae.safetensors:\n"
                "  hf download black-forest-labs/flux-3-image-klein ae.safetensors --local-dir weights/flux-3-image-klein\n"
            )

        text_enc_dir = shared / "text_encoder"
        if not text_enc_dir.exists():
            raise FileNotFoundError(f"Missing text_encoder directory at '{text_enc_dir}'.")

        self.pipeline = Pipeline(
            dits=dits,
            text_encoder=str(text_enc_dir),
            autoencoder=str(ae_path),
            device=device,
            sequential=sequential,
        )
        self.safety = (
            Safety(self.pipeline.text_encoder, str(WEIGHTS / "nsfw-classifier"), self.upsampler)
            if load_safety
            else None
        )


def generate(
    models: Models,
    model: str,
    prompt: str,
    images: list[Image.Image],
    boxes: list[dict] | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    pup: bool = True,
    safety: bool = False,
) -> tuple[Image.Image, dict]:
    """Generates an image on Intel Arc A770 GPU using torch.xpu with sequential loading."""
    if len(images) > MAX_REFERENCES:
        raise ValueError(f"at most {MAX_REFERENCES} references")
    defaults = MODELS[model]
    if defaults["guidance"] is None and guidance not in (None, 1.0):
        raise ValueError(f"{model} samples without CFG; drop the guidance")
    steps = steps or defaults["steps"]
    guidance = defaults["guidance"] if guidance is None else guidance

    if safety:
        if models.safety is None:
            models.safety = Safety(
                models.pipeline.text_encoder, str(WEIGHTS / "nsfw-classifier"), models.upsampler
            )
        screened = " ".join([prompt] + [box.get("desc", "") for box in boxes or []])
        if models.safety.prompt_flagged(screened):
            raise Refused("The prompt may name a copyrighted character, a brand or a public figure.")
        for i, image in enumerate(images):
            if reason := models.safety.image_flagged(image):
                raise Refused(f"Reference image {i + 1} may show {reason}.")

    finished = not boxes and bbox.is_finished(prompt)
    use_pup = pup and models.upsampler is not None and not finished
    caption = None
    if use_pup:
        try:
            print("[Stage 0] Upsampling prompt with PUP...")
            caption = models.upsampler(prompt, images, boxes)
            if models.sequential:
                models.upsampler.unload()
        except Exception as error:
            print(f"[Stage 0] Upsampling skipped: {error}", flush=True)
            use_pup = False

    if caption is None:
        caption = f"{prompt.rstrip()} {json.dumps(bbox.to_canvas(boxes), ensure_ascii=False)}" if boxes else prompt

    if len(images) > 1:
        references, crop = [prepare_multi_reference(image) for image in images], None
    else:
        prepared = [prepare_reference(image) for image in images]
        references = [image for image, _ in prepared]
        crop = prepared[0][1] if prepared and not (width or height) else None

    default_width, default_height = default_size(references)
    width, height = width or default_width, height or default_height
    seed = seed if seed is not None else int(hashlib.sha256(caption.encode()).hexdigest(), 16) % 2**32

    image = models.pipeline.generate(model, caption, references, width, height, steps, guidance, seed)
    if crop is not None:
        image = image.crop(crop)

    if safety and models.safety and (reason := models.safety.image_flagged(image)):
        raise Refused(f"The generated image may show {reason}, so it was not returned. Try another prompt or seed.")

    record = {
        "model": model,
        "prompt": caption,
        "input_prompt": prompt,
        "boxes": boxes,
        "width": image.width,
        "height": image.height,
        "steps": steps,
        "guidance": guidance,
        "seed": seed,
        "upsampled": use_pup,
        "device": models.device,
        "sequential": models.sequential,
    }
    return image, record
