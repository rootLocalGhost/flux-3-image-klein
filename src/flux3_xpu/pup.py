"""Local prompt upsampling adapted for Intel Arc XPU using Transformers and native torch.xpu.

Supports both Hugging Face Transformers native pipeline (fully functional on Intel Arc XPU)
and vLLM when available. Designed for sequential execution: loads, upsamples, and unloads
cleanly so that subsequent diffusion stages retain the full 16GB VRAM budget.
"""

import argparse
import io
import json
import math
import os
from pathlib import Path

import torch
from PIL import Image, ImageOps

from flux3_xpu import bbox

WEIGHTS = Path(os.environ.get("WEIGHTS_DIR", "weights"))
MODEL = str(WEIGHTS / "pup")
PROMPTS = Path(__file__).with_name("pup_prompts.json")
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
MAX_REFERENCES = 10

SAMPLING = {"temperature": 0.0, "top_p": 0.95, "top_k": 20, "max_new_tokens": 1024}
PRESENCE_PENALTY = {"t2i": 1.5, "i2i": 0.0}
I2I_SCAFFOLD = {
    False: (
        "Reference images follow in order (the first image, the second image, ...).\n"
        "User request: {instruction}\n"
        "Output the caption."
    ),
    True: (
        "Reference images follow in attachment order: ref_image_0, ref_image_1, ...\n"
        "User editing instruction: {instruction}\n"
        "Decide the edit, imagine the target image, and output the annotation record."
    ),
}
REFERENCE_MULTIPLE = 32
MAX_ASPECT_RATIO = 200
MAX_PIXELS_BY_REFS = {1: 1024 * 1024, 2: 724 * 724, **{n: 512 * 512 for n in range(3, MAX_REFERENCES + 1)}}
JPEG_QUALITY = 92


def reference_size(width: int, height: int, n_refs: int) -> tuple[int, int]:
    if max(width, height) / min(width, height) > MAX_ASPECT_RATIO:
        raise ValueError(f"aspect ratio of {width}x{height} exceeds {MAX_ASPECT_RATIO}:1")
    floor = lambda v: max(REFERENCE_MULTIPLE, int(math.floor(v / REFERENCE_MULTIPLE)) * REFERENCE_MULTIPLE)
    cap = MAX_PIXELS_BY_REFS[n_refs]
    w = max(REFERENCE_MULTIPLE, round(width / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    h = max(REFERENCE_MULTIPLE, round(height / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    if w * h > cap:
        beta = math.sqrt(width * height / cap)
        w, h = floor(width / beta), floor(height / beta)
    if w * h > cap:
        if w == REFERENCE_MULTIPLE:
            h = floor(cap / w)
        else:
            w = floor(cap / h)
    return w, h


def policy_reference(image: Image.Image, n_refs: int) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    image = image.convert("RGB")
    size = reference_size(image.width, image.height, n_refs)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


class Upsampler:
    """Prompt upsampler adapted for Intel Arc XPU."""

    def __init__(self, model: str = MODEL, device: str = "xpu"):
        self.model_path = model
        self.device = device
        self._llm = None
        self._processor = None
        self._tokenizer = None
        self._hf_model = None

        shipped = Path(model) / "system_prompts.json"
        self.prompts = json.loads((shipped if shipped.exists() else PROMPTS).read_text())

    def _load_hf(self):
        if self._hf_model is None:
            from transformers import AutoModelForCausalLM, AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
            self._tokenizer = self._processor.tokenizer
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).to(self.device).eval()

    def unload(self):
        """Free model from XPU memory."""
        if self._hf_model is not None:
            del self._hf_model
            self._hf_model = None
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        return self._tokenizer

    def __call__(self, prompt: str, images: list[Image.Image], boxes: list[dict] | None = None) -> str:
        if len(images) > MAX_REFERENCES:
            raise ValueError(f"the upsampler takes at most {MAX_REFERENCES} references")

        task = "i2i" if images else "t2i"
        system_key, prompt, locked = bbox.plan(prompt, boxes, len(images))

        self._load_hf()

        if images:
            references = [policy_reference(image, len(images)) for image in images]
            scaffold = I2I_SCAFFOLD[locked is not None].format(instruction=prompt)
            content = [{"type": "image"} for _ in references] + [{"type": "text", "text": scaffold}]
            messages = [{"role": "system", "content": self.prompts[system_key]}, {"role": "user", "content": content}]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=references, return_tensors="pt").to(self.device)
        else:
            messages = [{"role": "system", "content": self.prompts[system_key]}, {"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._hf_model.generate(
                **inputs,
                max_new_tokens=SAMPLING["max_new_tokens"],
                temperature=SAMPLING["temperature"] or 1e-5,
                top_p=SAMPLING["top_p"],
                do_sample=False
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        caption = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if not caption:
            caption = prompt

        return bbox.reconcile(caption, locked) if locked else caption


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", action="append", default=[], help=f"reference image; repeat for several (max {MAX_REFERENCES})")
    parser.add_argument("--boxes", help="JSON list of boxes, see README")
    parser.add_argument("--out", help="write the caption here")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default="xpu", help="device to run on (xpu / cpu)")
    args = parser.parse_args()
    boxes = json.loads(Path(args.boxes).read_text()) if args.boxes else None
    upsampler = Upsampler(args.model, args.device)
    caption = upsampler(args.prompt, [Image.open(p) for p in args.image], boxes)
    upsampler.unload()
    if args.out:
        Path(args.out).write_text(caption + "\n")
    print(f"\n{caption}", flush=True)


if __name__ == "__main__":
    main()

