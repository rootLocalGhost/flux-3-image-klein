"""Gradio Web UI for FLUX 3 Image Klein on Intel Arc A770 (torch.xpu).

Run with:
  python scripts_xpu/app_xpu.py [--port 7860] [--share] [--no-pup]
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Add src to python path
src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont

from flux3_xpu import bbox
from flux3_xpu.generate import Models, Refused, generate
from flux3_xpu.pipeline import MAX_REFERENCES, MODELS, WEIGHTS, dit_path
from flux3_xpu.safety import exif

SIZES = {
    "1:1": (1024, 1024), "4:3": (1184, 880), "3:4": (880, 1184), "3:2": (1248, 832),
    "2:3": (832, 1248), "16:9": (1360, 768), "9:16": (768, 1360),
}
GRID = 64
TITLE = "FLUX 3 Image Klein (Intel Arc A770 XPU)"
SUBTITLE = "Reference inference running on Intel Arc A770 GPU with native torch.xpu and zero-RAM overhead."
CSS = (
    "#header {text-align: center; margin: 8px 0 4px} #header h1 {margin-bottom: 4px} #header p {opacity: .7}"
    " button[aria-label^='Rotate'] {display: none}"
)
OUTPUT_DIR = tempfile.mkdtemp(prefix="demo-xpu-")


def blank_canvas(size: str) -> Image.Image:
    width, height = SIZES[size]
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for x in range(GRID, width, GRID):
        draw.line([(x, 0), (x, height)], fill=(232, 232, 232))
    for y in range(GRID, height, GRID):
        draw.line([(0, y), (width, y)], fill=(232, 232, 232))
    return canvas


def gallery_images(gallery) -> list[Image.Image]:
    items = [item[0] if isinstance(item, (tuple, list)) else item for item in gallery or []]
    return [item if isinstance(item, Image.Image) else Image.open(item) for item in items]


def canvas_boxes(annotation) -> list[dict]:
    if not annotation or annotation.get("image") is None:
        return []
    width, height = annotation["image"].size
    boxes = []
    for box in annotation.get("boxes") or []:
        y1, x1 = max(0.0, box["ymin"] / height), max(0.0, box["xmin"] / width)
        y2, x2 = min(1.0, box["ymax"] / height), min(1.0, box["xmax"] / width)
        boxes.append({
            "id": f"obj_{len(boxes) + 1}",
            "desc": (box.get("label") or "").strip(),
            "bbox": [round(v, 4) for v in (y1, x1, y2, x2)],
        })
    return boxes


def rescale(annotation, canvas: Image.Image) -> dict:
    old = annotation.get("image") if annotation else None
    boxes = []
    for box in (annotation or {}).get("boxes") or []:
        sx, sy = canvas.width / old.width, canvas.height / old.height
        boxes.append({
            **box,
            "xmin": box["xmin"] * sx, "xmax": box["xmax"] * sx,
            "ymin": box["ymin"] * sy, "ymax": box["ymax"] * sy,
        })
    return {"image": canvas, "boxes": boxes}


def output_boxes(caption: str) -> list[dict]:
    rows = bbox.split(caption)[1] or []
    boxes = []
    for row in rows:
        box = bbox.valid_box(row.get("bbox") or row.get("tgt_bbox")) if isinstance(row, dict) else None
        if box is not None:
            boxes.append({"id": str(row.get("id", "")), "bbox": [v / bbox.SCALE for v in box]})
    return boxes


def overlay(image: Image.Image, boxes: list[dict]) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    line = max(2, round(image.width / 400))
    font = ImageFont.load_default(size=max(12, image.width // 60))
    for box in boxes:
        y1, x1, y2, x2 = box["bbox"]
        rect = (x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height)
        draw.rectangle(rect, outline=(239, 64, 53), width=line)
        draw.rectangle([v + (line if i < 2 else -line) for i, v in enumerate(rect)], outline="white", width=1)
        if box["id"]:
            left, top, right, bottom = draw.textbbox((rect[0], rect[1]), box["id"], font=font)
            pad = line + 2
            draw.rectangle((left, top, right + 2 * pad, bottom + 2 * pad), fill=(239, 64, 53))
            draw.text((rect[0] + pad, rect[1] + pad), box["id"], fill="white", font=font)
    return image


def build(models: Models, has_pup: bool) -> gr.Blocks:
    try:
        from gradio_image_annotation import image_annotator
        has_annotator = True
    except ImportError:
        has_annotator = False

    def run(model, use_pup, seed, steps, guidance, show, safety, prompt, gallery, boxes, size):
        images = gallery_images(gallery)
        if not prompt or not prompt.strip():
            raise gr.Error("Write a prompt first.")
        width, height = (None, None) if images or size is None else SIZES[size]
        if images and boxes:
            boxes = [{"id": b["id"], "desc": b["desc"], "from": "target", "src_bbox": None, "tgt_bbox": b["bbox"]}
                     for b in boxes]
        try:
            image, record = generate(
                models, model, prompt, images, boxes or None, width, height,
                steps=int(steps) if steps else None,
                guidance=guidance if MODELS[model]["guidance"] is not None else None,
                seed=None if seed is None or seed < 0 else int(seed), pup=use_pup,
                safety=safety
            )
        except Refused as error:
            raise gr.Error(f"{error}", title="Blocked by safety check")

        path = tempfile.NamedTemporaryFile(suffix=".png", dir=OUTPUT_DIR, delete=False).name
        image.save(path, exif=exif(model))
        drawn = output_boxes(record["prompt"])
        boxed = None
        if drawn:
            boxed = path[:-4] + "_boxes.png"
            overlay(image, drawn).save(boxed)
        toggle = gr.Checkbox(label=f"Show boxes on the output ({len(drawn)})" if drawn else "No boxes",
                             interactive=bool(drawn), value=bool(drawn) and show)
        return (boxed if boxed and show else path), (path, boxed), toggle, path, record["prompt"], record

    refs = lambda label: gr.Gallery(type="pil", label=label, columns=4, height="auto", object_fit="contain",
                                    allow_preview=False, interactive=True)

    available_models = list(models.pipeline.dits_paths.keys())

    with gr.Blocks(title=TITLE) as demo:
        gr.HTML(f"<div id='header'><h1>{TITLE}</h1><p>{SUBTITLE}</p></div>")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                with gr.Row():
                    model = gr.Radio(available_models, value=available_models[0], label="Model")
                    use_pup = gr.Checkbox(value=has_pup, interactive=has_pup, label="Prompt upsampling")
                with gr.Tabs():
                    with gr.Tab("Text to image"):
                        t2i_prompt = gr.Textbox(
                            label="Prompt", lines=4,
                            value="A cinematic photograph of an astronaut exploring a vibrant alien rainforest, biomorphic flora, volumetric mist, 8k",
                            placeholder="Type what you want to generate..."
                        )
                        t2i_size = gr.Radio(list(SIZES), value="1:1", label="Size")
                        t2i_go = gr.Button("Generate", variant="primary", size="lg")
                    with gr.Tab("Edit"):
                        edit_prompt = gr.Textbox(label="Instruction", lines=3, placeholder="Change the background to a snowy mountain")
                        edit_refs = refs(f"References (1-{MAX_REFERENCES}; the output follows the first)")
                        edit_go = gr.Button("Generate", variant="primary", size="lg")
                    if has_annotator:
                        with gr.Tab("Layout"):
                            layout_prompt = gr.Textbox(label="Prompt", lines=2, placeholder="hot-air balloons over a valley at sunrise")
                            layout_size = gr.Radio(list(SIZES), value="1:1", label="Canvas")
                            canvas = image_annotator(value={"image": blank_canvas("1:1"), "boxes": []}, image_type="pil",
                                                     label="Draw a box, then type what goes in it", sources=[], height=560,
                                                     show_clear_button=False, show_remove_button=True)
                            with gr.Accordion(f"References (optional, up to {MAX_REFERENCES}; boxes go on the first)", open=False):
                                layout_refs = refs("References")
                            layout_go = gr.Button("Generate", variant="primary", size="lg")
                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        seed = gr.Number(value=-1, precision=0, label="Seed (-1: random)")
                        steps = gr.Number(value=None, precision=0, label="Steps (empty: default)")
                        guidance = gr.Slider(1.0, 10.0, value=4.0, step=0.1, label="CFG scale (base only)")
                    safety = gr.Checkbox(value=False, label="Safety checks")
            with gr.Column(scale=1):
                output = gr.Image(type="filepath", label="Output", format="png", height=640, interactive=False)
                show_boxes = gr.Checkbox(value=True, interactive=False, label="No boxes")
                paths = gr.State((None, None))
                caption = gr.Textbox(label="Prompt the model saw", lines=4, max_lines=12)
                with gr.Accordion("Record", open=False):
                    download = gr.File(label="Download (with EXIF)")
                    record = gr.JSON(label="Record")

        shared = [model, use_pup, seed, steps, guidance, show_boxes, safety]
        outputs = [output, paths, show_boxes, download, caption, record]

        t2i_go.click(lambda *a: run(*a[:7], a[7], None, [], a[8]), shared + [t2i_prompt, t2i_size], outputs)
        edit_go.click(lambda *a: run(*a[:7], a[7], a[8], [], None), shared + [edit_prompt, edit_refs], outputs)
        if has_annotator:
            layout_go.click(lambda *a: run(*a[:7], a[7], a[8], canvas_boxes(a[9]), a[10]),
                            shared + [layout_prompt, layout_refs, canvas, layout_size], outputs)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true", help="Public Gradio link")
    parser.add_argument("--device", default="xpu", help="Target device (xpu / cpu)")
    parser.add_argument("--no-pup", action="store_true", help="Do not load the prompt upsampler")
    args = parser.parse_args()

    effective_device = args.device
    if effective_device.startswith("xpu") and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("[Notice] torch.xpu not detected; running Gradio demo on CPU.")
        effective_device = "cpu"

    dits = {name: str(dit_path(name)) for name in MODELS if dit_path(name).exists()}
    if not dits:
        print(f"[Warning] No downloaded checkpoints found under {WEIGHTS}.")
        return

    has_pup = not args.no_pup and (WEIGHTS / "pup").exists()
    models = Models(
        dits=dits,
        pup=str(WEIGHTS / "pup") if has_pup else None,
        device=effective_device,
        sequential=True,
    )
    build(models, has_pup).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
