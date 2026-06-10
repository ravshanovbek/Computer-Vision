"""
Image Colorization — Gradio web app.

Loads the pretrained model from the Fisherman Team repo
(https://github.com/ravshanovbek/image_colorization) and serves a small
web UI for drag-and-drop colorization of grayscale images.

Usage:
    1. Place `final_pretrained_model.pth` (or `final_pretrained_model_30.pth`)
       in this directory — download from the repo above.
    2. pip install -r requirements.txt
    3. python app.py
    4. Open the URL printed in the terminal (default: http://127.0.0.1:7860)
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

from model import load_pretrained

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = "final_pretrained_model.pth"
FALLBACK_WEIGHTS = "final_pretrained_model_30.pth"
INPUT_SIZE = 128          # the original model was trained at 128x128
def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon GPU (M1/M2/M3/M4)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _pick_device()

# Theme colors (matching the project deck)
INK = "#1F2937"
CORAL = "#E07856"
PAPER = "#F5F1EA"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _find_weights() -> str:
    """Locate a pretrained weights file in the current directory."""
    here = Path(__file__).parent.resolve()
    for name in (DEFAULT_WEIGHTS, FALLBACK_WEIGHTS):
        path = here / name
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        f"No pretrained weights found. Place one of "
        f"[{DEFAULT_WEIGHTS}, {FALLBACK_WEIGHTS}] next to app.py.\n"
        f"You can download them from "
        f"https://github.com/ravshanovbek/image_colorization"
    )


print(f"[colorize] Device: {DEVICE}")
weights_path = _find_weights()
print(f"[colorize] Loading weights: {weights_path}")
model = load_pretrained(weights_path, device=DEVICE)
print("[colorize] Model loaded.")


# ---------------------------------------------------------------------------
# Preprocessing — mirrors camera.py from the repo
# ---------------------------------------------------------------------------
def preprocess(img_rgb: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Take an RGB uint8 image (any size) and return:
        - a [1,3,128,128] float tensor in [0,1] that is the grayscale-replicated input
        - the original (H, W) so we can resize the result back

    Steps:
        1. Center-square-crop to avoid stretching.
        2. Resize to 128x128 (the training resolution).
        3. Convert to grayscale, then replicate to 3 channels.
        4. HWC -> CHW, add batch, normalize to [0,1].
    """
    h, w = img_rgb.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    cropped = img_rgb[y0 : y0 + side, x0 : x0 + side]

    resized = cv2.resize(cropped, (INPUT_SIZE, INPUT_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    gray3 = cv2.merge([gray, gray, gray])  # H, W, 3

    tensor = torch.from_numpy(gray3).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(DEVICE)
    return tensor, (side, side)


def postprocess(out: torch.Tensor, out_size: tuple[int, int]) -> np.ndarray:
    """Tensor [1,3,128,128] in [0,1] -> uint8 RGB image at out_size."""
    arr = out.squeeze(0).detach().cpu().numpy()  # 3,128,128
    arr = np.transpose(arr, (1, 2, 0))           # 128,128,3
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    h, w = out_size
    if (h, w) != (INPUT_SIZE, INPUT_SIZE):
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_CUBIC)
    return arr


def make_grayscale_preview(img_rgb: np.ndarray) -> np.ndarray:
    """For the 'before' panel: same center-square crop, displayed as gray."""
    h, w = img_rgb.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    cropped = img_rgb[y0 : y0 + side, x0 : x0 + side]
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    return cv2.merge([gray, gray, gray])


# ---------------------------------------------------------------------------
# Inference handler
# ---------------------------------------------------------------------------
@torch.no_grad()
def colorize(image: np.ndarray | None):
    """Gradio callback. `image` arrives as an HxWx3 uint8 RGB ndarray."""
    if image is None:
        return None, None, "Please upload an image."

    # If the user uploaded a single-channel image, broadcast it.
    if image.ndim == 2:
        image = cv2.merge([image, image, image])
    if image.shape[2] == 4:  # strip alpha
        image = image[:, :, :3]

    gray_preview = make_grayscale_preview(image)
    tensor, original_size = preprocess(image)
    output = model(tensor)
    colorized = postprocess(output, original_size)

    info = (
        f"Done. Inference on **{DEVICE.upper()}** at "
        f"{INPUT_SIZE}×{INPUT_SIZE}, then up-scaled to "
        f"{original_size[1]}×{original_size[0]}."
    )
    return gray_preview, colorized, info


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = f"""
.gradio-container {{
    max-width: 1100px !important;
    margin: 0 auto !important;
    font-family: 'Calibri', 'Segoe UI', sans-serif !important;
}}
#app-title h1 {{
    color: {INK};
    font-family: 'Cambria', Georgia, serif !important;
    font-size: 2.4rem !important;
    margin-bottom: 0.2rem !important;
}}
#app-title h1 em {{ color: {CORAL}; font-style: italic; }}
#app-sub {{ color: #6B7280; font-size: 0.95rem; margin-top: 0; }}
.eyebrow {{
    color: {CORAL};
    letter-spacing: 0.18em;
    font-weight: 700;
    font-size: 0.75rem;
    text-transform: uppercase;
}}
footer {{ visibility: hidden; }}
"""

with gr.Blocks(title="Image Colorization") as demo:
    gr.HTML(
        """
        <div id="app-title">
          <p class="eyebrow">Fisherman Team · Computer Vision</p>
          <h1>Image Colorization <em>— from grayscale to color</em></h1>
          <p id="app-sub">
            A custom PyTorch CNN encoder–decoder. Upload any image —
            we'll grayscale it and let the model predict the color.
          </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(
                label="Upload an image",
                type="numpy",
                image_mode="RGB",
                sources=["upload", "clipboard", "webcam"],
                height=360,
            )
            with gr.Row():
                btn = gr.Button("Colorize", variant="primary", scale=2)
                clear = gr.Button("Clear", scale=1)
            status = gr.Markdown("")

        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("Side by side"):
                    with gr.Row():
                        gray_out = gr.Image(label="Input (grayscale)", height=260, interactive=False)
                        color_out = gr.Image(label="Colorized", height=260, interactive=False)
                with gr.Tab("Result only"):
                    big_out = gr.Image(label="Colorized output", height=540, interactive=False)

    def _run(image):
        g, c, msg = colorize(image)
        return g, c, c, msg

    btn.click(_run, inputs=inp, outputs=[gray_out, color_out, big_out, status])
    inp.change(_run, inputs=inp, outputs=[gray_out, color_out, big_out, status])

    def _clear():
        return None, None, None, None, ""

    clear.click(_clear, outputs=[inp, gray_out, color_out, big_out, status])

    gr.Markdown(
        f"""
        ---
        **About this model.** Trained from scratch on paired grayscale/RGB images
        at 128×128. Inputs at other resolutions are center-cropped, processed at
        128×128, then resized back. Saturation may be slightly muted — a known
        trait of MSE-based colorizers.

        **Repo:** https://github.com/ravshanovbek/image_colorization
        """
    )


if __name__ == "__main__":
    # Gradio 6.0: theme and css go on launch(), not Blocks()
    demo.launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
        css=CUSTOM_CSS,
    )
