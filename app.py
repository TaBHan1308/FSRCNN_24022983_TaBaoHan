import inspect
import base64
import html
import os
import shutil
import subprocess
import time
import uuid

import cv2
import gradio as gr
import numpy as np
import torch

import imgproc
from model import FSRCNN


SCALE = 3
CHECKPOINT_CANDIDATES = [
    "best_epoch300_psnr3299.pth.tar",
    "best.pth.tar",
    "weights/best_epoch300_psnr3299.pth.tar",
    "weights/best.pth.tar",
    "results/fsrcnn_y_x3_pretrained_mixed/best_epoch300_psnr3299.pth.tar",
    "results/fsrcnn_y_x3_pretrained_mixed/best.pth.tar",
]
OUTPUT_DIR = "outputs_hf"
COMPARE_JS = r"""
() => {
  if (window.__fsrcnnCompareLensInstalled) {
    return;
  }
  window.__fsrcnnCompareLensInstalled = true;

  function updateFromEvent(event, activePane) {
    const root = activePane.closest(".sr-compare-wrap");
    if (!root) return;

    const panes = Array.from(root.querySelectorAll(".sr-pane"));
    const radius = Number(root.dataset.radius || 88);
    const zoom = Number(root.dataset.zoom || 2.4);
    const rect = activePane.getBoundingClientRect();
    const xRatio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    const yRatio = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));

    panes.forEach((pane) => {
      const img = pane.querySelector("img");
      const lens = pane.querySelector(".sr-lens");
      const h = pane.querySelector(".sr-crosshair-h");
      const v = pane.querySelector(".sr-crosshair-v");
      if (!img || !lens || !h || !v) return;

      const paneWidth = pane.clientWidth;
      const paneHeight = img.clientHeight;
      const x = xRatio * paneWidth;
      const y = yRatio * paneHeight;

      lens.style.display = "block";
      h.style.display = "block";
      v.style.display = "block";
      lens.style.left = `${x - radius}px`;
      lens.style.top = `${y - radius}px`;
      h.style.top = `${y}px`;
      v.style.left = `${x}px`;
      lens.style.backgroundImage = `url(${img.src})`;
      lens.style.backgroundSize = `${paneWidth * zoom}px ${paneHeight * zoom}px`;
      lens.style.backgroundPosition = `${radius - x * zoom}px ${radius - y * zoom}px`;
    });
  }

  function hide(root) {
    if (!root) return;
    root.querySelectorAll(".sr-lens,.sr-crosshair-h,.sr-crosshair-v").forEach((el) => {
      el.style.display = "none";
    });
  }

  document.addEventListener("mousemove", (event) => {
    const pane = event.target.closest && event.target.closest(".sr-pane");
    if (!pane) return;
    updateFromEvent(event, pane);
  });

  document.addEventListener("mouseleave", (event) => {
    const pane = event.target.closest && event.target.closest(".sr-pane");
    if (!pane) return;
    hide(pane.closest(".sr-compare-wrap"));
  }, true);
}
"""


def resolve_checkpoint():
    for checkpoint_path in CHECKPOINT_CANDIDATES:
        if os.path.exists(checkpoint_path):
            return checkpoint_path
    raise FileNotFoundError(
        "No checkpoint found. Tried:\n" + "\n".join(f"- {path}" for path in CHECKPOINT_CANDIDATES)
    )


def build_model(scale):
    signature = inspect.signature(FSRCNN)
    if "upscale_factor" in signature.parameters:
        return FSRCNN(upscale_factor=scale)
    if "scale_factor" in signature.parameters:
        return FSRCNN(scale_factor=scale)
    return FSRCNN(scale)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for key, value in state_dict.items():
        clean_state_dict[key.replace("module.", "")] = value

    model.load_state_dict(clean_state_dict)
    return model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

CHECKPOINT_PATH = resolve_checkpoint()
MODEL = build_model(SCALE).to(DEVICE)
MODEL = load_checkpoint(MODEL, CHECKPOINT_PATH, DEVICE)
MODEL.eval()


def resize_keep_aspect(image, max_width=None, max_height=None):
    height, width = image.shape[:2]
    if max_width is None and max_height is None:
        return image

    scale = 1.0
    if max_width is not None:
        scale = min(scale, max_width / width)
    if max_height is not None:
        scale = min(scale, max_height / height)

    if scale >= 1.0:
        return image

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def crop_to_multiple(image, multiple):
    height, width = image.shape[:2]
    cropped_height = height - (height % multiple)
    cropped_width = width - (width % multiple)
    if cropped_height <= 0 or cropped_width <= 0:
        return image

    top = (height - cropped_height) // 2
    left = (width - cropped_width) // 2
    return image[top:top + cropped_height, left:left + cropped_width]


def resize_to_height(image, target_height):
    height, width = image.shape[:2]
    scale = target_height / height
    target_width = max(1, int(round(width * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def draw_label(image, label):
    labeled = image.copy()
    overlay = labeled.copy()
    cv2.rectangle(overlay, (0, 0), (labeled.shape[1], 44), (0, 0, 0), -1)
    labeled = cv2.addWeighted(overlay, 0.45, labeled, 0.55, 0)
    cv2.putText(
        labeled,
        label,
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def ensure_even_frame(image):
    height, width = image.shape[:2]
    even_height = height - (height % 2)
    even_width = width - (width % 2)
    if even_height == height and even_width == width:
        return image
    return image[:even_height, :even_width]


def get_ffmpeg_executable():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def convert_to_browser_mp4(input_path):
    ffmpeg = get_ffmpeg_executable()
    if ffmpeg is None or not os.path.exists(input_path):
        return input_path

    base, _ = os.path.splitext(input_path)
    output_path = f"{base}_browser.mp4"
    command = [
        ffmpeg,
        "-y",
        "-i",
        input_path,
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
    except Exception:
        pass

    return input_path


def make_comparison(input_rgb, sr_rgb, panel_height=480):
    input_panel = resize_to_height(input_rgb, panel_height)
    sr_panel = resize_to_height(sr_rgb, panel_height)
    input_panel = draw_label(input_panel, f"Input ({input_rgb.shape[1]}x{input_rgb.shape[0]})")
    sr_panel = draw_label(sr_panel, f"FSRCNN-Y x3 Output ({sr_rgb.shape[1]}x{sr_rgb.shape[0]})")
    return np.hstack([input_panel, sr_panel])


def make_video_comparison_bgr(input_bgr, sr_bgr):
    sr_h, sr_w = sr_bgr.shape[:2]
    input_upscaled = cv2.resize(input_bgr, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)
    input_panel = draw_label(input_upscaled, "Input x3 preview")
    sr_panel = draw_label(sr_bgr, "FSRCNN-Y x3 Output")
    return np.hstack([input_panel, sr_panel])


def apply_circular_magnifier(image, center_x, center_y, radius, zoom):
    height, width = image.shape[:2]
    radius = int(max(12, radius))
    zoom = max(1.0, float(zoom))
    center_x = int(np.clip(center_x, 0, width - 1))
    center_y = int(np.clip(center_y, 0, height - 1))

    src_radius = max(2, int(radius / zoom))
    src_left = max(0, center_x - src_radius)
    src_right = min(width, center_x + src_radius)
    src_top = max(0, center_y - src_radius)
    src_bottom = min(height, center_y + src_radius)
    crop = image[src_top:src_bottom, src_left:src_right]
    if crop.size == 0:
        return image

    lens_size = radius * 2
    magnified = cv2.resize(crop, (lens_size, lens_size), interpolation=cv2.INTER_CUBIC)

    dst_left = max(0, center_x - radius)
    dst_right = min(width, center_x + radius)
    dst_top = max(0, center_y - radius)
    dst_bottom = min(height, center_y + radius)

    mag_left = dst_left - (center_x - radius)
    mag_right = mag_left + (dst_right - dst_left)
    mag_top = dst_top - (center_y - radius)
    mag_bottom = mag_top + (dst_bottom - dst_top)

    output = image.copy()
    roi = output[dst_top:dst_bottom, dst_left:dst_right]
    mag_roi = magnified[mag_top:mag_bottom, mag_left:mag_right]

    yy, xx = np.ogrid[dst_top:dst_bottom, dst_left:dst_right]
    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
    roi[mask] = mag_roi[mask]

    cv2.circle(output, (center_x, center_y), radius, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(output, (center_x, center_y), radius + 3, (0, 0, 0), 2, cv2.LINE_AA)
    return output


def make_magnifier_comparison(
    input_rgb,
    sr_rgb,
    lens_x_percent,
    lens_y_percent,
    lens_radius,
    lens_zoom,
    panel_height=520,
):
    sr_h, sr_w = sr_rgb.shape[:2]
    input_upscaled = cv2.resize(input_rgb, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)

    center_x = int(sr_w * float(lens_x_percent) / 100.0)
    center_y = int(sr_h * float(lens_y_percent) / 100.0)
    input_lens = apply_circular_magnifier(input_upscaled, center_x, center_y, lens_radius, lens_zoom)
    sr_lens = apply_circular_magnifier(sr_rgb, center_x, center_y, lens_radius, lens_zoom)

    input_panel = resize_to_height(input_lens, panel_height)
    sr_panel = resize_to_height(sr_lens, panel_height)
    input_panel = draw_label(input_panel, "Input x3 preview")
    sr_panel = draw_label(sr_panel, "FSRCNN-Y x3 Output")
    return np.hstack([input_panel, sr_panel])


def rgb_to_data_url(image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        return ""
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def make_interactive_compare_html(input_rgb, sr_rgb, lens_radius=88, lens_zoom=2.4):
    sr_h, sr_w = sr_rgb.shape[:2]
    input_upscaled = cv2.resize(input_rgb, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)
    input_url = rgb_to_data_url(input_upscaled)
    sr_url = rgb_to_data_url(sr_rgb)
    uid = f"sr_compare_{uuid.uuid4().hex}"

    inner_html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body {{ margin: 0; padding: 0; background: #111; color: #eee; font-family: Arial, sans-serif; }}
  .sr-compare-wrap {{ width: 100%; box-sizing: border-box; padding: 8px; }}
  .sr-help {{ margin: 0 0 8px 0; color: #d7d7d7; font-size: 14px; }}
  .sr-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: start; }}
  .sr-pane {{ position: relative; overflow: hidden; background: #050505; border: 1px solid #3a3a3a; border-radius: 8px; cursor: crosshair; }}
  .sr-pane img {{ display: block; width: 100%; height: auto; user-select: none; pointer-events: none; }}
  .sr-title {{ position: absolute; left: 0; top: 0; right: 0; z-index: 3; padding: 8px 12px; background: rgba(0, 0, 0, 0.48); color: white; font-weight: 700; font-size: 15px; text-shadow: 0 1px 2px black; pointer-events: none; }}
  .sr-lens {{ position: absolute; width: {lens_radius * 2}px; height: {lens_radius * 2}px; border-radius: 999px; border: 3px solid rgba(255, 255, 255, 0.95); box-shadow: 0 0 0 2px rgba(0,0,0,0.75), 0 8px 24px rgba(0,0,0,0.45); display: none; pointer-events: none; z-index: 4; background-repeat: no-repeat; }}
  .sr-crosshair-h, .sr-crosshair-v {{ position: absolute; display: none; pointer-events: none; z-index: 2; background: rgba(255, 255, 255, 0.45); }}
  .sr-crosshair-h {{ height: 1px; left: 0; right: 0; }}
  .sr-crosshair-v {{ width: 1px; top: 0; bottom: 0; }}
  @media (max-width: 900px) {{ .sr-row {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div id="{uid}" class="sr-compare-wrap" data-radius="{lens_radius}" data-zoom="{lens_zoom}">
  <div class="sr-help">Move the mouse over either image. Both circular lenses follow the same relative location for direct comparison.</div>
  <div class="sr-row">
    <div class="sr-pane sr-left">
      <img src="{input_url}" alt="Input x3 preview">
      <div class="sr-title">Input x3 preview</div>
      <div class="sr-crosshair-h"></div><div class="sr-crosshair-v"></div><div class="sr-lens"></div>
    </div>
    <div class="sr-pane sr-right">
      <img src="{sr_url}" alt="FSRCNN-Y x3 Output">
      <div class="sr-title">FSRCNN-Y x3 Output</div>
      <div class="sr-crosshair-h"></div><div class="sr-crosshair-v"></div><div class="sr-lens"></div>
    </div>
  </div>
</div>
<script>
(() => {{
  const root = document.getElementById("{uid}");
  if (!root) return;
  const panes = Array.from(root.querySelectorAll(".sr-pane"));
  const radius = Number(root.dataset.radius || {lens_radius});
  const zoom = Number(root.dataset.zoom || {lens_zoom});

  function updateFromEvent(event, activePane) {{
    const rect = activePane.getBoundingClientRect();
    const xRatio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    const yRatio = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));

    panes.forEach((pane) => {{
      const img = pane.querySelector("img");
      const lens = pane.querySelector(".sr-lens");
      const h = pane.querySelector(".sr-crosshair-h");
      const v = pane.querySelector(".sr-crosshair-v");
      const paneWidth = pane.clientWidth;
      const paneHeight = img.clientHeight;
      const x = xRatio * paneWidth;
      const y = yRatio * paneHeight;

      lens.style.display = "block";
      h.style.display = "block";
      v.style.display = "block";
      lens.style.left = `${{x - radius}}px`;
      lens.style.top = `${{y - radius}}px`;
      h.style.top = `${{y}}px`;
      v.style.left = `${{x}}px`;
      lens.style.backgroundImage = `url(${{img.src}})`;
      lens.style.backgroundSize = `${{paneWidth * zoom}}px ${{paneHeight * zoom}}px`;
      lens.style.backgroundPosition = `${{radius - x * zoom}}px ${{radius - y * zoom}}px`;
    }});
  }}

  function hide() {{
    panes.forEach((pane) => {{
      pane.querySelector(".sr-lens").style.display = "none";
      pane.querySelector(".sr-crosshair-h").style.display = "none";
      pane.querySelector(".sr-crosshair-v").style.display = "none";
    }});
  }}

  panes.forEach((pane) => {{
    pane.addEventListener("mousemove", (event) => updateFromEvent(event, pane));
    pane.addEventListener("mouseleave", hide);
  }});
}})();
</script>
</body>
</html>
"""
    iframe_srcdoc = html.escape(inner_html, quote=True)
    return (
        f'<iframe srcdoc="{iframe_srcdoc}" '
        'style="width:100%; height:680px; border:0; border-radius:8px; background:#111;" '
        'loading="lazy"></iframe>'
    )


def make_compare_placeholder_html():
    return """
<div style="height: 360px; display: flex; align-items: center; justify-content: center;
            background: #151515; border: 1px solid #333; border-radius: 8px; color: #bbb;
            font-family: Arial, sans-serif;">
  Upload an image and click Run Image SR to show the interactive comparison.
</div>
"""


def make_native_sr_comparison(input_rgb, sr_rgb):
    sr_h, sr_w = sr_rgb.shape[:2]
    input_upscaled = cv2.resize(input_rgb, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)
    input_panel = draw_label(input_upscaled, "Input bicubic x3 preview")
    sr_panel = draw_label(sr_rgb, "FSRCNN-Y x3 Output")
    return np.hstack([input_panel, sr_panel])


def center_crop(image, crop_ratio=0.45):
    height, width = image.shape[:2]
    crop_height = max(1, int(height * crop_ratio))
    crop_width = max(1, int(width * crop_ratio))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return image[top:top + crop_height, left:left + crop_width]


def make_zoom_comparison(input_rgb, sr_rgb, panel_height=420):
    sr_h, sr_w = sr_rgb.shape[:2]
    input_upscaled = cv2.resize(input_rgb, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)
    input_crop = center_crop(input_upscaled)
    sr_crop = center_crop(sr_rgb)
    input_panel = resize_to_height(input_crop, panel_height)
    sr_panel = resize_to_height(sr_crop, panel_height)
    input_panel = draw_label(input_panel, "Input x3 preview - center crop")
    sr_panel = draw_label(sr_panel, "FSRCNN-Y x3 - center crop")
    return np.hstack([input_panel, sr_panel])


def describe_difference(input_rgb, sr_rgb):
    sr_h, sr_w = sr_rgb.shape[:2]
    input_upscaled = cv2.resize(input_rgb, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)
    diff = np.abs(sr_rgb.astype(np.float32) - input_upscaled.astype(np.float32))
    return (
        f"Input used by model: {input_rgb.shape[1]}x{input_rgb.shape[0]} | "
        f"FSRCNN output: {sr_rgb.shape[1]}x{sr_rgb.shape[0]} | "
        f"Mean abs diff vs input x3 preview: {float(diff.mean()):.2f} | "
        f"Max diff: {int(diff.max())}"
    )


def make_diff_heatmap(input_rgb, sr_rgb):
    sr_h, sr_w = sr_rgb.shape[:2]
    input_upscaled = cv2.resize(input_rgb, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)
    diff = np.mean(np.abs(sr_rgb.astype(np.float32) - input_upscaled.astype(np.float32)), axis=2)
    if diff.max() > 0:
        diff = diff / diff.max() * 255.0
    diff_uint8 = diff.round().astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_TURBO)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return draw_label(heatmap_rgb, "Difference heatmap vs input bicubic x3")


def visual_sharpen_rgb(image_rgb, amount):
    if amount <= 0:
        return image_rgb

    blurred = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(image_rgb, 1.0 + amount, blurred, -amount, 0)
    return sharpened.clip(0, 255).astype(np.uint8)


def prepare_input_bgr(frame_bgr, max_input_width):
    frame_bgr = resize_keep_aspect(frame_bgr, max_width=int(max_input_width))
    return crop_to_multiple(frame_bgr, SCALE)


def get_y_channel_bgr(frame_bgr):
    y = imgproc.bgr2ycbcr(frame_bgr.astype(np.float32) / 255.0, use_y_channel=True)
    return (y * 255.0).round().clip(0, 255).astype(np.uint8)


def y_to_tensor(y_channel):
    y = y_channel.astype(np.float32) / 255.0
    tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)


def tensor_to_y(tensor):
    y = tensor.squeeze(0).squeeze(0).detach().cpu().clamp(0, 1).numpy()
    return (y * 255.0).round().clip(0, 255).astype(np.uint8)


def fsrcnn_y_upscale_bgr(frame_bgr):
    lr_h, lr_w = frame_bgr.shape[:2]
    sr_h = lr_h * SCALE
    sr_w = lr_w * SCALE

    lr_y = get_y_channel_bgr(frame_bgr)
    lr_ycbcr = imgproc.bgr2ycbcr(frame_bgr.astype(np.float32) / 255.0, use_y_channel=False)
    _, lr_cb, lr_cr = cv2.split(lr_ycbcr)

    with torch.inference_mode():
        sr_y_tensor = MODEL(y_to_tensor(lr_y))

    sr_y = tensor_to_y(sr_y_tensor)
    if sr_y.shape[:2] != (sr_h, sr_w):
        sr_y = cv2.resize(sr_y, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)

    sr_cb = cv2.resize(lr_cb, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)
    sr_cr = cv2.resize(lr_cr, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)
    sr_ycbcr = cv2.merge([sr_y.astype(np.float32) / 255.0, sr_cb, sr_cr])
    sr_bgr = imgproc.ycbcr2bgr(sr_ycbcr)
    return (sr_bgr * 255.0).round().clip(0, 255).astype(np.uint8)


def run_image_sr(
    input_rgb,
    max_input_width,
    use_visual_sharpen,
    sharpen_amount,
    show_heatmap,
):
    if input_rgb is None:
        return None, "", gr.update(value=None, visible=False), "Please upload an image."

    input_bgr = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)
    input_bgr = prepare_input_bgr(input_bgr, max_input_width)
    sr_bgr = fsrcnn_y_upscale_bgr(input_bgr)

    prepared_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
    sr_rgb = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB)
    display_sr_rgb = visual_sharpen_rgb(sr_rgb, float(sharpen_amount)) if use_visual_sharpen else sr_rgb
    comparison_html = make_interactive_compare_html(prepared_rgb, display_sr_rgb)
    diff_heatmap = make_diff_heatmap(prepared_rgb, display_sr_rgb) if show_heatmap else None
    diff_output = gr.update(value=diff_heatmap, visible=bool(show_heatmap))
    info = describe_difference(prepared_rgb, sr_rgb)
    if use_visual_sharpen:
        info += (
            f" | Visual sharpen preview ON, amount={float(sharpen_amount):.2f}. "
            "This is post-processing for display, not the pure FSRCNN metric output."
        )
    return display_sr_rgb, comparison_html, diff_output, info


def get_video_path(video_input):
    if video_input is None:
        return None
    if isinstance(video_input, str):
        return video_input
    if isinstance(video_input, dict):
        return video_input.get("video") or video_input.get("name") or video_input.get("path")
    return getattr(video_input, "name", None)


def process_video_sr(video_input, max_input_width, max_frames, create_compare_video, progress=gr.Progress()):
    video_path = get_video_path(video_input)
    if not video_path:
        raise gr.Error("Please upload a video first.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Cannot open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    requested_max_frames = int(max_frames)
    if requested_max_frames <= 0:
        frame_limit = total_frames if total_frames > 0 else 20000
    else:
        frame_limit = requested_max_frames
        if total_frames > 0:
            frame_limit = min(frame_limit, total_frames)

    sr_path = os.path.join(OUTPUT_DIR, "output_fsrcnn_y_x3.mp4")
    compare_path = os.path.join(OUTPUT_DIR, "compare_input_fsrcnn.mp4")
    sr_writer = None
    compare_writer = None
    processed = 0
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_input_shape = None
    first_sr_shape = None
    start_time = time.perf_counter()

    try:
        while processed < frame_limit:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            input_bgr = prepare_input_bgr(frame_bgr, max_input_width)
            sr_bgr = fsrcnn_y_upscale_bgr(input_bgr)
            sr_write_frame = ensure_even_frame(sr_bgr)
            if first_input_shape is None:
                first_input_shape = input_bgr.shape[:2]
                first_sr_shape = sr_bgr.shape[:2]

            if sr_writer is None:
                sr_h, sr_w = sr_write_frame.shape[:2]
                sr_writer = cv2.VideoWriter(
                    sr_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (sr_w, sr_h),
                )

                if create_compare_video:
                    compare_write_frame = ensure_even_frame(make_video_comparison_bgr(input_bgr, sr_bgr))
                    cmp_h, cmp_w = compare_write_frame.shape[:2]
                    compare_writer = cv2.VideoWriter(
                        compare_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (cmp_w, cmp_h),
                    )

            sr_writer.write(sr_write_frame)

            if create_compare_video and compare_writer is not None:
                compare_writer.write(ensure_even_frame(make_video_comparison_bgr(input_bgr, sr_bgr)))

            processed += 1
            if processed == 1 or processed % 10 == 0 or processed == frame_limit:
                progress(processed / max(frame_limit, 1), desc=f"Processing frame {processed}/{frame_limit}")
    finally:
        cap.release()
        if sr_writer is not None:
            sr_writer.release()
        if compare_writer is not None:
            compare_writer.release()

    if processed == 0:
        raise gr.Error("No video frames were processed.")

    sr_browser_path = convert_to_browser_mp4(sr_path)
    compare_browser_path = convert_to_browser_mp4(compare_path) if create_compare_video else None
    elapsed = max(time.perf_counter() - start_time, 1e-8)
    processing_fps = processed / elapsed
    input_h, input_w = first_input_shape if first_input_shape is not None else (0, 0)
    sr_h, sr_w = first_sr_shape if first_sr_shape is not None else (0, 0)
    duration = processed / fps if fps > 0 else 0.0
    info = (
        f"Source: {source_width}x{source_height} @ {fps:.2f} FPS | "
        f"Processed frames: {processed}/{total_frames if total_frames > 0 else '?'} | "
        f"Output duration: {duration:.2f}s | "
        f"Model input: {input_w}x{input_h} | "
        f"FSRCNN output: {sr_w}x{sr_h} | "
        f"Processing speed: {processing_fps:.2f} FPS | "
        f"Elapsed: {elapsed:.1f}s"
    )
    if sr_w >= 1280:
        info += " | Tip: lower max_input_width to 256-320 if browser preview lags."
    return sr_browser_path, info, compare_browser_path


WEB_CAM_STREAM_COUNTER = 0
WEB_CAM_LAST_OUTPUT = None
WEB_CAM_RECORDING = False
WEB_CAM_WRITER = None
WEB_CAM_RECORD_PATH = os.path.join(OUTPUT_DIR, "webcam_compare_recording.mp4")
WEB_CAM_RECORDED_FRAMES = 0


def run_webcam_sr(input_rgb, webcam_max_width):
    if input_rgb is None:
        return None

    input_bgr = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)
    input_bgr = prepare_input_bgr(input_bgr, webcam_max_width)
    sr_bgr = fsrcnn_y_upscale_bgr(input_bgr)

    sr_rgb = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB)
    return sr_rgb


def set_webcam_realtime(enabled):
    status = "Realtime SR: running" if enabled else "Realtime SR: stopped"
    return enabled, status


def start_webcam_recording():
    global WEB_CAM_RECORDING, WEB_CAM_WRITER, WEB_CAM_RECORDED_FRAMES

    if WEB_CAM_WRITER is not None:
        WEB_CAM_WRITER.release()
        WEB_CAM_WRITER = None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    WEB_CAM_RECORDING = True
    WEB_CAM_RECORDED_FRAMES = 0
    return "Recording webcam compare video..."


def stop_webcam_recording():
    global WEB_CAM_RECORDING, WEB_CAM_WRITER

    WEB_CAM_RECORDING = False
    if WEB_CAM_WRITER is not None:
        WEB_CAM_WRITER.release()
        WEB_CAM_WRITER = None

    if WEB_CAM_RECORDED_FRAMES <= 0 or not os.path.exists(WEB_CAM_RECORD_PATH):
        return "Recording stopped. No frames were recorded.", None

    browser_path = convert_to_browser_mp4(WEB_CAM_RECORD_PATH)
    return f"Recording saved: {browser_path}", browser_path


def run_webcam_stream_sr(input_rgb):
    global WEB_CAM_STREAM_COUNTER, WEB_CAM_LAST_OUTPUT

    if input_rgb is None:
        return gr.update()

    WEB_CAM_STREAM_COUNTER += 1
    process_every_n = 1
    if WEB_CAM_STREAM_COUNTER % process_every_n != 0 and WEB_CAM_LAST_OUTPUT is not None:
        return WEB_CAM_LAST_OUTPUT

    input_bgr = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)
    input_bgr = prepare_input_bgr(input_bgr, 256)
    sr_bgr = fsrcnn_y_upscale_bgr(input_bgr)
    sr_rgb = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB)
    WEB_CAM_LAST_OUTPUT = sr_rgb

    return sr_rgb


def process_current_webcam_frame(input_rgb, webcam_max_width):
    sr_rgb = run_webcam_sr(input_rgb, webcam_max_width)
    if sr_rgb is None:
        return None, "No webcam frame available."
    return sr_rgb, "Processed current webcam frame."


def build_app():
    with gr.Blocks(title="FSRCNN-Y x3 Super-Resolution") as demo:
        gr.Markdown("# FSRCNN-Y x3 Image, Video & Webcam Super-Resolution")
        gr.Markdown(
            "Upload image/video or use webcam. The app applies FSRCNN-Y x3 to generate "
            "higher-resolution output."
        )
        gr.Markdown(f"**Device:** `{DEVICE}`  \n**Checkpoint:** `{CHECKPOINT_PATH}`")

        with gr.Tab("Image Super-Resolution"):
            with gr.Row():
                image_input = gr.Image(label="Input Image", type="numpy")
                image_output = gr.Image(label="FSRCNN-Y x3 Output", type="numpy")
            image_width = gr.Slider(128, 1280, value=640, step=32, label="max_input_width")
            image_sharpen = gr.Checkbox(value=False, label="Visual sharpen preview")
            image_sharpen_amount = gr.Slider(0.0, 1.5, value=0.45, step=0.05, label="sharpen_amount")
            show_heatmap = gr.Checkbox(value=False, label="Show difference heatmap")
            image_button = gr.Button("Run Image SR", variant="primary")
            image_compare = gr.HTML(
                value=make_compare_placeholder_html(),
                label="Compare: Input x3 Preview | FSRCNN-Y x3 Output",
            )
            image_diff = gr.Image(label="Difference Heatmap", type="numpy", visible=False)
            image_info = gr.Textbox(label="Output check", interactive=False)
            image_button.click(
                run_image_sr,
                inputs=[
                    image_input,
                    image_width,
                    image_sharpen,
                    image_sharpen_amount,
                    show_heatmap,
                ],
                outputs=[
                    image_output,
                    image_compare,
                    image_diff,
                    image_info,
                ],
            )

        with gr.Tab("Video Super-Resolution"):
            video_input = gr.Video(label="Input Video")
            video_width = gr.Slider(128, 1280, value=320, step=32, label="max_input_width")
            max_frames = gr.Slider(
                0,
                20000,
                value=0,
                step=10,
                label="max_frames (0 = process full video)",
            )
            create_compare = gr.Checkbox(
                value=False,
                label="Create comparison video: Input x3 preview | FSRCNN-Y x3 Output",
            )
            video_button = gr.Button("Run Video SR", variant="primary")
            with gr.Row():
                video_output = gr.Video(label="FSRCNN-Y x3 Output Video")
                video_info = gr.Textbox(label="Video stats", interactive=False, lines=5)
                compare_output = gr.Video(
                    label="Input | FSRCNN-Y x3 Output Video",
                    visible=False,
                )
            create_compare.change(
                lambda enabled: gr.update(visible=enabled),
                inputs=create_compare,
                outputs=compare_output,
            )
            video_button.click(
                process_video_sr,
                inputs=[video_input, video_width, max_frames, create_compare],
                outputs=[video_output, video_info, compare_output],
            )

        with gr.Tab("Real-time Webcam"):
            with gr.Row():
                webcam_input = gr.Image(
                    label="Live Webcam Input",
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    interactive=True,
                )
                webcam_output = gr.Image(label="FSRCNN-Y x3 Output", type="numpy")

            webcam_input.stream(
                run_webcam_stream_sr,
                inputs=webcam_input,
                outputs=webcam_output,
            )

        with gr.Accordion("About", open=False):
            gr.Markdown(
                "- The model processes the Y channel in YCbCr color space.\n"
                "- Cr/Cb channels are resized with Bicubic interpolation and merged back with Y_SR.\n"
                "- Video output keeps the source FPS, but processing may be slower than realtime.\n"
                "- Webcam on Hugging Face CPU Spaces can be slow; use a small webcam_max_width."
            )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue()
    app.launch()
