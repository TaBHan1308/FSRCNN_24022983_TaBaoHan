import argparse
import inspect
import os
import time

import cv2
import numpy as np
import torch

import imgproc
from model import FSRCNN


SCALE = 3
PRIMARY_CHECKPOINT = "results/fsrcnn_y_x3_pretrained_mixed/best_epoch300_psnr3299.pth.tar"
FALLBACK_CHECKPOINT = "results/fsrcnn_y_x3_pretrained_mixed/best.pth.tar"
OUTPUT_DIR = "outputs/video_benchmark"


def resolve_checkpoint():
    if os.path.exists(PRIMARY_CHECKPOINT):
        return PRIMARY_CHECKPOINT
    if os.path.exists(FALLBACK_CHECKPOINT):
        return FALLBACK_CHECKPOINT
    raise FileNotFoundError(
        "Cannot find checkpoint. Tried:\n"
        f"  {PRIMARY_CHECKPOINT}\n"
        f"  {FALLBACK_CHECKPOINT}"
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


def resize_keep_aspect_bgr(image, max_width):
    if max_width <= 0 or image.shape[1] <= max_width:
        return image
    scale = max_width / image.shape[1]
    new_width = int(round(image.shape[1] * scale))
    new_height = int(round(image.shape[0] * scale))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def crop_to_scale(image, scale):
    height, width = image.shape[:2]
    crop_height = height - (height % scale)
    crop_width = width - (width % scale)
    if crop_height <= 0 or crop_width <= 0:
        raise ValueError(f"Frame is too small after crop: {image.shape}")
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return image[top:top + crop_height, left:left + crop_width]


def bgr_to_y_uint8(image_bgr):
    y = imgproc.bgr2ycbcr(image_bgr.astype(np.float32) / 255.0, use_y_channel=True)
    return (y * 255.0).round().clip(0, 255).astype(np.uint8)


def y_to_tensor(y_channel, device):
    y = y_channel.astype(np.float32) / 255.0
    tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


def tensor_to_y(tensor):
    y = tensor.squeeze(0).squeeze(0).detach().cpu().clamp(0, 1).numpy()
    return (y * 255.0).round().clip(0, 255).astype(np.uint8)


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)


def calculate_ssim_optional(img1, img2, ssim_func):
    if ssim_func is None:
        return None
    return float(ssim_func(img1, img2, data_range=255))


def get_ssim_function():
    try:
        from skimage.metrics import structural_similarity

        return structural_similarity
    except ImportError:
        return None


def fsrcnn_y_upscale_frame(model, lr_bgr, device):
    lr_h, lr_w = lr_bgr.shape[:2]
    sr_h = lr_h * SCALE
    sr_w = lr_w * SCALE

    lr_y = bgr_to_y_uint8(lr_bgr)
    lr_ycbcr = imgproc.bgr2ycbcr(lr_bgr.astype(np.float32) / 255.0, use_y_channel=False)
    _, lr_cb, lr_cr = cv2.split(lr_ycbcr)

    with torch.inference_mode():
        sr_y_tensor = model(y_to_tensor(lr_y, device))

    sr_y = tensor_to_y(sr_y_tensor)
    if sr_y.shape[:2] != (sr_h, sr_w):
        sr_y = cv2.resize(sr_y, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)

    sr_cb = cv2.resize(lr_cb, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)
    sr_cr = cv2.resize(lr_cr, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)
    sr_ycbcr = cv2.merge([sr_y.astype(np.float32) / 255.0, sr_cb, sr_cr])
    sr_bgr = imgproc.ycbcr2bgr(sr_ycbcr)
    sr_bgr = (sr_bgr * 255.0).round().clip(0, 255).astype(np.uint8)
    return sr_bgr, sr_y


def add_label(image, text):
    image = image.copy()
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 48), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
    cv2.putText(
        image,
        text,
        (14, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def format_optional_metric(value):
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser(description="Benchmark FSRCNN-Y x3 on a HR video with synthetic LR.")
    parser.add_argument("--video", required=True, help="Path to HR input video.")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to process. Use 0 for full video.")
    parser.add_argument("--max_hr_width", type=int, default=0, help="Resize HR test frame to this width. 0 keeps original.")
    parser.add_argument("--save_frame_index", type=int, default=10, help="Frame index to save comparison image.")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Cannot find input video: {args.video}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_path = resolve_checkpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model = build_model(SCALE).to(device)
    model = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {args.video}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0 or np.isnan(source_fps):
        source_fps = 0.0
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = total_frames if args.frames == 0 else args.frames
    if total_frames > 0:
        frame_limit = min(frame_limit, total_frames)

    ssim_func = get_ssim_function()
    bicubic_psnr_y_values = []
    fsrcnn_psnr_y_values = []
    bicubic_ssim_y_values = []
    fsrcnn_ssim_y_values = []
    frame_times = []
    processed = 0
    hr_shape = None
    lr_shape = None
    sr_shape = None
    comparison_saved = False

    while processed < frame_limit:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        hr_bgr = resize_keep_aspect_bgr(frame_bgr, args.max_hr_width)
        hr_bgr = crop_to_scale(hr_bgr, SCALE)
        hr_h, hr_w = hr_bgr.shape[:2]

        lr_bgr = cv2.resize(hr_bgr, (hr_w // SCALE, hr_h // SCALE), interpolation=cv2.INTER_CUBIC)
        bicubic_bgr = cv2.resize(lr_bgr, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        sr_bgr, sr_y = fsrcnn_y_upscale_frame(model, lr_bgr, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        frame_times.append(elapsed)

        if sr_bgr.shape[:2] != (hr_h, hr_w):
            sr_bgr = cv2.resize(sr_bgr, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
        if sr_y.shape[:2] != (hr_h, hr_w):
            sr_y = cv2.resize(sr_y, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)

        hr_y = bgr_to_y_uint8(hr_bgr)
        bicubic_y = bgr_to_y_uint8(bicubic_bgr)

        bicubic_psnr_y_values.append(calculate_psnr(bicubic_y, hr_y))
        fsrcnn_psnr_y_values.append(calculate_psnr(sr_y, hr_y))

        bicubic_ssim = calculate_ssim_optional(bicubic_y, hr_y, ssim_func)
        fsrcnn_ssim = calculate_ssim_optional(sr_y, hr_y, ssim_func)
        if bicubic_ssim is not None and fsrcnn_ssim is not None:
            bicubic_ssim_y_values.append(bicubic_ssim)
            fsrcnn_ssim_y_values.append(fsrcnn_ssim)

        if hr_shape is None:
            hr_shape = hr_bgr.shape[:2]
            lr_shape = lr_bgr.shape[:2]
            sr_shape = sr_bgr.shape[:2]

        if processed == args.save_frame_index and not comparison_saved:
            compare = np.hstack([
                add_label(bicubic_bgr, "Bicubic x3"),
                add_label(sr_bgr, "FSRCNN-Y x3"),
                add_label(hr_bgr, "HR"),
            ])
            compare_path = os.path.join(OUTPUT_DIR, "compare_frame.png")
            cv2.imwrite(compare_path, compare)
            comparison_saved = True

        processed += 1
        print(f"Processed frame {processed}/{frame_limit}", end="\r")

    cap.release()
    print("")

    if processed == 0:
        raise RuntimeError("No frames were processed.")

    if not comparison_saved:
        print("Warning: save_frame_index was not reached, so compare_frame.png was not saved.")

    fps_values = [1.0 / t for t in frame_times if t > 0]
    avg_fps = float(np.mean(fps_values))
    min_fps = float(np.min(fps_values))
    max_fps = float(np.max(fps_values))
    avg_bicubic_psnr_y = float(np.mean(bicubic_psnr_y_values))
    avg_fsrcnn_psnr_y = float(np.mean(fsrcnn_psnr_y_values))
    avg_bicubic_ssim_y = float(np.mean(bicubic_ssim_y_values)) if bicubic_ssim_y_values else None
    avg_fsrcnn_ssim_y = float(np.mean(fsrcnn_ssim_y_values)) if fsrcnn_ssim_y_values else None

    hr_h, hr_w = hr_shape
    lr_h, lr_w = lr_shape
    sr_h, sr_w = sr_shape

    lines = [
        "FSRCNN-Y x3 Video Benchmark",
        "===========================",
        f"Video path: {args.video}",
        f"Checkpoint: {checkpoint_path}",
        f"Device: {device}",
        f"Source video FPS: {source_fps:.3f}",
        f"Source video resolution: {source_width}x{source_height}",
        f"HR test resolution: {hr_w}x{hr_h}",
        f"LR resolution: {lr_w}x{lr_h}",
        f"SR output resolution: {sr_w}x{sr_h}",
        f"Processed frames: {processed}",
        f"Average FSRCNN FPS: {avg_fps:.3f}",
        f"Lowest FSRCNN FPS: {min_fps:.3f}",
        f"Highest FSRCNN FPS: {max_fps:.3f}",
        f"Average Bicubic PSNR-Y: {avg_bicubic_psnr_y:.4f} dB",
        f"Average FSRCNN PSNR-Y: {avg_fsrcnn_psnr_y:.4f} dB",
        f"Average Bicubic SSIM-Y: {format_optional_metric(avg_bicubic_ssim_y)}",
        f"Average FSRCNN SSIM-Y: {format_optional_metric(avg_fsrcnn_ssim_y)}",
        f"SSIM backend: {'skimage.metrics.structural_similarity' if ssim_func is not None else 'N/A (install scikit-image)'}",
        f"Comparison frame: {os.path.join(OUTPUT_DIR, 'compare_frame.png') if comparison_saved else 'N/A'}",
    ]

    log_path = os.path.join(OUTPUT_DIR, "video_benchmark_results.txt")
    with open(log_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved log: {log_path}")


if __name__ == "__main__":
    main()
