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
OUTPUT_DIR = "outputs/webcam_benchmark"


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


def prepare_webcam_frame(frame_bgr, max_width):
    frame_bgr = resize_keep_aspect_bgr(frame_bgr, max_width)
    return crop_to_scale(frame_bgr, SCALE)


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
    return (sr_bgr * 255.0).round().clip(0, 255).astype(np.uint8)


def make_preview(input_bgr, sr_bgr):
    sr_h, sr_w = sr_bgr.shape[:2]
    input_preview = cv2.resize(input_bgr, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)
    return np.hstack([input_preview, sr_bgr])


def main():
    parser = argparse.ArgumentParser(description="Benchmark FSRCNN-Y x3 on live webcam frames.")
    parser.add_argument("--camera", type=int, default=0, help="Camera id.")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to benchmark.")
    parser.add_argument("--width", type=int, default=256, help="Max internal input width.")
    parser.add_argument("--show", action="store_true", help="Show input and FSRCNN output preview.")
    args = parser.parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be greater than 0 for webcam benchmark.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_path = resolve_checkpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model = build_model(SCALE).to(device)
    model = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam with camera id {args.camera}.")

    frame_times = []
    processed = 0
    input_shape = None
    sr_shape = None
    saved_images = False

    try:
        while processed < args.frames:
            ret, frame_bgr = cap.read()
            if not ret:
                raise RuntimeError("Cannot read frame from webcam.")

            lr_bgr = prepare_webcam_frame(frame_bgr, args.width)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            sr_bgr = fsrcnn_y_upscale_frame(model, lr_bgr, device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            frame_times.append(elapsed)

            if input_shape is None:
                input_shape = lr_bgr.shape[:2]
                sr_shape = sr_bgr.shape[:2]

            if not saved_images:
                cv2.imwrite(os.path.join(OUTPUT_DIR, "webcam_input.png"), lr_bgr)
                cv2.imwrite(os.path.join(OUTPUT_DIR, "webcam_fsrcnn_y_x3.png"), sr_bgr)
                saved_images = True

            if args.show:
                preview = make_preview(lr_bgr, sr_bgr)
                cv2.imshow("Webcam Input x3 Preview | FSRCNN-Y x3", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
            print(f"Processed frame {processed}/{args.frames}", end="\r")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    print("")

    if processed == 0:
        raise RuntimeError("No webcam frames were processed.")

    fps_values = [1.0 / t for t in frame_times if t > 0]
    avg_fps = float(np.mean(fps_values))
    min_fps = float(np.min(fps_values))
    max_fps = float(np.max(fps_values))

    input_h, input_w = input_shape
    sr_h, sr_w = sr_shape

    lines = [
        "FSRCNN-Y x3 Webcam Benchmark",
        "============================",
        f"Camera id: {args.camera}",
        f"Checkpoint: {checkpoint_path}",
        f"Device: {device}",
        f"Input resolution after resize/crop: {input_w}x{input_h}",
        f"SR output resolution: {sr_w}x{sr_h}",
        f"Benchmark frames: {processed}",
        f"Average FPS: {avg_fps:.3f}",
        f"Lowest FPS: {min_fps:.3f}",
        f"Highest FPS: {max_fps:.3f}",
        f"Saved input image: {os.path.join(OUTPUT_DIR, 'webcam_input.png')}",
        f"Saved FSRCNN image: {os.path.join(OUTPUT_DIR, 'webcam_fsrcnn_y_x3.png')}",
        "PSNR/SSIM: N/A for webcam because there is no independent HR ground truth.",
    ]

    log_path = os.path.join(OUTPUT_DIR, "webcam_benchmark_results.txt")
    with open(log_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved log: {log_path}")


if __name__ == "__main__":
    main()
