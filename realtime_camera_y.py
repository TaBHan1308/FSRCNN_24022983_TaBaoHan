import os
import threading
import time
from datetime import datetime
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "tkinter is not installed in this Python environment. On Windows, run the "
        "Python installer again, choose Modify, and enable 'tcl/tk and IDLE'. "
        "Then run: python realtime_camera_y.py"
    ) from exc

import cv2
import numpy as np
import torch

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    raise ImportError("Pillow is required for the Tkinter app. Install it with: pip install pillow") from exc

import imgproc
from model import FSRCNN


PRIMARY_CHECKPOINT = "results/fsrcnn_y_x3_pretrained_mixed/best_epoch300_psnr3299.pth.tar"
FALLBACK_CHECKPOINT = "results/fsrcnn_y_x3_pretrained_mixed/best.pth.tar"
VIDEO_OUTPUT_DIR = "outputs/video_y_x3_app"
CAMERA_OUTPUT_DIR = "outputs/camera_realtime_fsrcnn_y_x3"

SCALE = 3
MAX_INPUT_WIDTH = 426
CAMERA_INPUT_WIDTH = 320
PANEL_WIDTH = 640
PANEL_HEIGHT = 360


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


def resize_with_aspect_ratio(image, target_width=None, target_height=None):
    height, width = image.shape[:2]
    if target_width is None and target_height is None:
        return image

    if target_width is None:
        scale = target_height / height
        target_width = int(width * scale)
    elif target_height is None:
        scale = target_width / width
        target_height = int(height * scale)
    else:
        scale = min(target_width / width, target_height / height)
        target_width = int(width * scale)
        target_height = int(height * scale)

    target_width = max(target_width, 1)
    target_height = max(target_height, 1)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def crop_to_multiple(image, multiple):
    height, width = image.shape[:2]
    cropped_height = height - (height % multiple)
    cropped_width = width - (width % multiple)
    if cropped_height <= 0 or cropped_width <= 0:
        return image

    top = (height - cropped_height) // 2
    left = (width - cropped_width) // 2
    return image[top:top + cropped_height, left:left + cropped_width]


def make_panel(image, panel_width, panel_height, label):
    canvas = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    resized = resize_with_aspect_ratio(image, target_width=panel_width, target_height=panel_height)
    resized_height, resized_width = resized.shape[:2]

    top = (panel_height - resized_height) // 2
    left = (panel_width - resized_width) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, 44), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)
    cv2.putText(
        canvas,
        label,
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def get_y_channel_bgr(image_bgr):
    y = imgproc.bgr2ycbcr(image_bgr.astype(np.float32) / 255.0, use_y_channel=True)
    return (y * 255.0).round().clip(0, 255).astype(np.uint8)


def y_to_tensor(y_channel, device):
    y = y_channel.astype(np.float32) / 255.0
    tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


def tensor_to_y(tensor):
    y = tensor.squeeze(0).squeeze(0).detach().cpu().clamp(0, 1).numpy()
    return (y * 255.0).round().clip(0, 255).astype(np.uint8)


def prepare_lr_frame(frame_bgr, max_input_width):
    frame = frame_bgr
    if frame.shape[1] > max_input_width:
        frame = resize_with_aspect_ratio(frame, target_width=max_input_width)
    return crop_to_multiple(frame, SCALE)


def fsrcnn_y_upscale_frame(model, frame_bgr, device):
    lr_h, lr_w = frame_bgr.shape[:2]
    sr_h = lr_h * SCALE
    sr_w = lr_w * SCALE

    lr_y = get_y_channel_bgr(frame_bgr)
    lr_ycbcr = imgproc.bgr2ycbcr(frame_bgr.astype(np.float32) / 255.0, use_y_channel=False)
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


def make_compare_frame(input_bgr, output_bgr, fps=0.0):
    input_panel = make_panel(input_bgr, PANEL_WIDTH, PANEL_HEIGHT, "Input")
    output_panel = make_panel(output_bgr, PANEL_WIDTH, PANEL_HEIGHT, f"FSRCNN-Y x3 Output | FPS {fps:.1f}")
    return np.hstack([input_panel, output_panel])


def bgr_to_tk_image(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)
    return ImageTk.PhotoImage(image=image)


class FSRCNNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FSRCNN-Y x3 Video Super-Resolution App")
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self.checkpoint_path = resolve_checkpoint()
        self.model = FSRCNN(upscale_factor=SCALE).to(self.device)
        self.model = load_checkpoint(self.model, self.checkpoint_path, self.device)
        self.model.eval()

        self.video_path = None
        self.camera_cap = None
        self.camera_running = False
        self.processing_video = False
        self.recording = False
        self.video_writer = None
        self.last_compare_frame = None
        self.last_time = time.perf_counter()
        self.fps = 0.0
        self.display_photo = None
        self.closed = False

        self.device_var = tk.StringVar(value=f"Device: {self.device}")
        gpu_text = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        self.gpu_var = tk.StringVar(value=f"GPU: {gpu_text}")
        self.checkpoint_var = tk.StringVar(value=f"Checkpoint: {self.checkpoint_path}")
        self.fps_var = tk.StringVar(value="FPS: 0.0")
        self.status_var = tk.StringVar(value="Status: Ready")
        self.video_var = tk.StringVar(value="Selected video: None")
        self.progress_var = tk.StringVar(value="Progress: -")

        self._build_ui()
        self._show_blank_panels()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(main, text="Mode / Controls", padding=8)
        controls.grid(row=0, column=0, sticky="ew")

        ttk.Button(controls, text="Select Video", command=self.select_video).grid(row=0, column=0, padx=4, pady=4)
        ttk.Button(controls, text="Process Video", command=self.process_video_file).grid(row=0, column=1, padx=4, pady=4)
        ttk.Button(controls, text="Start Camera", command=self.start_camera).grid(row=0, column=2, padx=4, pady=4)
        ttk.Button(controls, text="Stop Camera", command=self.stop_camera).grid(row=0, column=3, padx=4, pady=4)
        ttk.Button(controls, text="Save Snapshot", command=self.save_snapshot).grid(row=0, column=4, padx=4, pady=4)
        ttk.Button(controls, text="Start/Stop Record", command=self.toggle_recording).grid(row=0, column=5, padx=4, pady=4)
        ttk.Button(controls, text="Exit", command=self.exit_app).grid(row=0, column=6, padx=4, pady=4)

        info = ttk.LabelFrame(main, text="Info", padding=8)
        info.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        ttk.Label(info, textvariable=self.device_var).grid(row=0, column=0, sticky="w", padx=4)
        ttk.Label(info, textvariable=self.gpu_var).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(info, textvariable=self.fps_var).grid(row=0, column=2, sticky="w", padx=4)
        ttk.Label(info, textvariable=self.status_var).grid(row=1, column=0, sticky="w", padx=4)
        ttk.Label(info, textvariable=self.progress_var).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(info, textvariable=self.video_var).grid(row=2, column=0, columnspan=3, sticky="w", padx=4)
        ttk.Label(info, textvariable=self.checkpoint_var).grid(row=3, column=0, columnspan=3, sticky="w", padx=4)

        display = ttk.LabelFrame(main, text="Input | FSRCNN-Y x3 Output", padding=8)
        display.grid(row=2, column=0, sticky="nsew")
        main.rowconfigure(2, weight=1)

        self.image_label = ttk.Label(display)
        self.image_label.grid(row=0, column=0)

    def _show_blank_panels(self):
        blank_input = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
        blank_output = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
        compare = make_compare_frame(blank_input, blank_output, fps=0.0)
        self._update_display(compare)

    def _update_display(self, compare_bgr):
        self.last_compare_frame = compare_bgr
        self.display_photo = bgr_to_tk_image(compare_bgr)
        self.image_label.configure(image=self.display_photo)

    def _update_fps(self):
        now = time.perf_counter()
        instant_fps = 1.0 / max(now - self.last_time, 1e-8)
        self.fps = instant_fps if self.fps == 0.0 else (0.9 * self.fps + 0.1 * instant_fps)
        self.last_time = now
        self.fps_var.set(f"FPS: {self.fps:.1f}")
        return self.fps

    def select_video(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")],
        )
        if path:
            self.video_path = path
            self.video_var.set(f"Selected video: {path}")
            self.status_var.set("Status: Ready")

    def process_video_file(self):
        if self.processing_video:
            messagebox.showinfo("FSRCNN", "Video processing is already running.")
            return
        if not self.video_path:
            messagebox.showwarning("FSRCNN", "Please select a video first.")
            return

        self.stop_camera()
        self.processing_video = True
        self.status_var.set("Status: Processing video")
        thread = threading.Thread(target=self._process_video_worker, daemon=True)
        thread.start()

    def _process_video_worker(self):
        process_video_file(self)
        self.processing_video = False

    def start_camera(self):
        if self.camera_running:
            return
        if self.processing_video:
            messagebox.showinfo("FSRCNN", "Please wait until video processing finishes.")
            return

        self.camera_cap = cv2.VideoCapture(0)
        if not self.camera_cap.isOpened():
            self.camera_cap = None
            messagebox.showerror("FSRCNN", "Cannot open webcam with cv2.VideoCapture(0).")
            return

        self.camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_INPUT_WIDTH)
        self.camera_running = True
        self.last_time = time.perf_counter()
        self.status_var.set("Status: Camera running")
        self.update_camera_frame()

    def stop_camera(self):
        if self.camera_cap is not None:
            self.camera_cap.release()
            self.camera_cap = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.camera_running = False
        self.recording = False
        self.status_var.set("Status: Stopped")

    def update_camera_frame(self):
        if not self.camera_running or self.camera_cap is None:
            return

        ret, frame = self.camera_cap.read()
        if not ret:
            self.stop_camera()
            messagebox.showerror("FSRCNN", "Cannot read frame from webcam.")
            return

        input_frame = prepare_lr_frame(frame, CAMERA_INPUT_WIDTH)
        output_frame = fsrcnn_y_upscale_frame(self.model, input_frame, self.device)
        fps = self._update_fps()
        compare = make_compare_frame(input_frame, output_frame, fps)
        self._update_display(compare)

        if self.recording:
            if self.video_writer is None:
                os.makedirs(CAMERA_OUTPUT_DIR, exist_ok=True)
                output_path = os.path.join(CAMERA_OUTPUT_DIR, "camera_compare.mp4")
                self.video_writer = self._create_video_writer(output_path, compare, 30.0)
            self.video_writer.write(compare)

        self.root.after(1, self.update_camera_frame)

    def save_snapshot(self):
        if self.last_compare_frame is None:
            messagebox.showwarning("FSRCNN", "No frame to save yet.")
            return

        os.makedirs(CAMERA_OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = os.path.join(CAMERA_OUTPUT_DIR, f"snapshot_{timestamp}.png")
        cv2.imwrite(save_path, self.last_compare_frame)
        self.status_var.set(f"Status: Snapshot saved: {save_path}")

    def toggle_recording(self):
        if not self.camera_running:
            messagebox.showinfo("FSRCNN", "Start camera before recording.")
            return

        self.recording = not self.recording
        if self.recording:
            self.status_var.set("Status: Recording")
        else:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.status_var.set("Status: Camera running")

    def _create_video_writer(self, path, frame, fps):
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(path, fourcc, fps, (width, height))

    def exit_app(self):
        self.closed = True
        self.stop_camera()
        self.processing_video = False
        self.root.destroy()


def process_video_file(app):
    cap = cv2.VideoCapture(app.video_path)
    if not cap.isOpened():
        app.root.after(0, lambda: messagebox.showerror("FSRCNN", "Cannot open selected video."))
        return

    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    sr_path = os.path.join(VIDEO_OUTPUT_DIR, "output_fsrcnn_y_x3.mp4")
    compare_path = os.path.join(VIDEO_OUTPUT_DIR, "compare_input_fsrcnn.mp4")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0 or np.isnan(input_fps):
        input_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sr_writer = None
    compare_writer = None
    frame_index = 0
    last_time = time.perf_counter()

    try:
        while app.processing_video:
            ret, frame = cap.read()
            if not ret:
                break

            input_frame = prepare_lr_frame(frame, MAX_INPUT_WIDTH)
            output_frame = fsrcnn_y_upscale_frame(app.model, input_frame, app.device)
            compare = make_compare_frame(input_frame, output_frame, app.fps)

            if sr_writer is None:
                sr_h, sr_w = output_frame.shape[:2]
                sr_writer = cv2.VideoWriter(
                    sr_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    input_fps,
                    (sr_w, sr_h),
                )
                cmp_h, cmp_w = compare.shape[:2]
                compare_writer = cv2.VideoWriter(
                    compare_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    input_fps,
                    (cmp_w, cmp_h),
                )

            sr_writer.write(output_frame)
            compare_writer.write(compare)
            frame_index += 1

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_time, 1e-8)
            last_time = now
            app.fps = instant_fps if app.fps == 0.0 else (0.9 * app.fps + 0.1 * instant_fps)

            progress = f"Progress: {frame_index}/{total_frames if total_frames > 0 else '?'}"
            if not app.closed:
                app.root.after(0, lambda c=compare: app._update_display(c))
                app.root.after(0, lambda f=app.fps: app.fps_var.set(f"FPS: {f:.1f}"))
                app.root.after(0, lambda p=progress: app.progress_var.set(p))

    finally:
        cap.release()
        if sr_writer is not None:
            sr_writer.release()
        if compare_writer is not None:
            compare_writer.release()

    def done():
        app.progress_var.set(f"Progress: {frame_index}/{total_frames if total_frames > 0 else '?'}")
        app.status_var.set(f"Status: Video saved to {VIDEO_OUTPUT_DIR}")

    if not app.closed:
        app.root.after(0, done)


def start_camera(app):
    app.start_camera()


def stop_camera(app):
    app.stop_camera()


def update_camera_frame(app):
    app.update_camera_frame()


def save_snapshot(app):
    app.save_snapshot()


def toggle_recording(app):
    app.toggle_recording()


def main():
    root = tk.Tk()
    FSRCNNApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
