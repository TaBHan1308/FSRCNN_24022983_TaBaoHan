import os
import cv2
import torch
import numpy as np

from model import FSRCNN
import imgproc


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

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    return model


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")

    return 10 * np.log10((255.0 ** 2) / mse)


def y_to_tensor(y_channel, device):
    y = y_channel.astype(np.float32) / 255.0
    tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


def tensor_to_y(tensor):
    y = tensor.squeeze(0).squeeze(0).detach().cpu().clamp(0, 1).numpy()
    y = (y * 255.0).round().astype(np.uint8)
    return y


def add_label(image, text):
    image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 45), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)

    cv2.putText(
        image,
        text,
        (15, 32),
        font,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    return image


def process_one_image(model, lr_bgr, hr_bgr, device):
    hr_h, hr_w = hr_bgr.shape[:2]

    # Bicubic baseline
    bicubic_bgr = cv2.resize(
        lr_bgr,
        (hr_w, hr_h),
        interpolation=cv2.INTER_CUBIC
    )

    # Lấy kênh Y giống pipeline train
    lr_y = imgproc.bgr2ycbcr(lr_bgr.astype(np.float32) / 255.0, use_y_channel=True)
    lr_y = (lr_y * 255.0).round().astype(np.uint8)

    # Lấy Cr/Cb từ OpenCV để ghép lại ảnh màu
    lr_ycrcb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2YCrCb)
    lr_cr = lr_ycrcb[:, :, 1]
    lr_cb = lr_ycrcb[:, :, 2]

    # Chạy model trên kênh Y
    lr_y_tensor = y_to_tensor(lr_y, device)
    with torch.no_grad():
        sr_y_tensor = model(lr_y_tensor)

    sr_y = tensor_to_y(sr_y_tensor)

    if sr_y.shape[:2] != (hr_h, hr_w):
        sr_y = cv2.resize(sr_y, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)

    # Resize Cr/Cb lên HR size
    sr_cr = cv2.resize(lr_cr, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
    sr_cb = cv2.resize(lr_cb, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)

    # Ghép thành ảnh màu
    sr_ycrcb = cv2.merge([sr_y, sr_cr, sr_cb])
    sr_bgr = cv2.cvtColor(sr_ycrcb, cv2.COLOR_YCrCb2BGR)

    # Tính PSNR RGB
    psnr_bicubic_rgb = calculate_psnr(bicubic_bgr, hr_bgr)
    psnr_fsrcnn_rgb = calculate_psnr(sr_bgr, hr_bgr)

    # Tính PSNR Y
    hr_y = imgproc.bgr2ycbcr(hr_bgr.astype(np.float32) / 255.0, use_y_channel=True)
    hr_y = (hr_y * 255.0).round().astype(np.uint8)

    bicubic_y = imgproc.bgr2ycbcr(bicubic_bgr.astype(np.float32) / 255.0, use_y_channel=True)
    bicubic_y = (bicubic_y * 255.0).round().astype(np.uint8)

    psnr_bicubic_y = calculate_psnr(bicubic_y, hr_y)

    # PSNR-Y của FSRCNN phải tính trực tiếp trên kênh Y do model sinh ra
    psnr_fsrcnn_y = calculate_psnr(sr_y, hr_y)

    return {
        "bicubic_bgr": bicubic_bgr,
        "sr_bgr": sr_bgr,
        "psnr_bicubic_rgb": psnr_bicubic_rgb,
        "psnr_fsrcnn_rgb": psnr_fsrcnn_rgb,
        "psnr_bicubic_y": psnr_bicubic_y,
        "psnr_fsrcnn_y": psnr_fsrcnn_y,
    }


def main():
    scale = 3
    checkpoint_path = "results/fsrcnn_y_x3_pretrained_mixed/best_epoch300_psnr3299.pth.tar"

    lr_dir = "data/Set5/LRbicx3"
    hr_dir = "data/Set5/GTmod12"

    output_dir = "outputs/set5_y_x3_final_best"
    sr_dir = os.path.join(output_dir, "sr")
    fsrcnn_x3_dir = os.path.join(output_dir, "fsrcnn_x3")
    bicubic_dir = os.path.join(output_dir, "bicubic")
    compare_dir = os.path.join(output_dir, "compare")

    os.makedirs(sr_dir, exist_ok=True)
    os.makedirs(fsrcnn_x3_dir, exist_ok=True)
    os.makedirs(bicubic_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = FSRCNN(upscale_factor=scale).to(device)
    model = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    image_names = sorted([
        f for f in os.listdir(lr_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ])

    print(f"Found {len(image_names)} images.")
    print(f"Checkpoint: {checkpoint_path}")
    print("")

    total_bicubic_rgb = 0.0
    total_fsrcnn_rgb = 0.0
    total_bicubic_y = 0.0
    total_fsrcnn_y = 0.0
    valid_count = 0

    for image_name in image_names:
        lr_path = os.path.join(lr_dir, image_name)
        hr_path = os.path.join(hr_dir, image_name)

        lr_bgr = cv2.imread(lr_path, cv2.IMREAD_COLOR)
        hr_bgr = cv2.imread(hr_path, cv2.IMREAD_COLOR)

        if lr_bgr is None:
            print(f"Cannot read LR image: {lr_path}")
            continue
        if hr_bgr is None:
            print(f"Cannot read HR image: {hr_path}")
            continue

        result = process_one_image(model, lr_bgr, hr_bgr, device)

        bicubic_bgr = result["bicubic_bgr"]
        sr_bgr = result["sr_bgr"]

        psnr_bicubic_rgb = result["psnr_bicubic_rgb"]
        psnr_fsrcnn_rgb = result["psnr_fsrcnn_rgb"]
        psnr_bicubic_y = result["psnr_bicubic_y"]
        psnr_fsrcnn_y = result["psnr_fsrcnn_y"]

        # Lưu SR riêng
        sr_save_path = os.path.join(sr_dir, image_name)
        cv2.imwrite(sr_save_path, sr_bgr)
        # Lưu FSRCNN x3 riêng
        fsrcnn_x3_save_path = os.path.join(fsrcnn_x3_dir, image_name)
        cv2.imwrite(fsrcnn_x3_save_path, sr_bgr)
        # Lưu Bicubic riêng
        bicubic_save_path = os.path.join(bicubic_dir, image_name)
        cv2.imwrite(bicubic_save_path, bicubic_bgr)

        # Tạo ảnh so sánh
        bicubic_label = add_label(
            bicubic_bgr,
            f"Bicubic x3 | PSNR-Y {psnr_bicubic_y:.2f}"
        )
        fsrcnn_label = add_label(
            sr_bgr,
            f"FSRCNN-Y x3 | PSNR-Y {psnr_fsrcnn_y:.2f}"
        )
        hr_label = add_label(hr_bgr, "HR Ground Truth")

        comparison = np.hstack([bicubic_label, fsrcnn_label, hr_label])

        compare_save_path = os.path.join(compare_dir, image_name)
        cv2.imwrite(compare_save_path, comparison)

        print(f"[DONE] {image_name}")
        print(f"       Bicubic PSNR RGB : {psnr_bicubic_rgb:.2f} dB")
        print(f"       FSRCNN PSNR RGB  : {psnr_fsrcnn_rgb:.2f} dB")
        print(f"       Bicubic PSNR Y   : {psnr_bicubic_y:.2f} dB")
        print(f"       FSRCNN PSNR Y    : {psnr_fsrcnn_y:.2f} dB")
        print(f"       Saved SR      -> {sr_save_path}")
        print(f"       Saved FSRCNN  -> {fsrcnn_x3_save_path}")
        print(f"       Saved Bicubic -> {bicubic_save_path}")
        print(f"       Saved compare -> {compare_save_path}")
        print("")

        total_bicubic_rgb += psnr_bicubic_rgb
        total_fsrcnn_rgb += psnr_fsrcnn_rgb
        total_bicubic_y += psnr_bicubic_y
        total_fsrcnn_y += psnr_fsrcnn_y
        valid_count += 1

    if valid_count > 0:
        avg_bicubic_rgb = total_bicubic_rgb / valid_count
        avg_fsrcnn_rgb = total_fsrcnn_rgb / valid_count
        avg_bicubic_y = total_bicubic_y / valid_count
        avg_fsrcnn_y = total_fsrcnn_y / valid_count

        print("========== AVERAGE RESULTS ==========")
        print(f"Average Bicubic PSNR RGB : {avg_bicubic_rgb:.2f} dB")
        print(f"Average FSRCNN PSNR RGB  : {avg_fsrcnn_rgb:.2f} dB")
        print(f"Average Bicubic PSNR Y   : {avg_bicubic_y:.2f} dB")
        print(f"Average FSRCNN PSNR Y    : {avg_fsrcnn_y:.2f} dB")
        print("=====================================")

    print("Inference finished.")


if __name__ == "__main__":
    main()
