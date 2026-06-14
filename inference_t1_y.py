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


def get_y_channel_bgr(image_bgr):
    y = imgproc.bgr2ycbcr(image_bgr.astype(np.float32) / 255.0, use_y_channel=True)
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


def main():
    scale = 3

    hr_image_path = "data/T91/FSRCNN/train/t1.png"
    checkpoint_path = "results/fsrcnn_y_x3_pretrained_mixed/best_epoch300_psnr3299.pth.tar"

    output_dir = "outputs/t1_y_x3_final_best"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = FSRCNN(upscale_factor=scale).to(device)
    model = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    hr_bgr = cv2.imread(hr_image_path, cv2.IMREAD_COLOR)
    if hr_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {hr_image_path}")

    h, w = hr_bgr.shape[:2]
    h_crop = h - h % scale
    w_crop = w - w % scale
    hr_bgr = hr_bgr[:h_crop, :w_crop]

    # Tạo LR từ HR bằng Bicubic downscale
    lr_bgr = cv2.resize(
        hr_bgr,
        (w_crop // scale, h_crop // scale),
        interpolation=cv2.INTER_CUBIC
    )

    # Bicubic baseline
    bicubic_bgr = cv2.resize(
        lr_bgr,
        (w_crop, h_crop),
        interpolation=cv2.INTER_CUBIC
    )

    # Lấy kênh Y giống pipeline train
    lr_y = get_y_channel_bgr(lr_bgr)

    # Lấy Cr/Cb để ghép ảnh màu
    lr_ycrcb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2YCrCb)
    lr_cr = lr_ycrcb[:, :, 1]
    lr_cb = lr_ycrcb[:, :, 2]

    # FSRCNN xử lý kênh Y
    lr_y_tensor = y_to_tensor(lr_y, device)

    with torch.no_grad():
        sr_y_tensor = model(lr_y_tensor)

    sr_y = tensor_to_y(sr_y_tensor)

    if sr_y.shape[:2] != (h_crop, w_crop):
        sr_y = cv2.resize(sr_y, (w_crop, h_crop), interpolation=cv2.INTER_CUBIC)

    # Resize Cr/Cb bằng Bicubic
    sr_cr = cv2.resize(lr_cr, (w_crop, h_crop), interpolation=cv2.INTER_CUBIC)
    sr_cb = cv2.resize(lr_cb, (w_crop, h_crop), interpolation=cv2.INTER_CUBIC)

    # Ghép ảnh màu
    sr_ycrcb = cv2.merge([sr_y, sr_cr, sr_cb])
    sr_bgr = cv2.cvtColor(sr_ycrcb, cv2.COLOR_YCrCb2BGR)

    # Tính PSNR RGB
    psnr_bicubic_rgb = calculate_psnr(bicubic_bgr, hr_bgr)
    psnr_fsrcnn_rgb = calculate_psnr(sr_bgr, hr_bgr)

    # Tính PSNR Y đúng cách
    hr_y = get_y_channel_bgr(hr_bgr)
    bicubic_y = get_y_channel_bgr(bicubic_bgr)

    psnr_bicubic_y = calculate_psnr(bicubic_y, hr_y)
    psnr_fsrcnn_y = calculate_psnr(sr_y, hr_y)

    # Lưu ảnh riêng
    cv2.imwrite(os.path.join(output_dir, "t1_hr.png"), hr_bgr)
    cv2.imwrite(os.path.join(output_dir, "t1_lr.png"), lr_bgr)
    cv2.imwrite(os.path.join(output_dir, "t1_bicubic.png"), bicubic_bgr)
    cv2.imwrite(os.path.join(output_dir, "t1_fsrcnn_y_x3.png"), sr_bgr)

    # Tạo ảnh so sánh
    bicubic_label = add_label(
        bicubic_bgr,
        f"Bicubic x3 | PSNR-Y {psnr_bicubic_y:.2f} dB"
    )

    fsrcnn_label = add_label(
        sr_bgr,
        f"FSRCNN-Y x3 | PSNR-Y {psnr_fsrcnn_y:.2f} dB"
    )

    hr_label = add_label(hr_bgr, "HR Ground Truth")

    comparison = np.hstack([bicubic_label, fsrcnn_label, hr_label])

    compare_path = os.path.join(
        output_dir,
        "t1_comparison_bicubic_fsrcnn_hr.png"
    )

    cv2.imwrite(compare_path, comparison)

    print("Done.")
    print(f"Input HR image : {hr_image_path}")
    print(f"Checkpoint     : {checkpoint_path}")
    print(f"Output folder  : {output_dir}")
    print(f"HR shape       : {hr_bgr.shape}")
    print(f"LR shape       : {lr_bgr.shape}")
    print(f"Bicubic PSNR RGB : {psnr_bicubic_rgb:.2f} dB")
    print(f"FSRCNN PSNR RGB  : {psnr_fsrcnn_rgb:.2f} dB")
    print(f"Bicubic PSNR Y   : {psnr_bicubic_y:.2f} dB")
    print(f"FSRCNN PSNR Y    : {psnr_fsrcnn_y:.2f} dB")
    print("")
    print("Saved files:")
    print(f"- {output_dir}/t1_hr.png")
    print(f"- {output_dir}/t1_lr.png")
    print(f"- {output_dir}/t1_bicubic.png")
    print(f"- {output_dir}/t1_fsrcnn_y_x3.png")
    print(f"- {compare_path}")


if __name__ == "__main__":
    main()