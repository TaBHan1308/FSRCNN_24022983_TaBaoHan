import shutil
import random
from pathlib import Path


random.seed(42)

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp"]


def collect_images(folder):
    folder = Path(folder)
    images = []

    if not folder.exists():
        print(f"[WARNING] Folder not found: {folder}")
        return images

    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.rglob(f"*{ext}"))
        images.extend(folder.rglob(f"*{ext.upper()}"))

    images = sorted(list(set(images)))
    return images


def clear_folder(folder):
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def split_images(images, valid_ratio=0.1):
    images = images.copy()
    random.shuffle(images)

    valid_count = max(1, int(len(images) * valid_ratio))
    valid_images = images[:valid_count]
    train_images = images[valid_count:]

    return train_images, valid_images


def copy_images(images, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, src_path in enumerate(images):
        src_path = Path(src_path)
        dst_name = f"{prefix}_{idx:04d}{src_path.suffix.lower()}"
        dst_path = output_dir / dst_name
        shutil.copy2(src_path, dst_path)


def main():
    # =========================
    # Source dataset folders
    # =========================
    t91_train_dir = "data/T91/FSRCNN/train"
    t91_valid_dir = "data/T91/FSRCNN/valid"

    general100_dir = "data/General100"
    bsds200_dir = "data/BSDS200"

    # =========================
    # Output mixed dataset
    # =========================
    mixed_train_dir = "data/TrainMixed/FSRCNN/train"
    mixed_valid_dir = "data/TrainMixed/FSRCNN/valid"

    clear_folder(mixed_train_dir)
    clear_folder(mixed_valid_dir)

    # =========================
    # Collect images
    # =========================
    t91_train = collect_images(t91_train_dir)
    t91_valid = collect_images(t91_valid_dir)
    general100 = collect_images(general100_dir)
    bsds200 = collect_images(bsds200_dir)

    print("========== FOUND IMAGES ==========")
    print(f"T91 train   : {len(t91_train)}")
    print(f"T91 valid   : {len(t91_valid)}")
    print(f"General100  : {len(general100)}")
    print(f"BSDS200     : {len(bsds200)}")
    print("==================================")

    # =========================
    # Split General100 and BSDS200
    # T91 already has train/valid
    # =========================
    general_train, general_valid = split_images(general100, valid_ratio=0.1)
    bsds_train, bsds_valid = split_images(bsds200, valid_ratio=0.1)

    # =========================
    # Copy to mixed train
    # =========================
    copy_images(t91_train, mixed_train_dir, "t91")
    copy_images(general_train, mixed_train_dir, "general100")
    copy_images(bsds_train, mixed_train_dir, "bsds200")

    # =========================
    # Copy to mixed valid
    # =========================
    copy_images(t91_valid, mixed_valid_dir, "t91")
    copy_images(general_valid, mixed_valid_dir, "general100")
    copy_images(bsds_valid, mixed_valid_dir, "bsds200")

    final_train = collect_images(mixed_train_dir)
    final_valid = collect_images(mixed_valid_dir)

    print("")
    print("========== MIXED DATASET ==========")
    print(f"Mixed train images : {len(final_train)}")
    print(f"Mixed valid images : {len(final_valid)}")
    print(f"Train folder       : {mixed_train_dir}")
    print(f"Valid folder       : {mixed_valid_dir}")
    print("===================================")


if __name__ == "__main__":
    main()