import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def run_script(script_name, *args):
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script_name), *map(str, args)],
        check=True,
        cwd=PROJECT_ROOT,
    )

# Prepare dataset
run_script(
    "augment_dataset.py",
    "--images_dir", PROJECT_ROOT / "data/T91/original",
    "--output_dir", PROJECT_ROOT / "data/T91/FSRCNN/original",
    "--num_workers", "10",
)
run_script(
    "prepare_dataset.py",
    "--images_dir", PROJECT_ROOT / "data/T91/FSRCNN/original",
    "--output_dir", PROJECT_ROOT / "data/T91/FSRCNN/train",
    "--image_size", "32",
    "--step", "16",
    "--num_workers", "10",
)

# Split train and valid
run_script(
    "split_train_valid_dataset.py",
    "--train_images_dir", PROJECT_ROOT / "data/T91/FSRCNN/train",
    "--valid_images_dir", PROJECT_ROOT / "data/T91/FSRCNN/valid",
    "--valid_samples_ratio", "0.1",
)
