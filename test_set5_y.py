from dataset import TestImageDataset
import config

dataset = TestImageDataset(
    config.test_lr_image_dir,
    config.test_hr_image_dir,
    config.upscale_factor
)

print("Number of test images:", len(dataset))

sample = dataset[0]
lr = sample["lr"]
hr = sample["hr"]

print("LR shape:", lr.shape)
print("HR shape:", hr.shape)
print("LR min/max:", lr.min().item(), lr.max().item())
print("HR min/max:", hr.min().item(), hr.max().item())