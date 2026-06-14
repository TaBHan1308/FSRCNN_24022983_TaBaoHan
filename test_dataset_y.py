from dataset import TrainValidImageDataset
import config

dataset = TrainValidImageDataset(
    config.train_image_dir,
    config.image_size,
    config.upscale_factor,
    "Train"
)

sample = dataset[0]

lr = sample["lr"]
hr = sample["hr"]

print("LR shape:", lr.shape)
print("HR shape:", hr.shape)
print("LR min/max:", lr.min().item(), lr.max().item())
print("HR min/max:", hr.min().item(), hr.max().item())