import torch
from model import FSRCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scale = 3
model = FSRCNN(upscale_factor=scale).to(device)

x = torch.randn(1, 1, 64, 64).to(device)
y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))