from ultralytics import YOLO
import torch

# Verify that CUDA (GPU) is available
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

# Select device
device = 'cuda' if torch.cuda.is_available() else 'cpu' 