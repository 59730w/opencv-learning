import sys

import torch
import ultralytics


print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print("VRAM:", round(memory, 2), "GB")