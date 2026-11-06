from pathlib import Path
import torch
import rasterio

print("PyTorch:", torch.__version__)
print("Input exists:", Path("data/sentinel2_selected_bands.tif").exists())
print("CPU inference device:", "cpu")
