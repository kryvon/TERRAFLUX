
from pathlib import Path
import rasterio
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent.parent
sample = root / "data" / "Sen1Floods11-master" / "sample"

files = {
    "Sentinel-1": sample / "S1" / "Spain_7370579_S1Hand.tif",
    "Sentinel-2": sample / "S2" / "Spain_7370579_S2Hand.tif",
    "Labels": sample / "Labels" / "Spain_7370579_LabelHand.tif",
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, path) in zip(axes, files.items()):
    with rasterio.open(path) as image:
        data = image.read()

    if name == "Sentinel-2" and data.shape[0] >= 3:
        rgb = np.moveaxis(data[[2, 1, 0]], 0, -1).astype(float)
        low, high = np.percentile(rgb, [2, 98])
        rgb = np.clip((rgb - low) / (high - low + 1e-8), 0, 1)
        ax.imshow(rgb)
    elif name == "Labels":
        ax.imshow(data[0], cmap="gray")
    else:
        ax.imshow(data[0], cmap="gray")

    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.show()
