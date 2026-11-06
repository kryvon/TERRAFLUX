
import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt

PROB_DIR = "outputs/validation5/probabilities"
LABEL_DIR = "data/validation5/labels"
OUTPUT = "outputs/validation5/probability_comparison.png"
THRESHOLD = 0.50

files = sorted(glob.glob(os.path.join(PROB_DIR, "*_floodprob.tif")))

if len(files) != 5:
    raise RuntimeError(f"Expected 5 probability maps, found {len(files)}")

fig, axes = plt.subplots(5, 3, figsize=(14, 20))

for row, prob_path in enumerate(files):
    name = os.path.basename(prob_path).replace("_floodprob.tif", "")
    label_path = os.path.join(LABEL_DIR, name + "_LabelHand.tif")

    if not os.path.exists(label_path):
        raise FileNotFoundError(label_path)

    with rasterio.open(prob_path) as src:
        probability = src.read(1)

    with rasterio.open(label_path) as src:
        label = src.read(1)

    if probability.shape != label.shape:
        raise RuntimeError(f"Shape mismatch: {name}")

    valid = (label == 0) | (label == 1)
    prediction = probability >= THRESHOLD

    probability_display = np.ma.masked_where(~valid, probability)
    truth_display = np.ma.masked_where(~valid, label)
    prediction_display = np.ma.masked_where(~valid, prediction)

    axes[row, 0].imshow(probability_display, cmap="viridis", vmin=0, vmax=1)
    axes[row, 0].set_title(f"{name}\nFlood probability")

    axes[row, 1].imshow(truth_display, cmap="gray", vmin=0, vmax=1)
    axes[row, 1].set_title("Ground truth")

    axes[row, 2].imshow(prediction_display, cmap="gray", vmin=0, vmax=1)
    axes[row, 2].set_title(f"Prediction (threshold {THRESHOLD})")

    for ax in axes[row]:
        ax.axis("off")

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.close()

print("Visual comparison saved:", os.path.abspath(OUTPUT))
