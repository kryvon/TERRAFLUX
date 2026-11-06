
import os
import glob
import csv
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

PROB_DIR = "outputs/validation5/probabilities"
LABEL_DIR = "data/validation5/labels"
OUTPUT_DIR = "outputs/validation5/error_analysis"
THRESHOLD = 0.30

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(PROB_DIR, "*_floodprob.tif")))

if len(files) != 5:
    raise RuntimeError(f"Expected 5 probability maps, found {len(files)}")

# 0 = True Negative, 1 = True Positive
# 2 = False Positive, 3 = False Negative
cmap = ListedColormap(["#eeeeee", "#16803c", "#e53935", "#f9a825"])

results = []

for prob_path in files:
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
    truth = label == 1
    prediction = probability >= THRESHOLD

    error_map = np.full(label.shape, 255, dtype=np.uint8)
    error_map[valid & ~truth & ~prediction] = 0
    error_map[valid & truth & prediction] = 1
    error_map[valid & ~truth & prediction] = 2
    error_map[valid & truth & ~prediction] = 3

    tp = int(np.sum(valid & truth & prediction))
    fp = int(np.sum(valid & ~truth & prediction))
    fn = int(np.sum(valid & truth & ~prediction))
    tn = int(np.sum(valid & ~truth & ~prediction))

    results.append([name, tp, fp, fn, tn])

    display = np.ma.masked_equal(error_map, 255)

    plt.figure(figsize=(7, 6))
    plt.imshow(display, cmap=cmap, vmin=0, vmax=3)
    plt.title(f"{name} | Error map | Threshold {THRESHOLD}")
    plt.axis("off")

    plt.savefig(
        os.path.join(OUTPUT_DIR, name + "_error_map.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

csv_path = os.path.join(OUTPUT_DIR, "error_counts.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "TP", "FP", "FN", "TN"])
    writer.writerows(results)

print("ERROR ANALYSIS COMPLETE")
for row in results:
    print(row)

print("\nMaps saved in:", os.path.abspath(OUTPUT_DIR))
print("Counts saved in:", os.path.abspath(csv_path))
print("\nMap legend:")
print("Grey = True Negative")
print("Green = True Positive")
print("Red = False Positive")
print("Yellow = False Negative")
print("White/transparent = NoData")

