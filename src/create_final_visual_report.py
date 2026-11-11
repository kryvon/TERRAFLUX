
import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

RAW_DIR = "data/validation5/raw_s2"
LABEL_DIR = "data/validation5/labels"
MAP_DIR = "outputs/validation5/flood_maps"
OUTPUT_DIR = "outputs/validation5/final_report"

os.makedirs(OUTPUT_DIR, exist_ok=True)

maps = sorted(glob.glob(os.path.join(MAP_DIR, "*_floodmap.tif")))

if len(maps) != 5:
    raise RuntimeError(f"Expected 5 maps, found {len(maps)}")

error_cmap = ListedColormap([
   "#555555",  # True Negative
    "#16803c",  # True Positive
    "#e53935",  # False Positive
    "#facc15"   # False Negative
])

fig, axes = plt.subplots(5, 4, figsize=(17, 20))

for row, map_path in enumerate(maps):
    name = os.path.basename(map_path).replace("_floodmap.tif", "")

    raw_path = os.path.join(RAW_DIR, name + "_S2Hand.tif")
    label_path = os.path.join(LABEL_DIR, name + "_LabelHand.tif")

    if not os.path.exists(raw_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing input or label for {name}")

    with rasterio.open(raw_path) as src:
        rgb = src.read([4, 3, 2]).astype(np.float32)

    with rasterio.open(label_path) as src:
        label = src.read(1)

    with rasterio.open(map_path) as src:
        prediction = src.read(1)

    if rgb.shape[1:] != label.shape or label.shape != prediction.shape:
        raise RuntimeError(f"Dimension mismatch: {name}")

    display_rgb = np.zeros_like(rgb)

    for band in range(3):
        values = rgb[band]
        finite = values[np.isfinite(values)]
        low, high = np.percentile(finite, [2, 98])
        if high > low:
            display_rgb[band] = np.clip(
                (values - low) / (high - low), 0, 1
            )

    display_rgb = np.moveaxis(display_rgb, 0, -1)

    valid = np.isin(label, [0, 1]) & np.isin(prediction, [0, 1])

    gt_display = np.full(label.shape, 2, dtype=np.uint8)
    pred_display = np.full(label.shape, 2, dtype=np.uint8)

    gt_display[valid] = label[valid]
    pred_display[valid] = prediction[valid]

    error = np.full(label.shape, 4, dtype=np.uint8)
    error[valid & (label == 0) & (prediction == 0)] = 0
    error[valid & (label == 1) & (prediction == 1)] = 1
    error[valid & (label == 0) & (prediction == 1)] = 2
    error[valid & (label == 1) & (prediction == 0)] = 3

    axes[row, 0].imshow(display_rgb)
    axes[row, 0].set_title(f"{name}\nSatellite RGB")

    axes[row, 1].imshow(
        gt_display,
        cmap=ListedColormap(["black", "white", "grey"]),
        vmin=0, vmax=2
    )
    axes[row, 1].set_title("Ground Truth")

    axes[row, 2].imshow(
        pred_display,
        cmap=ListedColormap(["black", "white", "grey"]),
        vmin=0, vmax=2
    )
    axes[row, 2].set_title("Prediction")

    error_display = np.ma.masked_where(error == 4, error)
    axes[row, 3].imshow(
        error_display, cmap=error_cmap, vmin=0, vmax=3
    )
    axes[row, 3].set_title("TN / TP / FP / FN")

    for ax in axes[row]:
        ax.set_xticks([])
        ax.set_yticks([])

fig.suptitle("AI Earth Guardian | Final Flood Validation", fontsize=17)
fig.legend(
    handles=[
        plt.Line2D([0], [0], color="black", lw=6, label="Background"),
        plt.Line2D([0], [0], color="white", lw=6, marker="s",
                   markeredgecolor="black", label="Flood"),
        plt.Line2D([0], [0], color="grey", lw=6, label="NoData"),
        plt.Line2D([0], [0], color="#555555", lw=6, label="True Negative"),
        plt.Line2D([0], [0], color="#16803c", lw=6, label="True Positive"),
        plt.Line2D([0], [0], color="#e53935", lw=6, label="False Positive"),
        plt.Line2D([0], [0], color="#facc15", lw=6, label="False Negative"),
    ],
    loc="lower center", ncol=3
)

fig.tight_layout(rect=[0, 0.07, 1, 0.97])

output_path = os.path.join(OUTPUT_DIR, "final_visual_report.png")
fig.savefig(output_path, dpi=180, bbox_inches="tight")
plt.close(fig)

print("PASS: Five images processed.")
print("Saved:", os.path.abspath(output_path))

