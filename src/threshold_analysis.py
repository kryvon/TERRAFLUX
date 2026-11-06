
import os
import glob
import numpy as np
import pandas as pd
import rasterio

PROB_DIR = "outputs/validation5/probabilities"
LABEL_DIR = "data/validation5/labels"
OUTPUT_DIR = "outputs/validation5"

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

def calculate_metrics(tp, fp, fn, tn):
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)

    return accuracy, precision, recall, f1, iou

prob_files = sorted(glob.glob(
    os.path.join(PROB_DIR, "*_floodprob.tif")
))

if len(prob_files) != 5:
    raise RuntimeError(f"Expected 5 probability maps, found {len(prob_files)}")

all_results = []

for threshold in THRESHOLDS:
    total_tp = total_fp = total_fn = total_tn = 0

    for prob_path in prob_files:
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
        truth = label[valid] == 1
        prediction = probability[valid] >= threshold

        tp = int(np.sum(prediction & truth))
        fp = int(np.sum(prediction & ~truth))
        fn = int(np.sum(~prediction & truth))
        tn = int(np.sum(~prediction & ~truth))

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    metrics = calculate_metrics(
        total_tp, total_fp, total_fn, total_tn
    )

    all_results.append({
        "Threshold": threshold,
        "Accuracy": metrics[0],
        "Precision": metrics[1],
        "Recall": metrics[2],
        "F1": metrics[3],
        "IoU": metrics[4],
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "TN": total_tn
    })

df = pd.DataFrame(all_results)

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "threshold_analysis.csv")
df.to_csv(output_path, index=False)

print("\nTHRESHOLD ANALYSIS")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved:", output_path)

print("\nHighest F1 threshold:",
      df.loc[df["F1"].idxmax(), "Threshold"])
print("Highest IoU threshold:",
      df.loc[df["IoU"].idxmax(), "Threshold"])

