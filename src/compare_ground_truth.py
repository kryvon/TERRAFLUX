from pathlib import Path
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent

PREDICTION = ROOT / "outputs/app_predictions/flood_maps/sentinel2_selected_bands_corrected_flood_map.tif"

LABEL = ROOT / "data/Sen1Floods11-master/sample/Labels/Spain_7370579_LabelHand.tif"

with rasterio.open(PREDICTION) as pred_src:
    prediction = pred_src.read(1)
    pred_transform = pred_src.transform
    pred_crs = pred_src.crs

with rasterio.open(LABEL) as label_src:
    labels = label_src.read(1)
    label_transform = label_src.transform
    label_crs = label_src.crs

print("Prediction shape:", prediction.shape)
print("Label shape:", labels.shape)
print("Prediction CRS:", pred_crs)
print("Label CRS:", label_crs)
print("Label values:", np.unique(labels))
print("Prediction values:", np.unique(prediction))

if prediction.shape != labels.shape:
    raise ValueError("Image dimensions do not match.")

if pred_crs != label_crs or not np.allclose(
    pred_transform, label_transform, atol=1e-9
):
    raise ValueError("Prediction and label geospatial alignment differs.")

valid = np.isin(labels, [0, 1]) & np.isin(prediction, [0, 1])

y_true = labels[valid].astype(bool)
y_pred = prediction[valid].astype(bool)

tp = int(np.sum(y_true & y_pred))
fp = int(np.sum(~y_true & y_pred))
fn = int(np.sum(y_true & ~y_pred))
tn = int(np.sum(~y_true & ~y_pred))

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
iou = tp / (tp + fp + fn) if tp + fp + fn else 0
accuracy = (tp + tn) / len(y_true) if len(y_true) else 0

print("\n--- GROUND TRUTH EVALUATION ---")
print("Evaluated pixels:", len(y_true))
print("TP:", tp, "| FP:", fp, "| FN:", fn, "| TN:", tn)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"IoU:       {iou:.4f}")