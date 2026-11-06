
import os
import csv
import math

OUTPUT_DIR = "outputs/validation5"
REPORT_DIR = os.path.join(OUTPUT_DIR, "final_report")
THRESHOLD_CSV = os.path.join(OUTPUT_DIR, "threshold_analysis.csv")

os.makedirs(REPORT_DIR, exist_ok=True)

# Verified aggregate results at threshold 0.30
TP, FP, FN, TN = 46204, 7813, 13153, 900600

def calculate_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0
    accuracy = (tp + tn) / total if total else 0

    return accuracy, precision, recall, f1, iou

accuracy, precision, recall, f1, iou = calculate_metrics(TP, FP, FN, TN)

threshold_rows = []
if not os.path.exists(THRESHOLD_CSV):
    raise FileNotFoundError(THRESHOLD_CSV)

with open(THRESHOLD_CSV, newline="") as file:
    threshold_rows = list(csv.DictReader(file))

if not threshold_rows:
    raise RuntimeError("Threshold analysis CSV is empty.")

best_row = max(threshold_rows, key=lambda row: float(row["F1"]))

image_counts = {
    "Ghana_1078550": (445, 357, 962, 260380),
    "Ghana_313799": (8725, 2809, 6208, 226722),
    "Ghana_319168": (34233, 3395, 5288, 128708),
    "Ghana_359826": (2801, 1252, 695, 257396),
    "Ghana_97059": (0, 0, 0, 27394),
}

lines = [
    "# AI Earth Guardian",
    "## Final Flood Detection Evaluation Report",
    "",
    "### Validation setup",
    "- Dataset: Sen1Floods11 validation samples",
    "- Validation images: 5",
    "- Location represented: Ghana",
    "- Model: Prithvi 100M",
    "- Inference: CPU",
    "- Input: six selected Sentinel-2 bands",
    "- Evaluation threshold: 0.30",
    "",
    "### Aggregate evaluation metrics",
    "| Metric | Result |",
    "|---|---:|",
    f"| Accuracy | {accuracy:.4f} |",
    f"| Flood precision | {precision:.4f} |",
    f"| Flood recall | {recall:.4f} |",
    f"| Flood F1-score | {f1:.4f} |",
    f"| Flood IoU | {iou:.4f} |",
    "",
    "### Confusion matrix counts",
    f"- True positives: {TP:,}",
    f"- False positives: {FP:,}",
    f"- False negatives: {FN:,}",
    f"- True negatives: {TN:,}",
    "",
    "### Per-image evaluation",
    "| Image | Precision | Recall | F1 | IoU |",
    "|---|---:|---:|---:|---:|",
]

for name, (tp, fp, fn, tn) in image_counts.items():
    _, p, r, image_f1, image_iou = calculate_metrics(tp, fp, fn, tn)
    lines.append(
        f"| {name} | {p:.4f} | {r:.4f} | {image_f1:.4f} | {image_iou:.4f} |"
    )

lines += [
    "",
    "### Threshold comparison",
    "| Threshold | Accuracy | Precision | Recall | F1 | IoU |",
    "|---:|---:|---:|---:|---:|---:|",
]

for row in threshold_rows:
    lines.append(
        f"| {float(row['Threshold']):.2f} "
        f"| {float(row['Accuracy']):.4f} "
        f"| {float(row['Precision']):.4f} "
        f"| {float(row['Recall']):.4f} "
        f"| {float(row['F1']):.4f} "
        f"| {float(row['IoU']):.4f} |"
    )

lines += [
    "",
    "### Threshold observation",
    f"- Best F1 among tested thresholds: {float(best_row['Threshold']):.2f}",
    f"- Corresponding F1: {float(best_row['F1']):.4f}",
    "- This is the best result among the tested values, not a universally optimal threshold.",
    "",
    "### Limitations",
    "- Evaluation uses only five validation images.",
    "- Performance varies substantially between images.",
    "- Cloud cover, image conditions and geographic variation may affect predictions.",
    "- Further testing across more regions and flood events is required.",
    "- This prototype is not a replacement for official flood warnings.",
    "",
    "### Conclusion",
    "The prototype successfully generated georeferenced flood probability maps "
    "and binary flood predictions for five validation images. Quantitative and "
    "visual evaluation was completed. Broader validation is required before "
    "operational use.",
]

report_path = os.path.join(REPORT_DIR, "final_evaluation_report.md")

with open(report_path, "w", encoding="utf-8") as file:
    file.write("\n".join(lines))

print("PASS: Evaluation report generated.")
print("Aggregate F1:", round(f1, 4))
print("Best tested threshold:", best_row["Threshold"])
print("Saved:", os.path.abspath(report_path))
