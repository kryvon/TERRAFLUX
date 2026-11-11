
import os
import sys
import glob
import numpy as np
import torch
import rasterio

from mmcv import Config
from mmcv.parallel import collate
from mmseg.apis import init_segmentor
from mmseg.datasets.pipelines import Compose

sys.path.insert(0, os.path.abspath("hls-foundation-os"))
import model_inference as mi


CONFIG = "hls-foundation-os/configs/sen1floods11_config.py"
CHECKPOINT = "sen1floods11_Prithvi_100M.pth"
INPUT_DIR = "data/validation5/input"
OUTPUT_DIR = "outputs/validation5/probabilities"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cfg = Config.fromfile(CONFIG)
cfg.model.backbone.pretrained = None

model = init_segmentor(cfg, CHECKPOINT, device="cpu")
model.eval()

pipeline_cfg = mi.process_test_pipeline(
    model.cfg.data.test.pipeline,
    bands=[0, 1, 2, 3, 4, 5]
)
pipeline = Compose(pipeline_cfg)

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tif")))

if not files:
    raise RuntimeError(f"No TIFF inputs found in {INPUT_DIR}")

print("Images:", len(files))
print("Device: CPU")
print("Output:", os.path.abspath(OUTPUT_DIR))

for path in files:
    name = os.path.basename(path).replace("_S2Hand.tif", "")
    out_path = os.path.join(
        OUTPUT_DIR, name + "_floodprob.tif"
    )

    print(f"\nProcessing {name}")

    data = pipeline({"img_info": {"filename": path}})
    batch = collate([data], samples_per_gpu=1)

    # With this MMCV pipeline, collate() keeps the image tensor in a
    # one-element list, while metadata is wrapped in a DataContainer.
    img = batch["img"][0]
    img_metas = batch["img_metas"].data[0][0]

    with torch.no_grad():
        probabilities = model.inference(
            img, img_metas, rescale=True
        )

        if not torch.is_tensor(probabilities):
            raise TypeError(
                f"Expected tensor, got {type(probabilities).__name__}"
            )

        if probabilities.ndim != 4 or probabilities.shape[1] != 2:
            raise RuntimeError(
                f"Unexpected output shape: {tuple(probabilities.shape)}"
            )

        # TemporalEncoderDecoder.inference() returns probabilities, not logits.
        flood_probability = probabilities[0, 1]
        probability = flood_probability.detach().cpu().numpy().astype(np.float32)

    with rasterio.open(path) as src:
        profile = src.profile.copy()
        valid_mask = np.all(src.read_masks(), axis=0) > 0
        if src.nodata is not None:
            raw = src.read()
            valid_mask &= np.all(raw != src.nodata, axis=0)
        valid_mask &= np.all(np.isfinite(src.read()), axis=0)

        if probability.shape != (src.height, src.width):
            raise RuntimeError(
                f"{name}: probability shape {probability.shape} "
                f"doesn't match input {(src.height, src.width)}"
            )

        if not np.all(np.isfinite(probability[valid_mask])):
            raise RuntimeError(f"{name}: non-finite probabilities on valid pixels")
        if np.any((probability[valid_mask] < 0.0) | (probability[valid_mask] > 1.0)):
            raise RuntimeError(f"{name}: probabilities outside [0, 1]")
        probability[~valid_mask] = -1.0

        profile.update(
            count=1,
            dtype="float32",
            nodata=-1.0,
            compress="lzw"
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(probability, 1)

    print("Saved:", out_path)
    print(
        "Probability min/max/mean:",
        round(float(probability[valid_mask].min()), 4),
        round(float(probability[valid_mask].max()), 4),
        round(float(probability[valid_mask].mean()), 4)
    )

print("\nDONE: probability maps exported.")
