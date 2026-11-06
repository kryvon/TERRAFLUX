
import os
import glob
import numpy as np
import rasterio

PROB_DIR = "outputs/validation5/probabilities"
LABEL_DIR = "data/validation5/labels"
INPUT_DIR = "data/validation5/input"

files = sorted(glob.glob(os.path.join(PROB_DIR, "*_floodprob.tif")))

if len(files) != 5:
    raise RuntimeError(f"Expected 5 probability maps, found {len(files)}")

overall = True

for prob_path in files:
    name = os.path.basename(prob_path).replace("_floodprob.tif", "")

    paths = [
        os.path.join(INPUT_DIR, name + "_S2Hand.tif"),
        prob_path,
        os.path.join(LABEL_DIR, name + "_LabelHand.tif")
    ]

    with rasterio.open(paths[0]) as image, \
         rasterio.open(paths[1]) as prob, \
         rasterio.open(paths[2]) as label:

        same_grid = (
            image.crs == prob.crs == label.crs
            and image.width == prob.width == label.width
            and image.height == prob.height == label.height
            and np.allclose(
                tuple(image.transform),
                tuple(prob.transform),
                atol=1e-9
            )
            and np.allclose(
                tuple(image.transform),
                tuple(label.transform),
                atol=1e-9
            )
        )

        print(name, ":", "PASS" if same_grid else "FAIL")
        overall = overall and same_grid

print("\nOVERALL:", "ALL CHECKS PASSED" if overall else "MISMATCH FOUND")
