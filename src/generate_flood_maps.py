

import os
import glob
import numpy as np
import rasterio

PROB_DIR = "outputs/validation5/probabilities"
OUTPUT_DIR = "outputs/validation5/flood_maps"
THRESHOLD = 0.30

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not 0 <= THRESHOLD <= 1:
    raise ValueError("THRESHOLD must be between 0 and 1.")

files = sorted(
    glob.glob(os.path.join(PROB_DIR, "*_floodprob.tif"))
)

if len(files) != 5:
    raise RuntimeError(
        f"Expected 5 probability maps, found {len(files)}"
    )

for path in files:
    name = os.path.basename(path).replace(
        "_floodprob.tif", ""
    )

    with rasterio.open(path) as src:
        probability = src.read(1, masked=True)
        profile = src.profile.copy()

    values = np.asarray(probability.data)

    valid = (
        ~np.ma.getmaskarray(probability)
        & np.isfinite(values)
        & (values >= 0)
        & (values <= 1)
    )

    if not np.any(valid):
        raise RuntimeError(
            f"No valid probability pixels found: {name}"
        )

    flood_map = np.full(
        values.shape, 255, dtype=np.uint8
    )

    flood_map[valid] = (
        values[valid] >= THRESHOLD
    ).astype(np.uint8)

    profile.update(
        count=1,
        dtype="uint8",
        nodata=255,
        compress="lzw"
    )

    output_path = os.path.join(
        OUTPUT_DIR, name + "_floodmap.tif"
    )

    with rasterio.open(
        output_path, "w", **profile
    ) as dst:
        dst.write(flood_map, 1)

    print(
        name,
        "| Flood:", int(np.sum(flood_map == 1)),
        "| Background:", int(np.sum(flood_map == 0)),
        "| NoData:", int(np.sum(flood_map == 255))
    )

print(
    "\nDONE. Flood maps saved:",
    os.path.abspath(OUTPUT_DIR)
)
