import os
import glob
import numpy as np
import rasterio

MAP_DIR = "outputs/validation5/flood_maps"
PROB_DIR = "outputs/validation5/probabilities"

maps = sorted(glob.glob(os.path.join(MAP_DIR, "*_floodmap.tif")))

if len(maps) != 5:
    raise RuntimeError(f"Expected 5 flood maps, found {len(maps)}")

all_passed = True

for map_path in maps:
    name = os.path.basename(map_path).replace("_floodmap.tif", "")
    prob_path = os.path.join(PROB_DIR, name + "_floodprob.tif")

    with rasterio.open(map_path) as flood_src, rasterio.open(prob_path) as prob_src:
        flood = flood_src.read(1)

        valid_classes = np.isin(flood, [0, 1, 255])
        same_shape = flood_src.shape == prob_src.shape
        same_crs = flood_src.crs == prob_src.crs
        same_transform = np.allclose(
            tuple(flood_src.transform),
            tuple(prob_src.transform),
            atol=1e-9
        )

        passed = (
            valid_classes.all()
            and same_shape
            and same_crs
            and same_transform
            and flood_src.nodata == 255
        )

        all_passed = all_passed and passed

        print(
            name,
            "|", "PASS" if passed else "FAIL",
            "| Shape:", flood_src.shape,
            "| CRS:", flood_src.crs,
            "| Classes:", np.unique(flood)
        )

print("\nOVERALL:", "ALL CHECKS PASSED" if all_passed else "CHECK FAILED")
