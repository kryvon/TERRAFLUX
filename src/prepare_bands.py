

import rasterio
from pathlib import Path

source = Path(
    "data/Sen1Floods11-master/sample/S2/Spain_7370579_S2Hand.tif"
)

output = Path("data/sentinel2_selected_bands.tif")

# Correct Rasterio indexes:
# B2, B3, B4, B8A, B11, B12
selected_bands = [2, 3, 4, 9, 12, 13]

with rasterio.open(source) as src:
    profile = src.profile.copy()
    profile.update(
        count=6,
        dtype="int16"
    )

    with rasterio.open(output, "w", **profile) as dst:
        for out_index, band_index in enumerate(selected_bands, start=1):
            dst.write(src.read(band_index), out_index)

print("Corrected six-band image saved:", output)
print("Band order: B2, B3, B4, B8A, B11, B12")
 