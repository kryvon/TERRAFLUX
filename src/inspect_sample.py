
from pathlib import Path
import rasterio

root = Path(__file__).resolve().parent.parent
sample = root / "data" / "Sen1Floods11-master" / "sample"

for folder in ["S1", "S2", "Labels"]:
    files = list((sample / folder).glob("*.tif"))

    for file in files:
        with rasterio.open(file) as image:
            print(f"\n{folder}: {file.name}")
            print("Dimensions:", image.width, "x", image.height)
            print("Bands:", image.count)
            print("Data type:", image.dtypes)
            print("CRS:", image.crs)
            print("No-data value:", image.nodata)
