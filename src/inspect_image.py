
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

project_folder = Path(__file__).resolve().parent.parent
image_path = project_folder / "data" / "satellite.jpg"

if not image_path.exists():
    print("Image not found:", image_path)
else:
    image = Image.open(image_path).convert("RGB")

    print("Image loaded successfully!")
    print("Image size:", image.size)

    plt.imshow(image)
    plt.title("Satellite Image")
    plt.axis("off")
    plt.show() 

