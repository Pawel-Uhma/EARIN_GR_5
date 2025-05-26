import os
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────
input_dir  = "../data/raw"        # directory containing original .jpg files
output_dir = "../data/processed"  # directory where processed images will be saved
size       = (64, 64)        # target size (width, height)
fill_color = 0                 # padding color (black) for grayscale mode

# ─── Ensure output directory exists ──────────────────────────────
os.makedirs(output_dir, exist_ok=True)

# ─── Process each .jpg in the input directory ────────────────────
for fname in os.listdir(input_dir):
    if not fname.lower().endswith(".jpg"):
        continue

    in_path  = os.path.join(input_dir,  fname)
    out_path = os.path.join(output_dir, fname)

    with Image.open(in_path) as img:
        # Convert to grayscale
        img = img.convert("L")
        # Resize with high-quality downsampling
        img.thumbnail(size, resample=Image.LANCZOS)

        # Create a black (gray=0) background and paste centered
        background = Image.new("L", size, fill_color)
        offset_x = (size[0] - img.width)  // 2
        offset_y = (size[1] - img.height) // 2
        background.paste(img, (offset_x, offset_y))

        # Save the final grayscale image
        background.save(out_path, "JPEG", quality=95)

print(f"Processed all images from {input_dir} → {output_dir}")
