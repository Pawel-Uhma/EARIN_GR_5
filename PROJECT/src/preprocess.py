import os
import glob
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────
# Directory containing the unprocessed "chip" images with unknown suffix,
# e.g., something like "0001_0_0_201701011234.jpg.chip.jpg"
input_dir  = "./data/raw"  # directory where files with “.jpg.chip.jpg” (or similar) live
output_dir = "./data/processed"    # directory where resized grayscale images will be saved
size       = (64, 64)             # target size (width, height)
fill_color = 0                    # padding color (black) for grayscale mode

# ─── Ensure output directory exists ──────────────────────────────
os.makedirs(output_dir, exist_ok=True)

# ─── 1) List top‐level contents ───────────────────────────────────
print("\nTop‐level contents of processed/:")
entries = os.listdir(input_dir)
for entry in entries[:20]:
    print(" ", entry)
if len(entries) > 20:
    print("  ...")

# ─── 2) Determine suffix from the first file ──────────────────────
if not entries:
    raise RuntimeError(f"No files found in {input_dir}")

first_file = entries[0]
dot_index = first_file.find('.')
if dot_index == -1:
    raise RuntimeError(f"First file '{first_file}' has no suffix to match.")

suffix = first_file[dot_index:]  # e.g. ".jpg.chip.jpg"
print(f"\nDetected suffix from first file: {suffix!r}")

# ─── 3) Glob for all files ending with that suffix ────────────────
pattern = os.path.join(input_dir, f"*{suffix}")
print(f"\nUsing glob pattern: {pattern!r}")
file_paths = glob.glob(pattern)

# ─── 4) Log how many matching files were found ────────────────────
print(f"Found {len(file_paths)} files matching '*{suffix}'")
if file_paths:
    print("First 10 matches:")
    for p in file_paths[:10]:
        print(" ", p)
else:
    print("No files found. Check suffix and directory structure.")
    exit(1)

# ─── 5) Process each matched file ─────────────────────────────────
for in_path in file_paths:
    fname = os.path.basename(in_path)
    out_path = os.path.join(output_dir, fname)

    with Image.open(in_path) as img:
        # Convert to grayscale
        img = img.convert("L")
        # Resize with high‐quality downsampling
        img.thumbnail(size, resample=Image.LANCZOS)

        # Create a black (gray=0) background and paste centered
        background = Image.new("L", size, fill_color)
        offset_x = (size[0] - img.width)  // 2
        offset_y = (size[1] - img.height) // 2
        background.paste(img, (offset_x, offset_y))

        # Save the final grayscale image
        background.save(out_path, "JPEG", quality=95)

print(f"\nProcessed all images from {input_dir} → {output_dir}")
