import os
import glob
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────
# input_dir: where the raw unprocessed images are (with weird suffixes like .jpg.chip.jpg)
# output_dir: where we’ll save the resized, padded grayscale versions
# size: final size of all images
# fill_color: background color for padding (0 = black)
input_dir  = "./data/raw"
output_dir = "./data/processed"
size       = (64, 64)
fill_color = 0

# make sure the output folder exists
os.makedirs(output_dir, exist_ok=True)

print("\nTop‐level contents of processed/:")
entries = os.listdir(input_dir)
for entry in entries[:20]:  # just preview first 20
    print(" ", entry)
if len(entries) > 20:
    print("  ...")

# we'll just check how the first file is named to guess the pattern
if not entries:
    raise RuntimeError(f"No files found in {input_dir}")

first_file = entries[0]
dot_index = first_file.find('.')  # look for first dot
if dot_index == -1:
    raise RuntimeError(f"First file '{first_file}' has no suffix to match.")

suffix = first_file[dot_index:]  # everything from the first dot onwards
print(f"\nDetected suffix from first file: {suffix!r}")

pattern = os.path.join(input_dir, f"*{suffix}")
print(f"\nUsing glob pattern: {pattern!r}")
file_paths = glob.glob(pattern)

print(f"Found {len(file_paths)} files matching '*{suffix}'")
if file_paths:
    print("First 10 matches:")
    for p in file_paths[:10]:
        print(" ", p)
else:
    print("No files found. Check suffix and directory structure.")
    exit(1)

for in_path in file_paths:
    fname = os.path.basename(in_path)
    out_path = os.path.join(output_dir, fname)

    with Image.open(in_path) as img:
        # turn image into grayscale
        img = img.convert("L")

        # resize the image to fit inside target size (keeps aspect ratio)
        img.thumbnail(size, resample=Image.LANCZOS)

        # create a black background canvas and paste image in the center
        background = Image.new("L", size, fill_color)
        offset_x = (size[0] - img.width)  // 2
        offset_y = (size[1] - img.height) // 2
        background.paste(img, (offset_x, offset_y))

        # save the final image as JPEG
        background.save(out_path, "JPEG", quality=95)

print(f"\nProcessed all images from {input_dir} → {output_dir}")
